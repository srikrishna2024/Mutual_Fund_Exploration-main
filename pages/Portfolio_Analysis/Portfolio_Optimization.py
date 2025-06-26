import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import psycopg
from scipy.optimize import minimize
import plotly.express as px
from datetime import datetime
from scipy.optimize import newton

# -------------------- DATABASE CONFIG --------------------

DB_PARAMS = {
    'dbname': 'postgres',
    'user': 'postgres',
    'password': 'admin123',
    'host': 'localhost',
    'port': '5432'
}

def get_db_connection():
    return psycopg.connect(**DB_PARAMS)

# -------------------- UTILITY FUNCTIONS --------------------

def format_indian_currency(value):
    """Format numbers in Indian style (lakhs, crores)"""
    if pd.isna(value):
        return "₹0"
    
    value = float(value)
    if value < 100000:
        return f"₹{value:,.2f}"
    elif value < 10000000:
        lakhs = value / 100000
        return f"₹{lakhs:,.2f} L"
    else:
        crores = value / 10000000
        return f"₹{crores:,.2f} Cr"

# -------------------- PORTFOLIO ANALYSIS FUNCTIONS --------------------

def get_portfolio_data():
    """Retrieve all records from portfolio_data table"""
    with get_db_connection() as conn:
        query = """
            SELECT date, scheme_name, code, transaction_type, value, units, amount 
            FROM portfolio_data 
            ORDER BY date, scheme_name
        """
        return pd.read_sql(query, conn)

def get_latest_nav():
    """Retrieve the latest NAVs from mutual_fund_nav table"""
    with get_db_connection() as conn:
        query = """
            SELECT code, value AS nav_value
            FROM mutual_fund_nav
            WHERE (code, nav) IN (
                SELECT code, MAX(nav) AS nav_date
                FROM mutual_fund_nav
                GROUP BY code
            )
        """
        return pd.read_sql(query, conn)

def get_goal_mappings():
    """Retrieve goal mappings from the goals table including both MF and debt investments"""
    with get_db_connection() as conn:
        query = """
            WITH mf_latest_values AS (
                SELECT g.goal_name, g.investment_type, g.scheme_name, g.scheme_code,
                       CASE 
                           WHEN g.is_manual_entry THEN g.current_value
                           ELSE COALESCE(p.units * n.value, 0)
                       END as current_value
                FROM goals g
                LEFT JOIN (
                    SELECT scheme_name, code,
                           SUM(CASE 
                               WHEN transaction_type IN ('switch_out', 'redeem') THEN -units
                               WHEN transaction_type IN ('invest', 'switch_in') THEN units
                               ELSE 0 
                           END) as units
                    FROM portfolio_data
                    GROUP BY scheme_name, code
                ) p ON g.scheme_code = p.code
                LEFT JOIN (
                    SELECT code, value
                    FROM mutual_fund_nav
                    WHERE (code, nav) IN (
                        SELECT code, MAX(nav)
                        FROM mutual_fund_nav
                        GROUP BY code
                    )
                ) n ON g.scheme_code = n.code
            )
            SELECT goal_name, investment_type, scheme_name, scheme_code, current_value
            FROM mf_latest_values
            ORDER BY goal_name, investment_type, scheme_name
        """
        return pd.read_sql(query, conn)

def prepare_cashflows(df):
    """Prepare cashflow data from portfolio transactions"""
    df['cashflow'] = df.apply(lambda x: 
        -x['amount'] if x['transaction_type'] in ('invest', 'switch_in')  # Negative because it's money going out
        else x['amount'] if x['transaction_type'] in ('redeem', 'switch_out')  # Positive because it's money coming in
        else 0, 
        axis=1
    )
    return df

def calculate_units(df):
    """Calculate net units for each scheme based on transactions"""
    df['units_change'] = df.apply(lambda x: 
        x['units'] if x['transaction_type'] in ('invest', 'switch_in')
        else -x['units'] if x['transaction_type'] in ('redeem', 'switch_out')
        else 0,
        axis=1
    )
    
    # Calculate cumulative units for each scheme
    df = df.sort_values(['scheme_name', 'date'])
    df['cumulative_units'] = df.groupby(['scheme_name', 'code'])['units_change'].cumsum()
    return df

def xirr(transactions):
    """Calculate XIRR given a set of transactions"""
    if len(transactions) < 2:
        return None

    def xnpv(rate):
        first_date = pd.to_datetime(transactions['date'].min())
        days = [(pd.to_datetime(date) - first_date).days for date in transactions['date']]
        return sum([cf * (1 + rate) ** (-d/365.0) for cf, d in zip(transactions['cashflow'], days)])

    def xnpv_der(rate):
        first_date = pd.to_datetime(transactions['date'].min())
        days = [(pd.to_datetime(date) - first_date).days for date in transactions['date']]
        # Ensure rate is valid to avoid invalid power operations
        if (1 + rate) <= 0:
            return np.inf  # Return a large value to avoid invalid rates
        return sum([cf * (-d/365.0) * (1 + rate) ** (-d/365.0 - 1) 
                   for cf, d in zip(transactions['cashflow'], days)])

    try:
        return newton(xnpv, x0=0.1, fprime=xnpv_der, maxiter=1000)
    except:
        return None

def calculate_portfolio_weights(df, latest_nav, goal_mappings):
    """Calculate current portfolio weights for each scheme including debt investments"""
    # First calculate MF values by getting the latest units for each scheme
    mf_df = df.groupby(['scheme_name', 'code']).agg({
        'cumulative_units': 'last'
    }).reset_index()

    mf_df = mf_df.merge(latest_nav, on='code', how='left')
    mf_df['current_value'] = mf_df['cumulative_units'] * mf_df['nav_value']

    # Get debt investments from goal mappings, excluding any that are already in MF investments
    debt_investments = goal_mappings[
        (goal_mappings['investment_type'] == 'Debt') & 
        (~goal_mappings['scheme_name'].isin(mf_df['scheme_name']))
    ]
    
    # Combine MF and debt investments
    combined_df = pd.concat([
        mf_df[['scheme_name', 'current_value']],
        debt_investments[['scheme_name', 'current_value']]
    ])

    # Group by scheme_name to handle any remaining duplicates
    combined_df = combined_df.groupby('scheme_name')['current_value'].sum().reset_index()

    total_value = combined_df['current_value'].sum()
    combined_df['weight'] = (combined_df['current_value'] / total_value) * 100 if total_value > 0 else 0

    return combined_df

# -------------------- PORTFOLIO OPTIMIZATION FUNCTIONS --------------------

def calculate_rolling_returns(df_nav_pivot, window_days):
    """Calculate rolling returns for given window"""
    if len(df_nav_pivot) < window_days:
        # If not enough data, calculate simple returns
        return df_nav_pivot.pct_change(periods=min(window_days, len(df_nav_pivot)//2)).dropna()
    
    rolling_returns = (df_nav_pivot.pct_change(periods=window_days) + 1).pow(252/window_days) - 1
    return rolling_returns.dropna()

def calculate_consistency_score(returns_series):
    """Calculate consistency score - higher score means more consistent returns"""
    if len(returns_series) == 0 or returns_series.std() == 0:
        return 0
    
    mean_return = returns_series.mean()
    std_return = returns_series.std()
    
    # Consistency score: reward positive returns and penalize high volatility
    # Formula: (Mean Return / Standard Deviation) * (1 + Mean Return if positive, else penalty)
    sharpe_like = mean_return / std_return if std_return > 0 else 0
    
    # Bonus for positive returns, penalty for negative
    return_bonus = max(0, mean_return) * 2 - abs(min(0, mean_return)) * 3
    
    # Final score combines risk-adjusted return with return quality
    consistency_score = sharpe_like + return_bonus
    
    return consistency_score

def calculate_portfolio_metrics(returns, weights):
    """Calculate portfolio return, volatility, and consistency score"""
    if len(returns) == 0 or len(weights) != len(returns.columns):
        return 0, 0, 0
    
    # Portfolio returns
    portfolio_returns = (returns * weights).sum(axis=1)
    
    # Metrics
    portfolio_return = portfolio_returns.mean()
    portfolio_volatility = portfolio_returns.std()
    portfolio_consistency = calculate_consistency_score(portfolio_returns)
    
    return portfolio_return, portfolio_volatility, portfolio_consistency

def optimize_portfolio_allocation(returns_df, risk_tolerance='moderate'):
    """
    Optimize portfolio allocation based on consistency and diversification
    
    Parameters:
    - returns_df: DataFrame of fund returns
    - risk_tolerance: 'conservative', 'moderate', 'aggressive'
    
    Returns:
    - optimal_weights: Dictionary of fund codes and their allocation percentages
    - portfolio_metrics: Dictionary of portfolio performance metrics
    """
    
    if len(returns_df.columns) < 2:
        return None, None
    
    n_assets = len(returns_df.columns)
    fund_codes = returns_df.columns.tolist()
    
    # Calculate individual fund metrics
    fund_returns = returns_df.mean()
    fund_volatilities = returns_df.std()
    fund_consistency = {code: calculate_consistency_score(returns_df[code]) for code in fund_codes}
    
    # Correlation matrix
    corr_matrix = returns_df.corr()
    
    def objective_function(weights):
        """
        Objective function that balances return, consistency, and diversification
        """
        weights = np.array(weights)
        
        # Portfolio metrics
        portfolio_return, portfolio_volatility, portfolio_consistency = calculate_portfolio_metrics(returns_df, weights)
        
        # Diversification score (lower correlation is better)
        diversification_score = 0
        for i in range(n_assets):
            for j in range(i+1, n_assets):
                diversification_score += weights[i] * weights[j] * abs(corr_matrix.iloc[i, j])
        
        # Risk tolerance parameters
        if risk_tolerance == 'conservative':
            return_weight = 0.2
            consistency_weight = 0.5
            diversification_weight = 0.3
        elif risk_tolerance == 'moderate':
            return_weight = 0.4
            consistency_weight = 0.4
            diversification_weight = 0.2
        else:  # aggressive
            return_weight = 0.6
            consistency_weight = 0.2
            diversification_weight = 0.2
        
        # Maximize: (return * consistency) - (volatility + correlation penalty)
        # We minimize the negative of this
        objective = -(
            return_weight * portfolio_return + 
            consistency_weight * portfolio_consistency - 
            diversification_weight * diversification_score -
            0.1 * portfolio_volatility  # Small volatility penalty
        )
        
        return objective
    
    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},  # Weights sum to 1
    ]
    
    # Bounds (each weight between 5% and 70% to ensure diversification)
    min_weight = 0.05
    max_weight = 0.70
    bounds = [(min_weight, max_weight) for _ in range(n_assets)]
    
    # Initial guess (equal weights)
    initial_weights = np.array([1.0/n_assets] * n_assets)
    
    # Optimization
    try:
        result = minimize(
            objective_function, 
            initial_weights, 
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if result.success:
            optimal_weights = dict(zip(fund_codes, result.x))
            
            # Calculate portfolio metrics with optimal weights
            portfolio_return, portfolio_volatility, portfolio_consistency = calculate_portfolio_metrics(
                returns_df, result.x
            )
            
            # Calculate portfolio Sharpe ratio
            portfolio_sharpe = (portfolio_return - 0.06) / portfolio_volatility if portfolio_volatility > 0 else 0
            
            portfolio_metrics = {
                'return': portfolio_return,
                'volatility': portfolio_volatility,
                'consistency_score': portfolio_consistency,
                'sharpe_ratio': portfolio_sharpe
            }
            
            return optimal_weights, portfolio_metrics
        else:
            return None, None
            
    except Exception as e:
        st.error(f"Optimization failed: {e}")
        return None, None

def get_alternative_allocations(returns_df):
    """Generate alternative allocation strategies"""
    fund_codes = returns_df.columns.tolist()
    n_assets = len(fund_codes)
    
    strategies = {}
    
    # Equal Weight Strategy
    equal_weights = {code: 1.0/n_assets for code in fund_codes}
    equal_return, equal_vol, equal_consistency = calculate_portfolio_metrics(
        returns_df, list(equal_weights.values())
    )
    strategies['Equal Weight'] = {
        'weights': equal_weights,
        'return': equal_return,
        'volatility': equal_vol,
        'consistency': equal_consistency,
        'sharpe': (equal_return - 0.06) / equal_vol if equal_vol > 0 else 0
    }
    
    # Consistency-Weighted Strategy
    fund_consistency = {code: calculate_consistency_score(returns_df[code]) for code in fund_codes}
    total_consistency = sum(max(0, score) for score in fund_consistency.values())
    
    if total_consistency > 0:
        consistency_weights = {
            code: max(0, score) / total_consistency 
            for code, score in fund_consistency.items()
        }
        cons_return, cons_vol, cons_consistency = calculate_portfolio_metrics(
            returns_df, list(consistency_weights.values())
        )
        strategies['Consistency-Weighted'] = {
            'weights': consistency_weights,
            'return': cons_return,
            'volatility': cons_vol,
            'consistency': cons_consistency,
            'sharpe': (cons_return - 0.06) / cons_vol if cons_vol > 0 else 0
        }
    
    # Low Correlation Strategy (inverse correlation weighting)
    corr_matrix = returns_df.corr()
    avg_correlations = corr_matrix.mean()
    inverse_corr_weights = {}
    total_inverse_corr = sum(1 / (1 + max(0, corr)) for corr in avg_correlations)
    
    for code in fund_codes:
        inverse_corr_weights[code] = (1 / (1 + max(0, avg_correlations[code]))) / total_inverse_corr
    
    inv_return, inv_vol, inv_consistency = calculate_portfolio_metrics(
        returns_df, list(inverse_corr_weights.values())
    )
    strategies['Low Correlation'] = {
        'weights': inverse_corr_weights,
        'return': inv_return,
        'volatility': inv_vol,
        'consistency': inv_consistency,
        'sharpe': (inv_return - 0.06) / inv_vol if inv_vol > 0 else 0
    }
    
    return strategies

def plot_allocation_pie_chart(weights_dict, title="Portfolio Allocation"):
    """Create a pie chart for portfolio allocation"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    labels = list(weights_dict.keys())
    sizes = [weights_dict[label] * 100 for label in labels]
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%', 
                                      colors=colors, startangle=90)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Improve text readability
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    plt.tight_layout()
    return fig

def plot_strategy_comparison(strategies):
    """Plot comparison of different allocation strategies"""
    strategy_names = list(strategies.keys())
    returns = [strategies[name]['return'] for name in strategy_names]
    volatilities = [strategies[name]['volatility'] for name in strategy_names]
    consistency_scores = [strategies[name]['consistency'] for name in strategy_names]
    sharpe_ratios = [strategies[name]['sharpe'] for name in strategy_names]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Returns comparison
    bars1 = ax1.bar(strategy_names, [r*100 for r in returns], color='green', alpha=0.7)
    ax1.set_title('Expected Annual Returns (%)', fontweight='bold')
    ax1.set_ylabel('Return (%)')
    for i, bar in enumerate(bars1):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{returns[i]*100:.1f}%', ha='center', va='bottom')
    
    # Volatility comparison
    bars2 = ax2.bar(strategy_names, [v*100 for v in volatilities], color='red', alpha=0.7)
    ax2.set_title('Volatility (%)', fontweight='bold')
    ax2.set_ylabel('Volatility (%)')
    for i, bar in enumerate(bars2):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{volatilities[i]*100:.1f}%', ha='center', va='bottom')
    
    # Consistency scores
    bars3 = ax3.bar(strategy_names, consistency_scores, color='blue', alpha=0.7)
    ax3.set_title('Consistency Scores', fontweight='bold')
    ax3.set_ylabel('Consistency Score')
    for i, bar in enumerate(bars3):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{consistency_scores[i]:.2f}', ha='center', va='bottom')
    
    # Sharpe ratios
    bars4 = ax4.bar(strategy_names, sharpe_ratios, color='purple', alpha=0.7)
    ax4.set_title('Sharpe Ratios', fontweight='bold')
    ax4.set_ylabel('Sharpe Ratio')
    for i, bar in enumerate(bars4):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{sharpe_ratios[i]:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    return fig

def get_top_performing_funds(selected_categories, top_n=10, rolling_period_years=5):
    """Get top N most consistent performing funds from selected categories"""
    with get_db_connection() as conn:
        # First get all funds from selected categories
        funds_query = """
            SELECT DISTINCT code, scheme_name, scheme_category
            FROM mutual_fund_master_data
            WHERE scheme_category = ANY(%s)
        """
        df_funds = pd.read_sql(funds_query, conn, params=(selected_categories,))
        
        if df_funds.empty:
            return None, None, None, f"No funds found for categories: {selected_categories}"
        
        fund_codes = df_funds['code'].tolist()
        
        # Get NAV data for these funds
        nav_query = """
            SELECT nav, code, value
            FROM mutual_fund_nav
            WHERE code = ANY(%s)
            ORDER BY nav ASC
        """
        df_nav = pd.read_sql(nav_query, conn, params=(fund_codes,))
        
        if df_nav.empty:
            return None, None, None, f"No NAV data found for {len(fund_codes)} funds"
    
    # Pivot NAV data
    df_nav_pivot = df_nav.pivot(index='nav', columns='code', values='value')
    
    # Remove columns with too much missing data (keep only funds with at least 70% data)
    min_data_points = int(0.7 * len(df_nav_pivot))
    df_nav_pivot = df_nav_pivot.dropna(thresh=min_data_points, axis=1)
    
    # Forward fill and backward fill remaining missing values
    df_nav_pivot = df_nav_pivot.fillna(method='ffill').fillna(method='bfill')
    
    if df_nav_pivot.empty:
        return None, None, None, "No funds with sufficient data after cleaning"
    
    # Calculate required data points for rolling returns
    window = int(252 * rolling_period_years)
    if len(df_nav_pivot) < window:
        # Adjust window size if not enough data
        window = max(252, len(df_nav_pivot) // 2)  # At least 1 year or half available data
        actual_years = window / 252
        st.warning(f"⚠️ Adjusted rolling period to {actual_years:.1f} years due to limited data availability")
    
    # Calculate rolling returns
    rolling_returns = calculate_rolling_returns(df_nav_pivot, window)
    
    if rolling_returns.empty:
        return None, None, None, f"Could not calculate rolling returns with window {window}"
    
    # Calculate consistency metrics for each fund
    fund_metrics = []
    
    for fund_code in rolling_returns.columns:
        fund_returns = rolling_returns[fund_code].dropna()
        
        if len(fund_returns) < 10:  # Need at least 10 data points
            continue
            
        latest_return = fund_returns.iloc[-1] if len(fund_returns) > 0 else 0
        avg_return = fund_returns.mean()
        return_std = fund_returns.std()
        consistency_score = calculate_consistency_score(fund_returns)
        
        # Calculate additional metrics
        positive_return_ratio = (fund_returns > 0).sum() / len(fund_returns)
        max_drawdown = (fund_returns.cumsum().expanding().max() - fund_returns.cumsum()).max()
        
        fund_metrics.append({
            'code': fund_code,
            'latest_return': latest_return,
            'avg_return': avg_return,
            'return_std': return_std,
            'consistency_score': consistency_score,
            'positive_ratio': positive_return_ratio,
            'max_drawdown': max_drawdown,
            'data_points': len(fund_returns)
        })
    
    if not fund_metrics:
        return None, None, None, "No funds with sufficient data for consistency analysis"
    
    # Convert to DataFrame for easier processing
    metrics_df = pd.DataFrame(fund_metrics)
    
    # Filter funds with reasonable performance (avg return > -10% and std < 50%)
    filtered_metrics = metrics_df[
        (metrics_df['avg_return'] > -0.10) & 
        (metrics_df['return_std'] < 0.50) &
        (metrics_df['data_points'] >= 20)  # At least 20 data points
    ].copy()
    
    if filtered_metrics.empty:
        # If no funds pass strict criteria, relax them
        filtered_metrics = metrics_df[metrics_df['data_points'] >= 10].copy()
    
    # Rank by consistency score (higher is better)
    filtered_metrics = filtered_metrics.sort_values('consistency_score', ascending=False)
    
    # Get top N consistent funds
    actual_top_n = min(top_n, len(filtered_metrics))
    top_funds_metrics = filtered_metrics.head(actual_top_n)
    top_fund_codes = top_funds_metrics['code'].tolist()
    
    # Get fund details
    fund_details = df_funds[df_funds['code'].isin(top_fund_codes)].set_index('code')
    
    # Add metrics to fund details
    for _, row in top_funds_metrics.iterrows():
        code = row['code']
        if code in fund_details.index:
            fund_details.loc[code, 'latest_return'] = row['latest_return']
            fund_details.loc[code, 'avg_return'] = row['avg_return']
            fund_details.loc[code, 'return_std'] = row['return_std']
            fund_details.loc[code, 'consistency_score'] = row['consistency_score']
            fund_details.loc[code, 'positive_ratio'] = row['positive_ratio']
    
    return top_fund_codes, fund_details, df_nav_pivot[top_fund_codes], None

def plot_correlation_heatmap(corr_matrix, title="Correlation Heatmap"):
    """Plot correlation heatmap"""
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax, 
                square=True, linewidths=0.5)
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    return fig

def plot_rolling_returns(rolling_returns, fund_details, rolling_period_years, title="Rolling Returns"):
    """Plot rolling returns over time"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for code in rolling_returns.columns:
        fund_name = fund_details.loc[code, 'scheme_name'][:30] + "..." if len(fund_details.loc[code, 'scheme_name']) > 30 else fund_details.loc[code, 'scheme_name']
        ax.plot(rolling_returns.index, rolling_returns[code], label=f"{code}: {fund_name}", linewidth=2)
    
    ax.set_title(f"{rolling_period_years}-Year {title} (Annualized)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Annualized Return", fontsize=12)
    ax.set_xlabel("Date", fontsize=12)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

# -------------------- STREAMLIT APP --------------------

st.set_page_config(page_title="Portfolio Optimizer", layout="wide")
st.title("📊 Enhanced Mutual Fund Portfolio Optimizer")
st.markdown("*Prioritizing consistent returns over high volatility*")

# Sidebar for rolling period selection
st.sidebar.header("⚙️ Configuration")
rolling_period_years = st.sidebar.slider("Rolling Period (Years)", min_value=1, max_value=10, value=5, step=1)

# Load categories
with get_db_connection() as conn:
    categories_df = pd.read_sql(
        "SELECT DISTINCT scheme_category FROM mutual_fund_master_data WHERE scheme_category IS NOT NULL",
        conn
    )
categories = sorted(categories_df['scheme_category'].dropna().unique().tolist())

# Main interface with 3 tabs
tab1, tab2, tab3 = st.tabs(["📈 Analyze Categories", "🔍 Compare Selected Funds", "📊 Analyze Existing Portfolio"])

# -------------------- TAB 1: ANALYZE CATEGORIES --------------------
with tab1:
    st.header("📈 Category Analysis")
    st.markdown("Select up to 3 fund categories to analyze **most consistent performing** funds and their correlations.")
    
    selected_categories = st.multiselect(
        "Select up to 3 Fund Categories", 
        categories, 
        max_selections=3,
        key="categories_tab1"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        analyze_categories = st.button("🔍 Analyze Categories", type="primary", disabled=len(selected_categories) == 0)
    
    if analyze_categories and selected_categories:
        with st.spinner("Analyzing categories and finding most consistent funds..."):
            # Debug: Show selected categories
            st.info(f"🔍 Analyzing categories: {', '.join(selected_categories)}")
            
            # Get top performing funds
            result = get_top_performing_funds(
                selected_categories, top_n=10, rolling_period_years=rolling_period_years
            )
            
            top_fund_codes, fund_details, df_nav_pivot_top, error_msg = result
            
            if error_msg:
                st.error(f"❌ {error_msg}")
                
                # Show diagnostic information
                with st.expander("🔧 Diagnostic Information"):
                    with get_db_connection() as conn:
                        # Check available categories
                        cat_check = pd.read_sql(
                            "SELECT scheme_category, COUNT(*) as fund_count FROM mutual_fund_master_data GROUP BY scheme_category ORDER BY fund_count DESC",
                            conn
                        )
                        st.write("**Available categories and fund counts:**")
                        st.dataframe(cat_check.head(20))
                        
                        # Check if selected categories exist
                        selected_cat_check = pd.read_sql(
                            "SELECT scheme_category, COUNT(*) as fund_count FROM mutual_fund_master_data WHERE scheme_category = ANY(%s) GROUP BY scheme_category",
                            conn, params=(selected_categories,))
                        st.write("**Your selected categories:**")
                        st.dataframe(selected_cat_check)
            
            elif top_fund_codes is not None:
                # Calculate returns and correlations
                df_returns = df_nav_pivot_top.pct_change().dropna()
                corr_matrix = df_returns.corr()
                
                # Calculate rolling returns
                window = int(252 * rolling_period_years)
                rolling_returns = calculate_rolling_returns(df_nav_pivot_top, window)
                
                # Display results
                st.success(f"✅ Analyzed top {len(top_fund_codes)} most consistent funds from {len(selected_categories)} categories")
                
                # Show data availability info
                st.info(f"📊 Data period: {df_nav_pivot_top.index.min().strftime('%Y-%m-%d')} to {df_nav_pivot_top.index.max().strftime('%Y-%m-%d')}")
                
                # Fund details table with consistency metrics
                st.subheader("🏆 Top Consistent Performing Funds")
                display_df = fund_details.copy()
                
                # Format the display columns
                display_columns = ['scheme_name', 'scheme_category']
                if 'latest_return' in display_df.columns:
                    display_df['Latest Return'] = display_df['latest_return'].map('{:.2%}'.format)
                    display_columns.append('Latest Return')
                if 'avg_return' in display_df.columns:
                    display_df['Avg Return'] = display_df['avg_return'].map('{:.2%}'.format)
                    display_columns.append('Avg Return')
                if 'return_std' in display_df.columns:
                    display_df['Volatility'] = display_df['return_std'].map('{:.2%}'.format)
                    display_columns.append('Volatility')
                if 'consistency_score' in display_df.columns:
                    display_df['Consistency Score'] = display_df['consistency_score'].map('{:.3f}'.format)
                    display_columns.append('Consistency Score')
                if 'positive_ratio' in display_df.columns:
                    display_df['Positive Return %'] = display_df['positive_ratio'].map('{:.1%}'.format)
                    display_columns.append('Positive Return %')
                
                st.dataframe(display_df[display_columns], use_container_width=True)
                
                # Explain consistency score
                with st.expander("📊 Understanding Consistency Score"):
                    st.markdown("""
                    **Consistency Score** measures how reliable a fund's returns are:
                    - **Higher scores** indicate more consistent, predictable returns
                    - **Positive scores** generally indicate good risk-adjusted performance
                    - **Considers**: Return stability, positive return frequency, and risk-adjusted returns
                    - **Penalizes**: High volatility and frequent negative returns
                    """)
                
                # Correlation analysis
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🔗 Correlation Matrix")
                    st.dataframe(corr_matrix.round(3), use_container_width=True)
                    
                    # Average correlation
                    avg_corr = corr_matrix.where(~np.eye(corr_matrix.shape[0], dtype=bool)).mean().mean()
                    st.metric("Average Pairwise Correlation", f"{avg_corr:.3f}")
                
                with col2:
                    st.subheader("🎯 Correlation Heatmap")
                    fig_heatmap = plot_correlation_heatmap(corr_matrix, "Most Consistent Funds Correlation")
                    st.pyplot(fig_heatmap)
                
                # Rolling returns plot
                st.subheader(f"📊 {rolling_period_years}-Year Rolling Returns")
                fig_returns = plot_rolling_returns(rolling_returns, fund_details, rolling_period_years)
                st.pyplot(fig_returns)
                
                # Interpretation
                with st.expander("🧠 Interpretation & Insights"):
                    high_corr_pairs = []
                    low_corr_pairs = []
                    
                    for i in range(len(corr_matrix.columns)):
                        for j in range(i+1, len(corr_matrix.columns)):
                            corr_val = corr_matrix.iloc[i, j]
                            pair = (corr_matrix.columns[i], corr_matrix.columns[j], corr_val)
                            if corr_val > 0.7:
                                high_corr_pairs.append(pair)
                            elif corr_val < 0.3:
                                low_corr_pairs.append(pair)
                    
                    st.markdown("### 📊 Correlation Analysis")
                    st.markdown(f"**Average correlation:** {avg_corr:.3f}")
                    
                    if avg_corr < 0.3:
                        st.success("✅ **Excellent diversification potential** - Low average correlation suggests these consistent funds move relatively independently.")
                    elif avg_corr < 0.7:
                        st.info("ℹ️ **Good diversification** - Moderate correlation provides decent diversification benefits with consistent performers.")
                    else:
                        st.warning("⚠️ **Limited diversification** - High correlation means these funds tend to move together, but they're still consistent performers.")
                    
                    if high_corr_pairs:
                        st.markdown("**Highly correlated pairs (>0.7):**")
                        for code1, code2, corr in high_corr_pairs[:5]:
                            st.markdown(f"- {code1} & {code2}: {corr:.3f}")
                    
                    if low_corr_pairs:
                        st.markdown("**Low correlation pairs (<0.3) - Excellent for diversification:**")
                        for code1, code2, corr in low_corr_pairs[:5]:
                            st.markdown(f"- {code1} & {code2}: {corr:.3f}")
            
            else:
                st.error("❌ Could not process the selected categories. Please try different categories or check the diagnostic information above.")

# -------------------- TAB 2: COMPARE SELECTED FUNDS --------------------
with tab2:
    st.header("🔍 Fund Comparison & Portfolio Allocation")
    st.markdown("Select up to 3 specific funds from any category for detailed comparison and optimal allocation suggestions.")
    
    # Load all funds for selection
    with get_db_connection() as conn:
        query = """
            SELECT DISTINCT code, scheme_name, scheme_category
            FROM mutual_fund_master_data
            ORDER BY scheme_name
        """
        df_all_funds = pd.read_sql(query, conn)
    
    # Create fund options with category info
    fund_options = [f"{row['scheme_name']} ({row['scheme_category']})" for _, row in df_all_funds.iterrows()]
    
    selected_fund_options = st.multiselect(
        "Choose up to 3 Mutual Funds", 
        fund_options, 
        max_selections=3,
        key="funds_tab2"
    )
    
    # Risk tolerance selection for portfolio optimization
    st.subheader("🎯 Portfolio Optimization Settings")
    col1, col2 = st.columns(2)
    
    with col1:
        risk_tolerance = st.selectbox(
            "Risk Tolerance",
            options=['conservative', 'moderate', 'aggressive'],
            index=1,
            help="Conservative: Focus on consistency and low volatility\nModerate: Balance return and consistency\nAggressive: Prioritize returns over stability"
        )
    
    with col2:
        analyze_funds = st.button("🔍 Analyze Funds & Generate Allocations", type="primary", disabled=len(selected_fund_options) < 2)
    
    if analyze_funds and len(selected_fund_options) >= 2:
        with st.spinner("Analyzing selected funds for consistency and optimizing portfolio allocation..."):
            # Extract fund names from options
            selected_fund_names = [option.split(' (')[0] for option in selected_fund_options]
            
            # Get fund codes
            selected_funds_df = df_all_funds[df_all_funds['scheme_name'].isin(selected_fund_names)]
            
            # Load NAV data
            with get_db_connection() as conn:
                placeholders = ','.join(['%s'] * len(selected_fund_names))
                query = f"""
                    SELECT nav, code, value
                    FROM mutual_fund_nav
                    WHERE code IN (
                        SELECT code FROM mutual_fund_master_data WHERE scheme_name IN ({placeholders})
                    )
                    ORDER BY nav ASC
                """
                df_nav = pd.read_sql(query, conn, params=selected_fund_names)
            
            if not df_nav.empty:
                df_nav_pivot = df_nav.pivot(index='nav', columns='code', values='value').dropna()
                df_returns = df_nav_pivot.pct_change().dropna()
                
                # Calculate rolling returns
                window = int(252 * rolling_period_years)
                rolling_returns = calculate_rolling_returns(df_nav_pivot, window)
                
                if not rolling_returns.empty:
                    # Fund details
                    st.success(f"✅ Analyzing {len(selected_fund_names)} selected funds for consistency and generating optimal allocations")
                    
                    st.subheader("📋 Selected Funds")
                    display_funds = selected_funds_df.set_index('code')
                    latest_returns = rolling_returns.tail(1).T
                    display_funds['Latest_Return'] = latest_returns.iloc[:, 0]
                    display_funds['Latest_Return'] = display_funds['Latest_Return'].map('{:.2%}'.format)
                    st.dataframe(display_funds[['scheme_name', 'scheme_category', 'Latest_Return']], use_container_width=True)
                    
                    # Correlation analysis
                    corr_matrix = df_returns.corr()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("🔗 Correlation Matrix")
                        st.dataframe(corr_matrix.round(3), use_container_width=True)
                        
                        if len(corr_matrix) > 1:
                            avg_corr = corr_matrix.where(~np.eye(corr_matrix.shape[0], dtype=bool)).mean().mean()
                            st.metric("Average Pairwise Correlation", f"{avg_corr:.3f}")
                    
                    with col2:
                        st.subheader("🎯 Correlation Heatmap")
                        fig_heatmap = plot_correlation_heatmap(corr_matrix, "Selected Funds Correlation")
                        st.pyplot(fig_heatmap)
                    
                    # Rolling returns comparison
                    st.subheader(f"📈 {rolling_period_years}-Year Rolling Returns Comparison")
                    fig_returns = plot_rolling_returns(rolling_returns, display_funds, rolling_period_years, "Rolling Returns Comparison")
                    st.pyplot(fig_returns)
                    
                    # Enhanced performance metrics with consistency
                    st.subheader("📊 Performance & Consistency Metrics")
                    metrics_data = []
                    
                    for code in rolling_returns.columns:
                        fund_returns = rolling_returns[code].dropna()
                        consistency_score = calculate_consistency_score(fund_returns)
                        positive_ratio = (fund_returns > 0).sum() / len(fund_returns) if len(fund_returns) > 0 else 0
                        
                        metrics_data.append({
                            'Fund Code': code,
                            'Latest Return': fund_returns.iloc[-1] if len(fund_returns) > 0 else 0,
                            'Average Return': fund_returns.mean(),
                            'Volatility': fund_returns.std(),
                            'Consistency Score': consistency_score,
                            'Positive Return %': positive_ratio,
                            'Sharpe Ratio': (fund_returns.mean() - 0.06) / fund_returns.std() if fund_returns.std() > 0 else 0
                        })
                    
                    metrics_df = pd.DataFrame(metrics_data)
                    
                    # Format the metrics for display
                    metrics_display = metrics_df.copy()
                    metrics_display['Latest Return'] = metrics_display['Latest Return'].map('{:.2%}'.format)
                    metrics_display['Average Return'] = metrics_display['Average Return'].map('{:.2%}'.format)
                    metrics_display['Volatility'] = metrics_display['Volatility'].map('{:.2%}'.format)
                    metrics_display['Consistency Score'] = metrics_display['Consistency Score'].map('{:.3f}'.format)
                    metrics_display['Positive Return %'] = metrics_display['Positive Return %'].map('{:.1%}'.format)
                    metrics_display['Sharpe Ratio'] = metrics_display['Sharpe Ratio'].map('{:.3f}'.format)
                    
                    st.dataframe(metrics_display.set_index('Fund Code'), use_container_width=True)
                    
                    # ================== PORTFOLIO ALLOCATION OPTIMIZATION ==================
                    st.header("🎯 Portfolio Allocation Optimization")
                    st.markdown(f"**Risk Tolerance:** {risk_tolerance.title()}")
                    
                    # Get optimal allocation
                    optimal_weights, portfolio_metrics = optimize_portfolio_allocation(
                        rolling_returns, risk_tolerance=risk_tolerance
                    )
                    
                    # Get alternative strategies
                    alternative_strategies = get_alternative_allocations(rolling_returns)
                    
                    if optimal_weights and portfolio_metrics:
                        # Add optimal strategy to alternatives
                        alternative_strategies['Optimized'] = {
                            'weights': optimal_weights,
                            'return': portfolio_metrics['return'],
                            'volatility': portfolio_metrics['volatility'],
                            'consistency': portfolio_metrics['consistency_score'],
                            'sharpe': portfolio_metrics['sharpe_ratio']
                        }
                        
                        # Display optimal allocation
                        st.subheader("🏆 Recommended Portfolio Allocation")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("### 📊 Allocation Breakdown")
                            allocation_df = pd.DataFrame([
                                {'Fund Code': code, 'Fund Name': display_funds.loc[code, 'scheme_name'][:30], 
                                 'Allocation %': f"{weight*100:.1f}%"}
                                for code, weight in optimal_weights.items()
                            ])
                            st.dataframe(allocation_df, use_container_width=True, hide_index=True)
                            
                            # Portfolio metrics
                            st.markdown("### 📈 Expected Portfolio Performance")
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.metric("Expected Return", f"{portfolio_metrics['return']*100:.2f}%")
                                st.metric("Volatility", f"{portfolio_metrics['volatility']*100:.2f}%")
                            with col_b:
                                st.metric("Consistency Score", f"{portfolio_metrics['consistency_score']:.3f}")
                                st.metric("Sharpe Ratio", f"{portfolio_metrics['sharpe_ratio']:.3f}")
                        
                        with col2:
                            st.markdown("### 🥧 Allocation Visualization")
                            fig_pie = plot_allocation_pie_chart(optimal_weights, "Optimized Portfolio Allocation")
                            st.pyplot(fig_pie)
                    
                    # Strategy comparison
                    if alternative_strategies:
                        st.subheader("📊 Strategy Comparison")
                        
                        # Create comparison table
                        strategy_comparison = []
                        for strategy_name, strategy_data in alternative_strategies.items():
                            strategy_comparison.append({
                                'Strategy': strategy_name,
                                'Expected Return': f"{strategy_data['return']*100:.2f}%",
                                'Volatility': f"{strategy_data['volatility']*100:.2f}%",
                                'Consistency Score': f"{strategy_data['consistency']:.3f}",
                                'Sharpe Ratio': f"{strategy_data['sharpe']:.3f}"
                            })
                        
                        comparison_df = pd.DataFrame(strategy_comparison)
                        st.dataframe(comparison_df.set_index('Strategy'), use_container_width=True)
                        
                        # Visual comparison
                        st.subheader("📊 Strategy Performance Comparison")
                        fig_comparison = plot_strategy_comparison(alternative_strategies)
                        st.pyplot(fig_comparison)
                        
                        # Show allocation pie charts for all strategies
                        st.subheader("🥧 Allocation Comparison")
                        cols = st.columns(len(alternative_strategies))
                        
                        for i, (strategy_name, strategy_data) in enumerate(alternative_strategies.items()):
                            with cols[i % len(cols)]:
                                fig_strategy_pie = plot_allocation_pie_chart(
                                    strategy_data['weights'], 
                                    f"{strategy_name} Strategy"
                                )
                                st.pyplot(fig_strategy_pie)
                    
                    # Investment recommendations
                    st.subheader("💡 Investment Recommendations")
                    
                    with st.expander("🎯 Detailed Investment Insights", expanded=True):
                        if optimal_weights and portfolio_metrics:
                            # Find best performing strategy
                            best_strategy = max(alternative_strategies.items(), 
                                              key=lambda x: x[1]['consistency'] + x[1]['return'])
                            
                            st.markdown("### 🏆 Key Recommendations")
                            
                            if best_strategy[0] == 'Optimized':
                                st.success("✅ **The Optimized allocation is your best choice** based on your risk tolerance and consistency requirements.")
                            else:
                                st.info(f"ℹ️ **Consider the {best_strategy[0]} strategy** for potentially better risk-adjusted returns.")
                            
                            # Allocation insights
                            st.markdown("### 📊 Allocation Insights")
                            
                            # Find dominant allocation
                            max_allocation = max(optimal_weights.items(), key=lambda x: x[1])
                            min_allocation = min(optimal_weights.items(), key=lambda x: x[1])
                            
                            st.markdown(f"**Largest allocation:** {max_allocation[0]} ({max_allocation[1]*100:.1f}%)")
                            st.markdown(f"**Smallest allocation:** {min_allocation[0]} ({min_allocation[1]*100:.1f}%)")
                            
                            # Diversification analysis
                            if len(optimal_weights) > 2:
                                allocation_std = np.std(list(optimal_weights.values()))
                                if allocation_std < 0.1:
                                    st.info("ℹ️ **Well-diversified portfolio** - Allocations are relatively balanced")
                                elif allocation_std < 0.2:
                                    st.warning("⚠️ **Moderately concentrated** - Consider rebalancing if needed")
                                else:
                                    st.warning("⚠️ **Concentrated portfolio** - Higher risk due to uneven allocation")
                            
                            # Risk-return trade-off analysis
                            st.markdown("### ⚖️ Risk-Return Analysis")
                            
                            if portfolio_metrics['consistency_score'] > 0.5:
                                st.success("✅ **High consistency portfolio** - Expected stable performance")
                            elif portfolio_metrics['consistency_score'] > 0:
                                st.info("ℹ️ **Moderate consistency** - Balanced risk-return profile")
                            else:
                                st.warning("⚠️ **Lower consistency** - Higher volatility expected")
                            
                            if portfolio_metrics['sharpe_ratio'] > 1.0:
                                st.success("✅ **Excellent risk-adjusted returns** - Strong Sharpe ratio")
                            elif portfolio_metrics['sharpe_ratio'] > 0.5:
                                st.info("ℹ️ **Good risk-adjusted returns** - Decent Sharpe ratio")
                            else:
                                st.warning("⚠️ **Below-average risk adjustment** - Consider reviewing allocation")
                            
                            # Correlation insights
                            if len(corr_matrix) > 1:
                                st.markdown("### 🔗 Diversification Benefits")
                                if avg_corr < 0.3:
                                    st.success("✅ **Excellent diversification** - Low correlation provides strong risk reduction")
                                elif avg_corr < 0.7:
                                    st.info("ℹ️ **Good diversification** - Moderate correlation still offers benefits")
                                else:
                                    st.warning("⚠️ **Limited diversification** - High correlation reduces risk reduction benefits")
                            
                            # Action items
                            st.markdown("### 📋 Action Items")
                            st.markdown("""
                            1. **Review allocation percentages** and ensure they align with your investment goals
                            2. **Monitor consistency scores** - Higher scores indicate more predictable returns  
                            3. **Rebalance quarterly** to maintain optimal allocation ratios
                            4. **Track correlation changes** - Market conditions can affect fund relationships
                            5. **Consider SIP investments** to dollar-cost average into these allocations
                            """)
                    
                    # Interpretation
                    with st.expander("🧠 Interpretation & Investment Insights"):
                        st.markdown("### 📈 Performance Analysis")
                        
                        best_performer = metrics_df.loc[metrics_df['Latest Return'].idxmax()]
                        most_consistent = metrics_df.loc[metrics_df['Consistency Score'].idxmax()]
                        lowest_volatility = metrics_df.loc[metrics_df['Volatility'].idxmin()]
                        
                        st.markdown(f"**Best Performing Fund:** {best_performer['Fund Code']} ({best_performer['Latest Return']:.2%})")
                        st.markdown(f"**Most Consistent Fund:** {most_consistent['Fund Code']} (Score: {most_consistent['Consistency Score']:.3f})")
                        st.markdown(f"**Lowest Volatility Fund:** {lowest_volatility['Fund Code']} ({lowest_volatility['Volatility']:.2%})")
                        
                        if len(corr_matrix) > 1:
                            st.markdown("### 🔗 Diversification Analysis")
                            if avg_corr < 0.3:
                                st.success("✅ **Excellent diversification** - These funds have low correlation and move relatively independently.")
                            elif avg_corr < 0.7:
                                st.info("ℹ️ **Good diversification** - Moderate correlation provides decent diversification benefits.")
                            else:
                                st.warning("⚠️ **Limited diversification** - High correlation means these funds tend to move together.")
                        
                        st.markdown("### 💡 Investment Recommendations")
                        
                        # Recommend based on consistency rather than just returns
                        if most_consistent['Fund Code'] != best_performer['Fund Code']:
                            st.markdown(f"**For Stable Growth:** Consider {most_consistent['Fund Code']} - Most consistent performer")
                        
                        best_sharpe = metrics_df.loc[metrics_df['Sharpe Ratio'].idxmax()]
                        st.markdown(f"**Best Risk-Adjusted Returns:** {best_sharpe['Fund Code']} (Sharpe: {best_sharpe['Sharpe Ratio']:.3f})")
                        
                        # Consistency-based recommendation
                        consistent_funds = metrics_df[metrics_df['Consistency Score'] > 0].sort_values('Consistency Score', ascending=False)
                        if len(consistent_funds) > 0:
                            st.markdown("**Most Consistent Funds (in order):**")
                            for i, (_, fund) in enumerate(consistent_funds.head(3).iterrows(), 1):
                                st.markdown(f"{i}. {fund['Fund Code']} - Score: {fund['Consistency Score']:.3f}, Volatility: {fund['Volatility']:.2%}")
                
                else:
                    st.error("Insufficient data for rolling returns calculation. Try a shorter rolling period.")
            else:
                st.error("No NAV data available for the selected funds.")

# -------------------- TAB 3: ANALYZE EXISTING PORTFOLIO --------------------
with tab3:
    st.header("📊 Existing Portfolio Analysis")
    st.markdown("Analyze your current portfolio allocation and get optimization suggestions based on your risk profile.")
    
    # Load portfolio data from the other script
    with st.spinner("Loading your portfolio data..."):
        try:
            # Get all the necessary data
            df = get_portfolio_data()
            latest_nav = get_latest_nav()
            goal_mappings = get_goal_mappings()
            
            if df.empty or latest_nav.empty:
                st.warning("No portfolio data found. Please ensure you have investments recorded.")
                st.stop()
                
            # Prepare the data
            df['date'] = pd.to_datetime(df['date'])
            df = prepare_cashflows(df)
            df = calculate_units(df)
            
            # Calculate current portfolio weights
            weights_df = calculate_portfolio_weights(df, latest_nav, goal_mappings)
            
            # Calculate equity and debt allocation
            equity_value = weights_df[weights_df['scheme_name'].isin(df['scheme_name'].unique())]['current_value'].sum()
            debt_value = goal_mappings[goal_mappings['investment_type'] == 'Debt']['current_value'].sum()
            total_portfolio_value = equity_value + debt_value
            
            equity_percent = (equity_value / total_portfolio_value) * 100 if total_portfolio_value > 0 else 0
            debt_percent = (debt_value / total_portfolio_value) * 100 if total_portfolio_value > 0 else 0
            
            # Display current allocation
            st.subheader("Current Asset Allocation")
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.pie(
                    values=[equity_value, debt_value],
                    names=['Equity', 'Debt'],
                    title="Overall Equity vs Debt Allocation",
                    height=300
                )
                fig.update_traces(textinfo='percent+label', pull=[0.1, 0])
                st.plotly_chart(fig, use_container_width=True)
                
            with col2:
                st.metric("Total Portfolio Value", format_indian_currency(total_portfolio_value))
                st.metric("Equity Allocation", f"{equity_percent:.1f}% ({format_indian_currency(equity_value)})")
                st.metric("Debt Allocation", f"{debt_percent:.1f}% ({format_indian_currency(debt_value)})")
            
            # Risk profile selection
            st.subheader("Portfolio Optimization Settings")
            risk_tolerance = st.selectbox(
                "Your Risk Tolerance",
                options=['conservative', 'moderate', 'aggressive'],
                index=1,
                help="Conservative: More debt instruments\nModerate: Balanced approach\nAggressive: Higher equity allocation"
            )
            
            # Suggested allocation based on risk profile
            st.subheader("Suggested Asset Allocation")
            
            if risk_tolerance == 'conservative':
                suggested_equity = 40
                suggested_debt = 60
            elif risk_tolerance == 'moderate':
                suggested_equity = 60
                suggested_debt = 40
            else:  # aggressive
                suggested_equity = 80
                suggested_debt = 20
                
            col3, col4 = st.columns(2)
            
            with col3:
                fig = px.pie(
                    values=[suggested_equity, suggested_debt],
                    names=['Equity', 'Debt'],
                    title=f"Suggested {risk_tolerance.title()} Allocation",
                    height=300
                )
                fig.update_traces(textinfo='percent+label', pull=[0.1, 0])
                st.plotly_chart(fig, use_container_width=True)
                
            with col4:
                current_diff = equity_percent - suggested_equity
                action = "Reduce" if current_diff > 0 else "Increase"
                st.metric("Suggested Equity Allocation", f"{suggested_equity}%", 
                         delta=f"{action} by {abs(current_diff):.1f}%")
                st.metric("Suggested Debt Allocation", f"{suggested_debt}%", 
                         delta=f"{'Increase' if current_diff > 0 else 'Reduce'} by {abs(current_diff):.1f}%")
            
            # Analyze current mutual fund holdings
            st.subheader("Current Mutual Fund Holdings Analysis")
            
            # Get NAV data for current holdings
            current_fund_codes = df['code'].unique()
            with get_db_connection() as conn:
                nav_query = """
                    SELECT nav, code, value
                    FROM mutual_fund_nav
                    WHERE code = ANY(%s)
                    ORDER BY nav ASC
                """
                df_nav = pd.read_sql(nav_query, conn, params=(list(current_fund_codes),))
            
            if not df_nav.empty:
                df_nav_pivot = df_nav.pivot(index='nav', columns='code', values='value')
                
                # Calculate rolling returns
                window = int(252 * rolling_period_years)
                rolling_returns = calculate_rolling_returns(df_nav_pivot, window)
                
                if not rolling_returns.empty:
                    # Get fund details
                    with get_db_connection() as conn:
                        fund_query = """
                            SELECT code, scheme_name, scheme_category
                            FROM mutual_fund_master_data
                            WHERE code = ANY(%s)
                        """
                        fund_details = pd.read_sql(fund_query, conn, params=(list(current_fund_codes),))
                    
                    # Check if we got any results
                    if fund_details.empty:
                        st.warning("No fund details found in the database for your portfolio holdings")
                        st.stop()
                    
                    # Calculate metrics for each fund
                    metrics_data = []
                    for code in rolling_returns.columns:
                        # Check if we have details for this fund
                        fund_info = fund_details[fund_details['code'] == code]
                        if fund_info.empty:
                            st.warning(f"No details found for fund code: {code}")
                            continue
                            
                        fund_returns = rolling_returns[code].dropna()
                        consistency_score = calculate_consistency_score(fund_returns)
                        positive_ratio = (fund_returns > 0).sum() / len(fund_returns) if len(fund_returns) > 0 else 0
                        
                        # Get current weight - handle case where fund might not be in weights_df
                        current_weight = 0
                        scheme_name = fund_info['scheme_name'].values[0]
                        weight_match = weights_df[weights_df['scheme_name'] == scheme_name]
                        if not weight_match.empty:
                            current_weight = weight_match['weight'].values[0]
                        
                        metrics_data.append({
                            'Fund Code': code,
                            'Fund Name': fund_info['scheme_name'].values[0],
                            'Category': fund_info['scheme_category'].values[0],
                            'Current Weight (%)': current_weight,
                            'Latest Return': fund_returns.iloc[-1] if len(fund_returns) > 0 else 0,
                            'Average Return': fund_returns.mean(),
                            'Volatility': fund_returns.std(),
                            'Consistency Score': consistency_score,
                            'Positive Return %': positive_ratio,
                            'Sharpe Ratio': (fund_returns.mean() - 0.06) / fund_returns.std() if fund_returns.std() > 0 else 0
                        })
                    
                    if not metrics_data:
                        st.error("No valid fund metrics could be calculated. Please check your portfolio data.")
                        st.stop()

                    metrics_df = pd.DataFrame(metrics_data)

                    # Only proceed if we have data to display
                    if metrics_df.empty:
                        st.warning("No performance metrics available for your current holdings")
                        st.stop()
                    
                    # Display metrics
                    st.dataframe(
                        metrics_df.sort_values('Consistency Score', ascending=False),
                        use_container_width=True
                    )
                    
                    # Optimization suggestions
                    st.subheader("Fund Optimization Suggestions")
                    
                    # Identify funds to consider reducing
                    low_performers = metrics_df[
                        (metrics_df['Consistency Score'] < 0) | 
                        (metrics_df['Sharpe Ratio'] < 0.5)
                    ].sort_values('Consistency Score')
                    
                    if not low_performers.empty:
                        st.warning("⚠️ Consider reducing exposure to these lower-performing funds:")
                        st.dataframe(low_performers, use_container_width=True)
                    
                    # Find similar categories in current portfolio
                    current_categories = metrics_df['Category'].unique()
                    similar_categories = []
                    for cat in current_categories:
                        if any(c.lower() in cat.lower() for c in ['equity', 'growth', 'flexi', 'multi']):
                            similar_categories.append('Equity')
                        elif any(c.lower() in cat.lower() for c in ['debt', 'income', 'gilt']):
                            similar_categories.append('Debt')
                        else:
                            similar_categories.append('Other')
                    
                    # Correlation analysis
                    corr_matrix = rolling_returns.corr()
                    avg_corr = corr_matrix.where(~np.eye(corr_matrix.shape[0], dtype=bool)).mean().mean()
                    
                    st.metric("Average Correlation Among Holdings", f"{avg_corr:.3f}")
                    
                    if avg_corr > 0.7:
                        st.warning("⚠️ High correlation among your funds - consider adding more diversified options")
                    elif avg_corr < 0.3:
                        st.success("✅ Good diversification - your funds have low correlation")
                    else:
                        st.info("ℹ️ Moderate correlation - could benefit from some additional diversification")
                    
                    # Suggested actions
                    st.subheader("Recommended Actions")
                    
                    if current_diff > 5:  # More equity than suggested
                        st.info(f"🔧 **Rebalance Portfolio**: Move {abs(current_diff):.1f}% from equity to debt to align with {risk_tolerance} risk profile")
                    elif current_diff < -5:  # Less equity than suggested
                        st.info(f"🔧 **Rebalance Portfolio**: Move {abs(current_diff):.1f}% from debt to equity to align with {risk_tolerance} risk profile")
                    else:
                        st.success("✅ Your current allocation aligns well with your selected risk profile")
                    
                    if not low_performers.empty:
                        st.info("🔧 **Consider replacing** lower-consistency funds with more consistent performers from similar categories")
                    
                    if avg_corr > 0.7:
                        st.info("🔧 **Add uncorrelated assets** to improve portfolio diversification")
                    
                else:
                    st.warning("Could not calculate rolling returns for current holdings")
            else:
                st.warning("No NAV data found for current holdings")
                
        except Exception as e:
            st.error(f"Error loading portfolio data: {e}")
            st.error("Please ensure the portfolio analysis database tables exist and contain data")

# -------------------- FOOTER --------------------
st.markdown("---")
st.markdown("### 📚 How to Use This Tool")
with st.expander("Click for detailed instructions"):
    st.markdown("""
    **Category Analysis Tab:**
    1. Select up to 3 fund categories
    2. Click "Analyze Categories" to see top 10 **most consistent** performing funds
    3. Review consistency scores, volatility, and correlation metrics
    4. Examine rolling returns chart for performance trends
    
    **Fund Comparison Tab:**
    1. Select up to 3 specific funds from any category
    2. Choose your risk tolerance (Conservative/Moderate/Aggressive)
    3. Click "Analyze Funds & Generate Allocations" for detailed analysis
    4. Review **optimized portfolio allocation** recommendations
    5. Compare different allocation strategies
    6. Use insights for investment decisions prioritizing stability
    
    **Existing Portfolio Analysis Tab:**
    1. Automatically loads your current portfolio from the database
    2. Shows your current equity vs debt allocation
    3. Provides suggested allocation based on your risk tolerance
    4. Analyzes each fund's consistency and performance
    5. Recommends specific actions to optimize your portfolio
    
    **Portfolio Optimization Features:**
    - **Optimized Allocation**: Uses mathematical optimization to balance returns, consistency, and diversification
    - **Alternative Strategies**: Compare Equal Weight, Consistency-Weighted, and Low Correlation approaches
    - **Risk Tolerance**: Adjust optimization based on your risk preference
    - **Visual Comparisons**: Pie charts and performance metrics for easy comparison
    
    **Consistency Focus:**
    - **Consistency Score**: Higher values indicate more reliable, predictable returns
    - **Volatility**: Lower values show more stable performance
    - **Positive Return %**: Shows frequency of positive returns
    - **Allocation %**: Recommended investment percentages for optimal portfolio
    
    **Configuration:**
    - Adjust rolling period in the sidebar (1-10 years)
    - Longer periods provide more stable consistency metrics
    - Shorter periods show recent consistency trends
    """)

st.markdown("### 🎯 Key Features")
st.info("""
**This tool prioritizes CONSISTENT returns and OPTIMAL ALLOCATION:**
- ✅ Selects funds with stable, predictable performance
- ✅ **Optimizes portfolio allocation** to minimize correlation and maximize consistency
- ✅ Provides **multiple allocation strategies** for comparison
- ✅ Considers your **risk tolerance** in optimization
- ✅ **Visual allocation recommendations** with pie charts and performance metrics
- ✅ Focuses on long-term reliability over short-term gains
- ✅ **Analyzes your existing portfolio** and suggests improvements
""")