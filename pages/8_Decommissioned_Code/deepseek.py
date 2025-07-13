import streamlit as st
import pandas as pd
import numpy as np
import psycopg
from psycopg_pool import ConnectionPool
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import time
import warnings
warnings.filterwarnings('ignore')

# Configuration
DB_CONFIG = {
    'conninfo': "dbname=postgres user=postgres password=admin123 host=localhost port=5432",
    'min_size': 1,
    'max_size': 5,
    'timeout': 30
}

# Initialize connection pool
@st.cache_resource
def init_connection_pool():
    try:
        pool = ConnectionPool(
            conninfo=DB_CONFIG['conninfo'],
            min_size=DB_CONFIG['min_size'],
            max_size=DB_CONFIG['max_size'],
            timeout=DB_CONFIG['timeout']
        )
        # Test the pool
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        return pool
    except Exception as e:
        st.error(f"Failed to initialize connection pool: {str(e)}")
        return None

# Get connection from pool with retry
def get_connection(pool, max_retries=3, retry_delay=2):
    for attempt in range(max_retries):
        try:
            conn = pool.getconn()
            # Test the connection
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
            return conn
        except Exception as e:
            if attempt == max_retries - 1:
                st.error(f"Failed to get connection after {max_retries} attempts")
                return None
            time.sleep(retry_delay)
    return None

# Robust data fetching with connection handling
@st.cache_data(ttl=3600, show_spinner="Loading NAV data...")
def get_historical_nav(pool):
    conn = None
    try:
        conn = get_connection(pool)
        if conn is None:
            return pd.DataFrame()
        
        # Get distinct codes first
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT code FROM mutual_fund_nav ORDER BY code")
            codes = [row[0] for row in cur.fetchall()]
        
        # Process in chunks
        chunk_size = 100
        dfs = []
        
        for i in range(0, len(codes), chunk_size):
            chunk = codes[i:i + chunk_size]
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT code, nav, value as nav_value
                    FROM mutual_fund_nav
                    WHERE code = ANY(%s)
                    ORDER BY code, nav
                """, (chunk,))
                
                chunk_df = pd.DataFrame(cur.fetchall(), columns=['code', 'nav', 'nav_value'])
                # Optimize data types
                chunk_df['code'] = chunk_df['code'].astype('category')
                chunk_df['nav_value'] = pd.to_numeric(chunk_df['nav_value'], downcast='float')
                dfs.append(chunk_df)
        
        if dfs:
            final_df = pd.concat(dfs, ignore_index=True)
            final_df['nav'] = pd.to_datetime(final_df['nav'])
            return final_df
        return pd.DataFrame()
        
    except Exception as e:
        st.error(f"Error fetching historical NAV data: {str(e)}")
        return pd.DataFrame()
    finally:
        if conn:
            pool.putconn(conn)

# Robust goal mappings fetch
@st.cache_data(ttl=600)
def get_goal_mappings(pool):
    conn = None
    try:
        conn = get_connection(pool)
        if conn is None:
            return pd.DataFrame()
        
        with conn.cursor() as cur:
            cur.execute("""
                SELECT goal_name, investment_type, scheme_name, scheme_code, current_value
                FROM goals
                WHERE investment_type = 'Equity'
                ORDER BY goal_name, scheme_name
            """)
            
            columns = [desc[0] for desc in cur.description]
            data = cur.fetchall()
            
            if data:
                df = pd.DataFrame(data, columns=columns)
                # Optimize data types
                df['scheme_code'] = df['scheme_code'].astype('category')
                df['goal_name'] = df['goal_name'].astype('category')
                df['scheme_name'] = df['scheme_name'].astype('category')
                df['current_value'] = pd.to_numeric(df['current_value'], downcast='float')
                return df
            return pd.DataFrame()
            
    except Exception as e:
        st.error(f"Error fetching goal mappings: {str(e)}")
        return pd.DataFrame()
    finally:
        if conn:
            pool.putconn(conn)

# Optimized compute_returns function
@st.cache_data(ttl=300)
def compute_returns(nav_df, period='daily'):
    try:
        if nav_df.empty:
            return pd.DataFrame(), pd.DataFrame()
        
        # Create pivot table more efficiently
        pivot = nav_df.pivot_table(
            index='nav', 
            columns='code', 
            values='nav_value',
            aggfunc='first'
        )
        
        # Handle missing values
        pivot.ffill(inplace=True)
        pivot.bfill(inplace=True)
        
        # Calculate returns based on period
        if period == 'daily':
            returns = pivot.pct_change().dropna()
        elif period == 'weekly':
            returns = pivot.resample('W').last().pct_change().dropna()
        elif period == 'monthly':
            returns = pivot.resample('M').last().pct_change().dropna()
        
        return returns, pivot
    except Exception as e:
        st.error(f"Error computing returns: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()

# Optimized advanced metrics calculation
def compute_advanced_metrics(returns, goal_funds):
    try:
        if returns.empty or goal_funds.empty:
            return pd.DataFrame()
        
        codes = goal_funds['scheme_code'].unique()
        filtered_returns = returns[returns.columns.intersection(codes)]
        
        if filtered_returns.empty:
            return pd.DataFrame()
        
        # Pre-calculate common values
        annual_factor = np.sqrt(252)
        metrics = {}
        
        for code in filtered_returns.columns:
            fund_returns = filtered_returns[code].dropna()
            
            if len(fund_returns) < 10:
                continue
            
            # Vectorized calculations
            mean_return = fund_returns.mean() * 252
            volatility = fund_returns.std() * annual_factor
            sharpe_ratio = mean_return / volatility if volatility > 0 else 0
            
            metrics[code] = {
                'mean_return': mean_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': calculate_max_drawdown(fund_returns),
                'var_95': fund_returns.quantile(0.05),
                'skewness': stats.skew(fund_returns),
                'kurtosis': stats.kurtosis(fund_returns)
            }
        
        return pd.DataFrame(metrics).T
    except Exception as e:
        st.error(f"Error computing advanced metrics: {str(e)}")
        return pd.DataFrame()

# Optimized max drawdown calculation
def calculate_max_drawdown(returns):
    try:
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        return drawdown.min()
    except:
        return np.nan

# Optimized portfolio metrics
def compute_portfolio_metrics(returns, weights):
    try:
        portfolio_return = (returns * weights).sum(axis=1)
        annual_factor = np.sqrt(252)
        
        return {
            'expected_return': portfolio_return.mean() * 252,
            'volatility': portfolio_return.std() * annual_factor,
            'sharpe_ratio': (portfolio_return.mean() * 252) / (portfolio_return.std() * annual_factor),
            'max_drawdown': calculate_max_drawdown(portfolio_return)
        }
    except:
        return {
            'expected_return': np.nan,
            'volatility': np.nan,
            'sharpe_ratio': np.nan,
            'max_drawdown': np.nan
        }

# Optimized portfolio optimization
def optimize_portfolio(returns, goal_funds, method='inverse_correlation'):
    try:
        codes = goal_funds['scheme_code'].unique()
        filtered_returns = returns[returns.columns.intersection(codes)]
        
        if len(filtered_returns.columns) < 2:
            return pd.Series(1.0, index=codes)
        
        if method == 'inverse_correlation':
            corr_matrix = filtered_returns.corr()
            avg_corr = corr_matrix.mean(axis=1)
            inv_corr = 1 / (avg_corr + 1e-6)
            weights = inv_corr / inv_corr.sum()
        
        elif method == 'equal_weight':
            weights = pd.Series(1/len(codes), index=codes)
        
        elif method == 'risk_parity':
            volatilities = filtered_returns.std()
            inv_vol = 1 / (volatilities + 1e-6)
            weights = inv_vol / inv_vol.sum()
        
        elif method == 'min_variance':
            cov_matrix = filtered_returns.cov()
            inv_cov = np.linalg.pinv(cov_matrix)
            ones = np.ones(len(codes))
            weights_array = inv_cov @ ones / (ones.T @ inv_cov @ ones)
            weights = pd.Series(weights_array, index=codes)
        
        return weights.fillna(0)
    except Exception as e:
        st.error(f"Optimization error: {str(e)}")
        return pd.Series(1.0, index=goal_funds['scheme_code'].unique())

# Optimized portfolio performance calculation
def calculate_portfolio_performance(nav_pivot, weights, scheme_codes):
    try:
        portfolio_nav = (nav_pivot[scheme_codes] * weights).sum(axis=1)
        portfolio_returns = portfolio_nav.pct_change().dropna()
        return portfolio_nav, portfolio_returns
    except:
        return pd.Series(), pd.Series()

# Formatting function
def format_indian_number(number):
    try:
        number = float(number)
        if number >= 10000000:
            return f"₹{number/10000000:.2f}Cr"
        elif number >= 100000:
            return f"₹{number/100000:.2f}L"
        else:
            return f"₹{number:,.0f}"
    except:
        return "₹0"

# Visualization functions
def create_correlation_heatmap(corr_matrix, goal_name):
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False
    ))
    fig.update_layout(
        title=f"Correlation Matrix - {goal_name}",
        xaxis_title="Funds",
        yaxis_title="Funds",
        height=400
    )
    return fig

def create_performance_chart(nav_pivot, weights, scheme_codes, goal_name):
    fig = go.Figure()
    
    # Add individual fund performance
    for code in scheme_codes:
        if code in nav_pivot.columns:
            normalized = nav_pivot[code] / nav_pivot[code].iloc[0] * 100
            fig.add_trace(go.Scatter(
                x=nav_pivot.index,
                y=normalized,
                mode='lines',
                name=f'Fund {code}',
                opacity=0.6,
                line=dict(width=1)
            ))
    
    # Add portfolio performance if we have data
    portfolio_nav, _ = calculate_portfolio_performance(nav_pivot, weights, scheme_codes)
    if not portfolio_nav.empty:
        portfolio_normalized = portfolio_nav / portfolio_nav.iloc[0] * 100
        fig.add_trace(go.Scatter(
            x=nav_pivot.index,
            y=portfolio_normalized,
            mode='lines',
            name='Optimized Portfolio',
            line=dict(width=3, color='red')
        ))
    
    fig.update_layout(
        title=f"Performance Comparison - {goal_name}",
        xaxis_title="Date",
        yaxis_title="Normalized Value (Base=100)",
        hovermode='x unified',
        height=400
    )
    return fig

def create_allocation_pie_chart(reallocation_df, goal_name):
    fig = px.pie(
        reallocation_df,
        values='recommended_allocation',
        names='scheme_name',
        title=f"Recommended Allocation - {goal_name}",
        height=400
    )
    return fig

def get_correlation_insights(corr_matrix):
    insights = {}
    mask = ~np.eye(len(corr_matrix), dtype=bool)
    masked_corr = corr_matrix.where(mask)
    valid_correlations = masked_corr.stack().dropna()
    
    if len(valid_correlations) > 0:
        insights['average_correlation'] = valid_correlations.mean()
        max_pair = valid_correlations.idxmax()
        min_pair = valid_correlations.idxmin()
        insights['highest_pair'] = (max_pair, valid_correlations[max_pair])
        insights['lowest_pair'] = (min_pair, valid_correlations[min_pair])
    else:
        insights['average_correlation'] = np.nan
        insights['highest_pair'] = (None, np.nan)
        insights['lowest_pair'] = (None, np.nan)
    
    return insights

def main():
    st.title("🎯 Mutual Fund Portfolio Analyzer")
    st.markdown("*Advanced analysis with reliable database connections*")
    
    # Initialize connection pool
    pool = init_connection_pool()
    if pool is None:
        st.error("❌ Failed to initialize database connection pool")
        st.stop()
    
    # Initialize session state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    
    # Sidebar controls
    st.sidebar.header("📊 Analysis Controls")
    analysis_type = st.sidebar.selectbox(
        "Select Analysis Type",
        ["Goal-wise Analysis", "Portfolio Overview", "Risk Analysis", "Performance Comparison"]
    )
    
    optimization_method = st.sidebar.selectbox(
        "Portfolio Optimization Method",
        ["inverse_correlation", "equal_weight", "risk_parity", "min_variance"]
    )
    
    return_period = st.sidebar.selectbox(
        "Return Calculation Period",
        ["daily", "weekly", "monthly"]
    )
    
    # Test database connection
    if st.sidebar.button("Test Database Connection"):
        with st.spinner("Testing connection..."):
            try:
                with pool.connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT 1")
                st.sidebar.success("✅ Database connection successful!")
            except Exception as e:
                st.sidebar.error(f"❌ Connection failed: {str(e)}")
    
    # Load data
    if not st.session_state.data_loaded:
        with st.spinner("Loading data (please wait)..."):
            try:
                # Load data using connection pool
                goal_mappings = get_goal_mappings(pool)
                nav_df = get_historical_nav(pool)
                
                if goal_mappings.empty or nav_df.empty:
                    st.error("❌ Loaded empty datasets. Please check your database content.")
                    st.stop()
                
                returns, nav_pivot = compute_returns(nav_df, return_period)
                
                # Store in session state
                st.session_state.returns = returns
                st.session_state.nav_pivot = nav_pivot
                st.session_state.goal_mappings = goal_mappings
                st.session_state.data_loaded = True
                
            except Exception as e:
                st.error(f"❌ Error loading data: {str(e)}")
                st.stop()
    else:
        # Use cached data
        goal_mappings = st.session_state.goal_mappings
        returns = st.session_state.returns
        nav_pivot = st.session_state.nav_pivot
    
    if analysis_type == "Portfolio Overview":
        st.header("📈 Portfolio Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Goals", len(goal_mappings['goal_name'].unique()))
        
        with col2:
            st.metric("Total Funds", len(goal_mappings['scheme_code'].unique()))
        
        with col3:
            total_value = goal_mappings['current_value'].sum()
            st.metric("Total Portfolio Value", format_indian_number(total_value))
        
        with col4:
            avg_allocation = goal_mappings['current_value'].mean()
            st.metric("Average Fund Allocation", format_indian_number(avg_allocation))
        
        goal_summary = goal_mappings.groupby('goal_name')['current_value'].sum().reset_index()
        fig_goals = px.pie(
            goal_summary,
            values='current_value',
            names='goal_name',
            title="Goal-wise Portfolio Distribution"
        )
        st.plotly_chart(fig_goals, use_container_width=True)
    
    elif analysis_type == "Risk Analysis":
        st.header("⚠️ Risk Analysis")
        all_metrics = compute_advanced_metrics(returns, goal_mappings)
        
        st.subheader("Fund Risk Metrics")
        if not all_metrics.empty and 'volatility' in all_metrics.columns and 'mean_return' in all_metrics.columns:
            st.dataframe(all_metrics.round(4))
            
            try:
                fig_risk = px.scatter(
                    all_metrics,
                    x='volatility',
                    y='mean_return',
                    size='sharpe_ratio',
                    hover_data=['max_drawdown', 'var_95'],
                    title="Risk-Return Profile of Funds"
                )
                st.plotly_chart(fig_risk, use_container_width=True)
            except Exception as e:
                st.error(f"Could not create risk-return plot: {str(e)}")
        else:
            st.warning("No valid risk metrics available. Check if you have enough historical data.")
    
    elif analysis_type == "Performance Comparison":
        st.header("📊 Performance Comparison")
        
        for goal in goal_mappings['goal_name'].unique():
            goal_df = goal_mappings[goal_mappings['goal_name'] == goal]
            if len(goal_df) < 1:
                continue
            
            st.subheader(f"🎯 {goal}")
            weights = optimize_portfolio(returns, goal_df, optimization_method)
            scheme_codes = goal_df['scheme_code'].unique()
            
            perf_chart = create_performance_chart(nav_pivot, weights, scheme_codes, goal)
            st.plotly_chart(perf_chart, use_container_width=True)
    
    else:  # Goal-wise Analysis
        st.header("🎯 Goal-wise Analysis")
        
        for goal in goal_mappings['goal_name'].unique():
            st.subheader(f"Goal: {goal}")
            goal_df = goal_mappings[goal_mappings['goal_name'] == goal]
            
            if len(goal_df) < 1:
                st.info("No funds under this goal.")
                continue
            
            tab1, tab2, tab3, tab4 = st.tabs(["Correlation", "Allocation", "Performance", "Metrics"])
            
            with tab1:
                codes = goal_df['scheme_code'].unique()
                filtered_returns = returns[returns.columns.intersection(codes)]
                
                if len(filtered_returns.columns) < 2:
                    st.warning("Need at least 2 funds with overlapping data to compute correlations.")
                else:
                    try:
                        corr_matrix = filtered_returns.corr()
                        corr_fig = create_correlation_heatmap(corr_matrix, goal)
                        st.plotly_chart(corr_fig, use_container_width=True)
                        
                        insights = get_correlation_insights(corr_matrix)
                        st.write("**Correlation Insights:**")
                        if not np.isnan(insights['average_correlation']):
                            st.write(f"• Average correlation: {insights['average_correlation']:.3f}")
                            st.write(f"• Highest correlated pair: {insights['highest_pair'][0]} ({insights['highest_pair'][1]:.3f})")
                            st.write(f"• Lowest correlated pair: {insights['lowest_pair'][0]} ({insights['lowest_pair'][1]:.3f})")
                        else:
                            st.warning("Could not compute correlations. Possible reasons:")
                            st.warning("- Funds don't have overlapping date ranges")
                            st.warning("- Insufficient historical NAV data")
                    except Exception as e:
                        st.error(f"Error computing correlations: {str(e)}")
            
            with tab2:
                try:
                    weights = optimize_portfolio(returns, goal_df, optimization_method)
                    reallocation_df = goal_df[['scheme_name', 'scheme_code', 'current_value']].copy()
                    total_value = reallocation_df['current_value'].sum()
                    
                    reallocation_df['weight'] = reallocation_df['scheme_code'].map(weights)
                    reallocation_df['recommended_allocation'] = reallocation_df['weight'] * total_value
                    reallocation_df['allocation_change'] = reallocation_df['recommended_allocation'] - reallocation_df['current_value']
                    reallocation_df['allocation_change_pct'] = (reallocation_df['allocation_change'] / reallocation_df['current_value']) * 100
                    
                    display_df = reallocation_df.copy()
                    display_df['Current Value'] = display_df['current_value'].apply(format_indian_number)
                    display_df['Recommended Allocation'] = display_df['recommended_allocation'].apply(format_indian_number)
                    display_df['Change'] = display_df['allocation_change'].apply(format_indian_number)
                    display_df['Change %'] = display_df['allocation_change_pct'].apply(lambda x: f"{x:+.1f}%")
                    
                    st.dataframe(
                        display_df[['scheme_name', 'Current Value', 'Recommended Allocation', 'Change', 'Change %']],
                        use_container_width=True
                    )
                    
                    if len(goal_df) > 1:
                        alloc_fig = create_allocation_pie_chart(reallocation_df, goal)
                        st.plotly_chart(alloc_fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error optimizing portfolio allocation: {str(e)}")
            
            with tab3:
                try:
                    scheme_codes = goal_df['scheme_code'].unique()
                    filtered_returns = returns[returns.columns.intersection(scheme_codes)]
                    
                    current_weights = goal_df.set_index('scheme_code')['current_value'] / goal_df['current_value'].sum()
                    current_metrics = compute_portfolio_metrics(filtered_returns, current_weights)
                    
                    optimized_metrics = compute_portfolio_metrics(filtered_returns, weights)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Current Portfolio")
                        st.metric("Expected Return", f"{current_metrics['expected_return']:.2%}")
                        st.metric("Volatility", f"{current_metrics['volatility']:.2%}")
                        st.metric("Sharpe Ratio", f"{current_metrics['sharpe_ratio']:.3f}")
                        st.metric("Max Drawdown", f"{current_metrics['max_drawdown']:.2%}")
                    
                    with col2:
                        st.subheader("Optimized Portfolio")
                        st.metric("Expected Return", f"{optimized_metrics['expected_return']:.2%}")
                        st.metric("Volatility", f"{optimized_metrics['volatility']:.2%}")
                        st.metric("Sharpe Ratio", f"{optimized_metrics['sharpe_ratio']:.3f}")
                        st.metric("Max Drawdown", f"{optimized_metrics['max_drawdown']:.2%}")
                    
                    perf_chart = create_performance_chart(nav_pivot, weights, scheme_codes, goal)
                    st.plotly_chart(perf_chart, use_container_width=True)
                except Exception as e:
                    st.error(f"Error analyzing performance: {str(e)}")
            
            with tab4:
                try:
                    fund_metrics = compute_advanced_metrics(returns, goal_df)
                    
                    if not fund_metrics.empty:
                        fund_metrics_display = fund_metrics.copy()
                        fund_metrics_display['Fund Name'] = goal_df.set_index('scheme_code')['scheme_name']
                        fund_metrics_display = fund_metrics_display.reset_index()
                        
                        metrics_formatted = fund_metrics_display.copy()
                        metrics_formatted['Expected Return'] = metrics_formatted['mean_return'].apply(lambda x: f"{x:.2%}")
                        metrics_formatted['Volatility'] = metrics_formatted['volatility'].apply(lambda x: f"{x:.2%}")
                        metrics_formatted['Sharpe Ratio'] = metrics_formatted['sharpe_ratio'].apply(lambda x: f"{x:.3f}")
                        metrics_formatted['Max Drawdown'] = metrics_formatted['max_drawdown'].apply(lambda x: f"{x:.2%}")
                        metrics_formatted['VaR (95%)'] = metrics_formatted['var_95'].apply(lambda x: f"{x:.2%}")
                        
                        st.dataframe(
                            metrics_formatted[['Fund Name', 'Expected Return', 'Volatility', 'Sharpe Ratio', 'Max Drawdown', 'VaR (95%)']],
                            use_container_width=True
                        )
                    else:
                        st.warning("No metrics available for these funds. Check if you have enough historical data.")
                except Exception as e:
                    st.error(f"Error computing fund metrics: {str(e)}")
            
            st.markdown("---")
    
    st.markdown("---")
    st.markdown("*Portfolio analysis based on historical data. Past performance doesn't guarantee future results.*")

if __name__ == '__main__':
    main()