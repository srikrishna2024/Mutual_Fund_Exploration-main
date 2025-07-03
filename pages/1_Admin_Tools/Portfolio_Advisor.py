import streamlit as st
import pandas as pd
import numpy as np
import psycopg
import plotly.express as px
import re
from datetime import datetime, timedelta

# Database connection
def connect_to_db():
    DB_PARAMS = {
        'dbname': 'postgres',
        'user': 'postgres',
        'password': 'admin123',
        'host': 'localhost',
        'port': '5432'
    }
    return psycopg.connect(**DB_PARAMS)

# Fetch fund data with proper validation and time period filter
def get_fund_data(period='3 years'):
    with connect_to_db() as conn:
        try:
            # Calculate date range based on period selection
            if period == 'YTD':
                start_date = datetime(datetime.now().year, 1, 1).strftime('%Y-%m-%d')
            elif period == 'All available':
                start_date = '1970-01-01'  # Very early date to get all data
            else:
                years = int(period.split()[0])
                start_date = (datetime.now() - timedelta(days=365*years)).strftime('%Y-%m-%d')
            
            # Get fund metadata
            funds = pd.read_sql(f"""
                SELECT DISTINCT n.code, n.scheme_name, m.scheme_category
                FROM mutual_fund_nav n
                JOIN mutual_fund_master_data m ON n.code = m.code
                WHERE n.nav >= '{start_date}'
            """, conn)
            
            # Clean category names using regex
            funds['category'] = funds['scheme_category'].str.extract(r'^(Equity|Hybrid|Debt)')[0]
            
            # Get NAV data for funds with recent data
            nav_data = pd.read_sql(f"""
                SELECT code, nav, value 
                FROM mutual_fund_nav 
                WHERE nav >= '{start_date}'
            """, conn)
            
            return funds, nav_data
            
        except Exception as e:
            st.error(f"Database error: {str(e)}")
            return pd.DataFrame(), pd.DataFrame()

# Calculate correlations and volatility with validation
def calculate_metrics(nav_data, min_data_points=50):
    try:
        nav_pivot = nav_data.pivot(index='nav', columns='code', values='value')
        
        # Filter funds with insufficient data
        valid_funds = nav_pivot.columns[nav_pivot.count() >= min_data_points]
        nav_pivot = nav_pivot[valid_funds]
        
        returns = nav_pivot.pct_change().dropna()
        
        if returns.empty:
            st.warning("Insufficient data to calculate metrics")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            
        # Calculate correlation matrix
        corr_matrix = returns.corr()
        
        # Calculate annualized volatility for each fund
        volatility = returns.std() * np.sqrt(252)
        volatility.name = 'Volatility'
        
        return corr_matrix, returns, volatility
        
    except Exception as e:
        st.error(f"Metrics calculation failed: {str(e)}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# Enhanced portfolio suggestion engine with volatility filtering
def suggest_portfolio(central_fund_code, time_horizon, corr_matrix, funds, volatility, top_n=3):
    try:
        # Validate inputs
        if corr_matrix.empty:
            raise ValueError("No correlation data available")
            
        if central_fund_code not in corr_matrix.columns:
            available_funds = corr_matrix.columns.tolist()
            raise ValueError(
                f"Central fund {central_fund_code} not found in correlation data. "
                f"Available funds: {available_funds[:5]}... (total: {len(available_funds)})"
            )
        
        # Get valid funds available in correlation matrix
        valid_funds = set(corr_matrix.columns)
        
        def get_valid_funds_by_category(category_pattern):
            category_funds = funds[
                funds['scheme_category'].str.contains(category_pattern, na=False, regex=True)
            ]['code'].tolist()
            return [f for f in category_funds if f in valid_funds]
        
        # Time horizon based allocation and category selection
        if time_horizon == "Short Term (<5 years)":
            fund_weights = {'Debt': 0.6, 'Hybrid': 0.3, 'Equity': 0.1}
            categories = ['Debt', 'Hybrid']  # Focus on lower risk categories
        elif time_horizon == "Medium Term (5-10 years)":
            fund_weights = {'Equity': 0.5, 'Hybrid': 0.3, 'Debt': 0.2}
            categories = ['Equity', 'Hybrid']
        else:  # Long Term (10+ years)
            fund_weights = {'Equity': 0.7, 'Hybrid': 0.2, 'Debt': 0.1}
            categories = ['Equity', 'Hybrid']
        
        def find_low_volatility_funds(candidate_funds, n=top_n):
            if not candidate_funds:
                return []
            try:
                # Get volatility for candidate funds and return least volatile ones
                fund_volatility = volatility[candidate_funds]
                return fund_volatility.nsmallest(n).index.tolist()
            except KeyError:
                return []
        
        def find_low_corr_funds(target_fund, candidate_funds, n=top_n):
            if not candidate_funds or target_fund not in corr_matrix.index:
                return []
            try:
                correlations = corr_matrix.loc[target_fund, candidate_funds]
                return correlations.nsmallest(n).index.tolist()
            except KeyError:
                return []
        
        # Build suggestions with focus on low volatility funds in relevant categories
        suggestions = {'Central Fund': [central_fund_code]}
        
        # Get suggestions for each relevant category
        for category in categories:
            category_funds = get_valid_funds_by_category(f'^{category}')
            if category_funds:
                # First find low volatility funds in category
                low_vol_funds = find_low_volatility_funds(category_funds)
                # Then find funds among these that have low correlation with central fund
                if low_vol_funds:
                    low_corr_funds = find_low_corr_funds(central_fund_code, low_vol_funds)
                    if low_corr_funds:
                        suggestions[category] = low_corr_funds
        
        return suggestions, fund_weights
        
    except Exception as e:
        st.error(f"Suggestion error: {str(e)}")
        return {}, {}

# Enhanced risk metrics calculation
def calculate_risk_metrics(portfolio, returns_data):
    if not portfolio or len(portfolio) < 2:
        return {}
    
    try:
        portfolio_returns = returns_data[portfolio].mean(axis=1)
        return {
            'Annual Volatility': portfolio_returns.std() * np.sqrt(252),
            'Max Drawdown': (portfolio_returns.cummax() - portfolio_returns).max(),
            'Sharpe Ratio': portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252),
            'Positive Months': (portfolio_returns > 0).mean()
        }
    except Exception as e:
        st.error(f"Risk calculation failed: {str(e)}")
        return {}

# Main function with improved workflow
def main():
    st.set_page_config(page_title="Portfolio Builder", layout="wide")
    st.title("🏗️ Smart Portfolio Builder")
    
    # User inputs - first step
    st.sidebar.header("Portfolio Configuration")
    
    # Step 1: Select analysis period
    analysis_period = st.sidebar.selectbox(
        "Analysis Period",
        options=["All available", "YTD", "1 year", "3 years", "5 years", "10 years"],
        index=3,  # Default to 3 years
        help="Time period for historical analysis"
    )
    
    # Load data with progress indicators
    with st.spinner(f"Loading {analysis_period} fund data..."):
        funds, nav_data = get_fund_data(analysis_period)
        
        if funds.empty or nav_data.empty:
            st.error("Failed to load required data. Please check database connection.")
            st.stop()
    
    with st.spinner("Calculating metrics..."):
        corr_matrix, returns_data, volatility_data = calculate_metrics(nav_data)
        
        if corr_matrix.empty:
            st.error("Insufficient data to calculate metrics. Need at least 50 data points per fund.")
            st.stop()
    
    # Step 2: Select central fund
    central_fund = st.sidebar.selectbox(
        "Select Central Fund",
        options=funds['scheme_name'].unique(),
        index=0,
        help="This will be your primary fund for correlation analysis"
    )
    
    # Step 3: Select time horizon
    time_horizon = st.sidebar.selectbox(
        "Investment Horizon",
        options=["Short Term (<5 years)", "Medium Term (5-10 years)", "Long Term (10+ years)"],
        index=1,
        help="Affects the recommended asset allocation"
    )
    
    # Additional option for minimum volatility filter
    min_vol_filter = st.sidebar.checkbox(
        "Prioritize low volatility funds",
        value=True,
        help="When enabled, selects funds with lower volatility within each category"
    )
    
    if st.sidebar.button("Build Optimal Portfolio", help="Generate recommendations based on your selections"):
        with st.spinner("Constructing optimal portfolio..."):
            try:
                central_fund_code = funds[funds['scheme_name'] == central_fund]['code'].values[0]
                
                # Get portfolio suggestions
                portfolio, weights = suggest_portfolio(
                    central_fund_code, 
                    time_horizon, 
                    corr_matrix, 
                    funds,
                    volatility_data
                )
                
                # Flatten and validate portfolio
                all_funds = []
                for funds_list in portfolio.values():
                    if isinstance(funds_list, list):
                        all_funds.extend(f for f in funds_list if f in corr_matrix.columns)
                
                if len(all_funds) < 2:
                    st.error("Could not find enough suitable funds. Try a different central fund or analysis period.")
                    st.stop()
                
                # Display results
                st.success("Portfolio Built Successfully!")
                
                # Portfolio composition
                st.subheader("🧩 Recommended Portfolio Composition")
                cols = st.columns(len(portfolio))
                
                for (category, funds_list), col in zip(portfolio.items(), cols):
                    with col:
                        if funds_list and funds_list[0] in funds['code'].values:
                            fund_name = funds[funds['code'] == funds_list[0]]['scheme_name'].values[0]
                            st.metric(
                                label=category,
                                value=fund_name,
                                delta=f"{weights.get(category, 0)*100:.0f}% allocation"
                            )
                
                # Risk metrics
                st.subheader("📊 Portfolio Risk Metrics")
                risk_metrics = calculate_risk_metrics(all_funds, returns_data)
                
                if risk_metrics:
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Annual Volatility", f"{risk_metrics.get('Annual Volatility', 0):.2%}")
                    m2.metric("Max Drawdown", f"{risk_metrics.get('Max Drawdown', 0):.2%}")
                    m3.metric("Sharpe Ratio", f"{risk_metrics.get('Sharpe Ratio', 0):.2f}")
                    m4.metric("Positive Months", f"{risk_metrics.get('Positive Months', 0):.2%}")
                
                # Correlation matrix
                st.subheader("🔄 Fund Correlations")
                try:
                    portfolio_corr = corr_matrix.loc[all_funds, all_funds]
                    fig = px.imshow(
                        portfolio_corr,
                        text_auto=".2f",
                        color_continuous_scale='RdBu_r',
                        range_color=[-1, 1],
                        labels=dict(x="Fund", y="Fund", color="Correlation"),
                        title="Correlation Between Selected Funds"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not display correlation matrix: {str(e)}")
                
                # Performance chart
                st.subheader(f"📈 Portfolio Growth (Hypothetical ₹10,000 Investment over {analysis_period})")
                try:
                    portfolio_value = (1 + returns_data[all_funds]).cumprod() * 10000
                    fig = px.line(
                        portfolio_value, 
                        labels={'value': 'Portfolio Value', 'nav': 'Date'},
                        title=f"Growth of ₹10,000 Investment ({analysis_period})"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not display performance chart: {str(e)}")
                
                # Volatility information
                st.subheader("📉 Fund Volatility")
                try:
                    portfolio_vol = volatility_data[all_funds].sort_values()
                    fig = px.bar(
                        portfolio_vol,
                        labels={'value': 'Annualized Volatility', 'index': 'Fund'},
                        title="Volatility of Selected Funds"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not display volatility chart: {str(e)}")
                
                # Data summary
                with st.expander("🔍 View raw data"):
                    st.write("Available funds in correlation matrix:", corr_matrix.columns.tolist())
                    st.write("Selected funds:", all_funds)
                    st.write("Fund metadata:", funds)
                    st.write("Volatility data:", volatility_data)
                
            except Exception as e:
                st.error(f"Portfolio construction failed: {str(e)}")
                st.error("Please check the console for detailed error messages")

if __name__ == "__main__":
    main()