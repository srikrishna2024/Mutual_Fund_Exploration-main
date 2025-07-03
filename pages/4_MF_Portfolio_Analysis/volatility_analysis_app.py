import streamlit as st
import pandas as pd
import numpy as np
import psycopg
import plotly.express as px
import plotly.graph_objects as go

def connect_to_db():
    """Create database connection"""
    DB_PARAMS = {
        'dbname': 'postgres',
        'user': 'postgres',
        'password': 'admin123',
        'host': 'localhost',
        'port': '5432'
    }
    return psycopg.connect(**DB_PARAMS)

def get_portfolio_data():
    """Retrieve portfolio data"""
    with connect_to_db() as conn:
        query = """
            SELECT date, scheme_name, code, transaction_type, value, units, amount 
            FROM portfolio_data 
            ORDER BY date, scheme_name
        """
        return pd.read_sql(query, conn)

def get_latest_nav(portfolio_funds):
    """Retrieve the latest NAVs for portfolio funds"""
    with connect_to_db() as conn:
        query = """
            SELECT code, scheme_name, nav as date, value as nav_value
            FROM mutual_fund_nav
            WHERE (code, nav) IN (
                SELECT code, MAX(nav) AS latest_date
                FROM mutual_fund_nav
                WHERE code = ANY(%s)
                GROUP BY code
            )
        """
        return pd.read_sql(query, conn, params=(portfolio_funds,))

def get_historical_nav(portfolio_funds):
    """Retrieve historical NAV data"""
    with connect_to_db() as conn:
        query = """
            SELECT code, scheme_name, nav as date, value as nav_value
            FROM mutual_fund_nav
            WHERE code = ANY(%s)
            ORDER BY code, nav
        """
        return pd.read_sql(query, conn, params=(portfolio_funds,))

def calculate_fund_metrics(df, historical_nav, latest_nav):
    """Calculate volatility metrics for each fund"""
    # Get current units for each fund
    current_units = df.groupby('code')['units'].sum()
    portfolio_funds = current_units[current_units > 0].index.tolist()

    # Calculate current values using latest NAV
    current_values = latest_nav.set_index('code')['nav_value'] * current_units
    total_value = current_values.sum()
    weights = current_values / total_value

    # Calculate returns
    nav_data = historical_nav[historical_nav['code'].isin(portfolio_funds)]
    nav_pivot = nav_data.pivot(index='date', columns='code', values='nav_value')
    daily_returns = nav_pivot.pct_change()

    # Calculate volatility metrics
    volatility_metrics = pd.DataFrame()
    volatility_metrics['Current Value'] = current_values
    volatility_metrics['Daily Volatility'] = daily_returns.std()
    volatility_metrics['Annualized Volatility'] = volatility_metrics['Daily Volatility'] * np.sqrt(252)
    volatility_metrics['Weight'] = weights

    # Calculate covariance and correlation
    daily_covariance = daily_returns.cov()
    correlation_matrix = daily_returns.corr()

    # Calculate Marginal Contribution to Risk (MCR)
    portfolio_variance = np.dot(weights, np.dot(daily_covariance, weights))
    portfolio_volatility = np.sqrt(portfolio_variance)

    mcr = np.dot(daily_covariance, weights) / portfolio_volatility
    volatility_metrics['MCR'] = mcr

    # Total Contribution to Risk (TCR)
    tcr = weights * mcr
    volatility_metrics['TCR'] = tcr

    # Risk contribution percentage
    volatility_metrics['Risk Contribution %'] = (tcr / portfolio_volatility) * 100

    return volatility_metrics, correlation_matrix, portfolio_volatility * np.sqrt(252), total_value

def analyze_risk_factors(volatility_metrics):
    """Analyze why a fund has high risk contribution"""
    fund_analysis = {}
    for fund in volatility_metrics.index:
        weight = volatility_metrics.loc[fund, 'Weight']
        vol = volatility_metrics.loc[fund, 'Annualized Volatility']
        risk_contrib = volatility_metrics.loc[fund, 'Risk Contribution %']

        # Compare against averages
        avg_weight = volatility_metrics['Weight'].mean()
        avg_vol = volatility_metrics['Annualized Volatility'].mean()

        weight_factor = weight / avg_weight
        vol_factor = vol / avg_vol

        # Determine primary risk factor
        if weight_factor > vol_factor:
            primary_factor = "weight"
            factor_ratio = weight_factor
        else:
            primary_factor = "volatility"
            factor_ratio = vol_factor

        fund_analysis[fund] = {
            'primary_factor': primary_factor,
            'factor_ratio': factor_ratio,
            'weight_vs_avg': weight_factor,
            'vol_vs_avg': vol_factor
        }

    return fund_analysis

def format_indian_number(number):
    """
    Format a number in Indian style (lakhs, crores)
    """
    if number >= 10000000:  # crores
        return f"₹{number/10000000:.2f} Cr"
    elif number >= 100000:  # lakhs
        return f"₹{number/100000:.2f} L"
    elif number >= 1000:  # thousands
        return f"₹{number/1000:.2f} K"
    else:
        return f"₹{number:.2f}"

def main():
    """
    Main function to run the Portfolio Volatility Analysis application.
    This function sets up the Streamlit page configuration, loads portfolio data,
    calculates volatility metrics, and displays various analyses and visualizations
    related to portfolio risk and fund performance.
    The function performs the following steps:
    1. Sets the page title and layout.
    2. Loads portfolio data and checks for its availability.
    3. Loads historical NAV data for the portfolio funds.
    4. Calculates volatility metrics, correlation matrix, and portfolio volatility.
    5. Displays portfolio risk overview including annualized volatility and highest risk contribution.
    6. Displays individual fund analysis with metrics and primary risk factors.
    7. Visualizes risk contribution analysis using bar charts.
    8. Displays fund correlation analysis using a heatmap.
    9. Handles exceptions and displays error messages if any issues occur.
    Raises:
        Exception: If there is an error in loading data, calculating metrics, or any other step.
    """
    st.set_page_config(page_title="Portfolio Volatility Analysis", layout="wide")
    st.title("Fund Volatility and Risk Analysis Dashboard")

    try:
        # Load data
        df = get_portfolio_data()
        if df.empty:
            st.warning("No portfolio data found.")
            return

        portfolio_funds = df.groupby('code')['units'].sum()
        portfolio_funds = portfolio_funds[portfolio_funds > 0].index.tolist()

        latest_nav = get_latest_nav(portfolio_funds)
        historical_nav = get_historical_nav(portfolio_funds)
        if historical_nav.empty or latest_nav.empty:
            st.warning("No NAV data found.")
            return

        # Calculate metrics
        volatility_metrics, correlation_matrix, portfolio_volatility, total_value = calculate_fund_metrics(df, historical_nav, latest_nav)

        # Display Portfolio Overview
        st.header("Portfolio Overview")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Portfolio Value", format_indian_number(total_value))

        with col2:
            st.metric("Portfolio Annualized Volatility", f"{portfolio_volatility:.2f}%")

        with col3:
            highest_risk_fund = volatility_metrics['Risk Contribution %'].idxmax()
            st.metric(
                "Highest Risk Contribution", 
                f"{volatility_metrics.loc[highest_risk_fund, 'Risk Contribution %']:.2f}%", 
                f"from {historical_nav[historical_nav['code'] == highest_risk_fund]['scheme_name'].iloc[0]}"
            )

        # Individual Fund Analysis
        st.header("Individual Fund Analysis")

        # Get scheme names for display
        scheme_names = historical_nav.drop_duplicates('code').set_index('code')['scheme_name']
        volatility_metrics['Scheme Name'] = volatility_metrics.index.map(scheme_names)
        volatility_metrics = volatility_metrics.set_index('Scheme Name')

        # Add Primary Risk Factor Analysis
        risk_factors = analyze_risk_factors(volatility_metrics)
        volatility_metrics['Primary Risk Factor'] = [risk_factors[code]['primary_factor'] for code in volatility_metrics.index]

        # Format metrics for display
        display_metrics = volatility_metrics.copy()
        display_metrics['Current Value'] = display_metrics['Current Value'].apply(format_indian_number)
        for col in ['Daily Volatility', 'Annualized Volatility', 'MCR', 'TCR', 'Risk Contribution %']:
            display_metrics[col] = display_metrics[col].map('{:.2f}%'.format)
        display_metrics['Weight'] = display_metrics['Weight'].map('{:.2f}%'.format)

        # Reorder columns for better display
        display_metrics = display_metrics[[
            'Current Value', 'Weight', 'Daily Volatility', 'Annualized Volatility',
            'Risk Contribution %', 'Primary Risk Factor'
        ]]

        st.dataframe(display_metrics)

        st.info("""
        **How to Interpret the Results:**
        - **Current Value:** Current market value of the fund holding
        - **Weight:** Proportion of portfolio value held in the fund
        - **Annualized Volatility:** Risk level of the fund
        - **Risk Contribution %:** Indicates the fund's contribution to overall portfolio risk
        - **Primary Risk Factor:** Highlights whether the fund's weight or volatility drives its risk contribution
        """)

        # Risk Contribution Visualization
        st.header("Risk Contribution Analysis")

        fig = go.Figure(data=[
            go.Bar(name='Weight', 
                  x=display_metrics.index, 
                  y=volatility_metrics['Weight'] * 100),
            go.Bar(name='Risk Contribution', 
                  x=display_metrics.index, 
                  y=volatility_metrics['Risk Contribution %'])
        ])

        fig.update_layout(
            barmode='group',
            title="Fund Weights vs Risk Contribution",
            xaxis_title="Fund Name",
            yaxis_title="Percentage (%)",
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

        # Correlation Analysis
        st.header("Fund Correlation Analysis")

        correlation_display = correlation_matrix.copy()
        correlation_display.index = correlation_display.index.map(scheme_names)
        correlation_display.columns = correlation_display.index

        fig = px.imshow(correlation_display,
                       labels=dict(x="Fund", y="Fund", color="Correlation"),
                       color_continuous_scale="RdBu_r",
                       aspect="auto")

        fig.update_layout(
            title="Fund Correlation Heatmap",
            height=600
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please check your database connection and data integrity.")

if __name__ == "__main__":
    main()