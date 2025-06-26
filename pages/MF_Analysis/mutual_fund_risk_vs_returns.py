import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
import psycopg

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

def get_categories():
    """Fetch unique mutual fund categories"""
    with connect_to_db() as conn:
        query = """
        SELECT DISTINCT scheme_category
        FROM mutual_fund_master_data
        ORDER BY scheme_category;
        """
        df = pd.read_sql(query, conn)
        return df['scheme_category'].tolist()

def get_funds_in_category(category):
    """Fetch funds in the selected category"""
    with connect_to_db() as conn:
        query = """
        SELECT code, scheme_name
        FROM mutual_fund_master_data
        WHERE scheme_category = %s
        ORDER BY scheme_name;
        """
        df = pd.read_sql(query, conn, params=(category,))
        return df

def get_fund_nav_data(code):
    """Fetch historical NAV data for a specific fund"""
    with connect_to_db() as conn:
        query = """
        SELECT nav, value
        FROM mutual_fund_nav
        WHERE code = %s
        ORDER BY nav;
        """
        df = pd.read_sql(query, conn, params=(code,))
        df['nav'] = pd.to_datetime(df['nav'])
        df['value'] = df['value'].astype(float)
        return df

def calculate_rolling_returns(df, period_years):
    """Calculate rolling CAGR for a given period"""
    df = df.copy()
    df.set_index('date', inplace=True)
    df = df.resample('D').ffill()  # Fill missing dates with the last available value
    periods = int(365 * period_years)  # Ensure periods is an integer
    rolling_returns = df['value'].pct_change(periods=periods)  # Calculate rolling returns
    rolling_cagr = (1 + rolling_returns) ** (1 / period_years) - 1  # Convert to CAGR
    return rolling_cagr.dropna()

def calculate_rolling_std(df, period_years):
    """Calculate rolling standard deviation for a given period"""
    df = df.copy()
    df.set_index('date', inplace=True)
    df = df.resample('D').ffill()  # Fill missing dates with the last available value
    periods = int(365 * period_years)  # Ensure periods is an integer
    rolling_std = df['value'].pct_change().rolling(window=periods).std()  # Calculate rolling std
    return rolling_std.dropna()

def main():
    st.set_page_config(page_title='Fund Risk-Return Plot', layout='wide')
    st.title('Fund Risk-Return Plot (Log Scale)')

    # Sidebar for inputs
    st.sidebar.header('Select Mutual Fund Category')
    categories = get_categories()
    selected_category = st.sidebar.selectbox('Select Category', categories)

    analyze_button = st.sidebar.button('Analyze Category')

    if analyze_button:
        with st.spinner('Analyzing funds risk-return profiles...'):
            # Fetch funds in the selected category
            funds_in_category = get_funds_in_category(selected_category)
            if funds_in_category.empty:
                st.error(f'No funds found in the selected category: {selected_category}.')
                return

            # Define rolling periods (1, 2, 3, 4, 5, and 10 years)
            periods = {
                '1 Year': 1,
                '2 Years': 2,
                '3 Years': 3,
                '4 Years': 4,
                '5 Years': 5,
                '10 Years': 10
            }

            # Initialize a DataFrame to store results
            results = pd.DataFrame(columns=['Fund', 'Period', 'Rolling CAGR', 'Rolling Std Dev'])

            # Calculate rolling returns and standard deviations for each fund
            for _, fund in funds_in_category.iterrows():
                fund_code = fund['code']
                fund_name = fund['scheme_name']
                fund_nav_data = get_fund_nav_data(fund_code)
                if fund_nav_data.empty:
                    st.warning(f'No NAV data found for fund: {fund_name}. Skipping...')
                    continue

                for period_name, period_years in periods.items():
                    # Calculate rolling returns for the fund
                    fund_rolling_cagr = calculate_rolling_returns(
                        fund_nav_data.rename(columns={'nav': 'date', 'value': 'value'}),
                        period_years
                    )
                    # Calculate rolling standard deviation for the fund
                    fund_rolling_std = calculate_rolling_std(
                        fund_nav_data.rename(columns={'nav': 'date', 'value': 'value'}),
                        period_years
                    )

                    # Calculate median rolling CAGR and standard deviation
                    median_cagr = np.median(fund_rolling_cagr) * 100  # Convert to percentage
                    median_std = np.median(fund_rolling_std) * 100  # Convert to percentage

                    # Ensure positive values for log scale
                    median_cagr = max(median_cagr, 0.01)  # Avoid zero or negative values
                    median_std = max(median_std, 0.01)  # Avoid zero or negative values

                    # Append results
                    results = pd.concat(
                        [results, pd.DataFrame({
                            'Fund': [fund_name],
                            'Period': [period_name],
                            'Rolling CAGR': [median_cagr],
                            'Rolling Std Dev': [median_std]
                        })],
                        ignore_index=True
                    )

            # Create separate plots for each time period
            for period_name in periods.keys():
                st.subheader(f'Risk-Return Plot ({period_name})')
                period_results = results[results['Period'] == period_name]

                # Calculate +10% of the maximum value for the x-axis
                max_cagr = period_results['Rolling CAGR'].max()
                x_axis_limit = max_cagr * 1.10  # +10% of the maximum value

                # Plot risk-return scatter plot with log scale
                fig = px.scatter(
                    period_results,
                    x='Rolling CAGR',
                    y='Rolling Std Dev',
                    hover_name='Fund',
                    labels={
                        'Rolling CAGR': 'Rolling CAGR (%)',
                        'Rolling Std Dev': 'Rolling Standard Deviation (%)'
                    },
                    title=f'Risk-Return Plot ({period_name})',
                    log_x=True,  # Log scale on x-axis
                    log_y=True   # Log scale on y-axis
                )
                # Set x-axis limit to +10% of the maximum value
                fig.update_xaxes(range=[0.01, x_axis_limit])
                st.plotly_chart(fig, use_container_width=True)

            # Display interpretation information
            st.subheader('How to Interpret the Risk-Return Plot')
            st.write("""
            - **X-Axis (Rolling CAGR)**: The median rolling Compound Annual Growth Rate (CAGR) for the fund over the selected period (log scale).
            - **Y-Axis (Rolling Std Dev)**: The median rolling standard deviation (risk) for the fund over the selected period (log scale).
            - **Upper-Left Quadrant**: Funds with **lower risk** and **higher returns**.
            - **Upper-Right Quadrant**: Funds with higher risk and higher returns.
            - **Lower-Left Quadrant**: Funds with lower risk and lower returns.
            - **Lower-Right Quadrant**: Funds with higher risk and lower returns.
            """)

if __name__ == "__main__":
    main()