import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import psycopg
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

def get_fund_list():
    """Fetch list of mutual funds (scheme name and code)"""
    with connect_to_db() as conn:
        query = """
        SELECT DISTINCT code, scheme_name
        FROM mutual_fund_master_data
        ORDER BY scheme_name;
        """
        df = pd.read_sql(query, conn)
        return df

def get_fund_nav_data(code, scheme_name):
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
        # Store the scheme name as a column instead of DataFrame attribute
        df['scheme_name'] = scheme_name
        return df

def get_benchmark_data():
    """Fetch historical benchmark data"""
    with connect_to_db() as conn:
        query = """
        SELECT date, price
        FROM benchmark
        ORDER BY date;
        """
        df = pd.read_sql(query, conn)
        df['date'] = pd.to_datetime(df['date'])
        df['price'] = df['price'].astype(float)
        return df

def calculate_rolling_returns(df, period_years):
    """Calculate rolling CAGR for a given period"""
    df = df.copy()
    df.set_index('date', inplace=True)
    
    # Drop duplicate dates if any
    df = df[~df.index.duplicated(keep='first')]
    
    df = df.resample('D').ffill()  # Fill missing dates with the last available value
    periods = int(365 * period_years)  # Ensure periods is an integer
    rolling_returns = df['value'].pct_change(periods=periods)  # Calculate rolling returns
    rolling_cagr = (1 + rolling_returns) ** (1 / period_years) - 1  # Convert to CAGR
    return rolling_cagr.dropna()

def calculate_benchmark_rolling_returns(df, period_years):
    """Calculate rolling CAGR for benchmark data"""
    df = df.copy()
    df.set_index('date', inplace=True)
    
    # Drop duplicate dates if any
    df = df[~df.index.duplicated(keep='first')]
    
    df = df.resample('D').ffill()  # Fill missing dates with the last available value
    periods = int(365 * period_years)  # Ensure periods is an integer
    rolling_returns = df['price'].pct_change(periods=periods)  # Calculate rolling returns
    rolling_cagr = (1 + rolling_returns) ** (1 / period_years) - 1  # Convert to CAGR
    return rolling_cagr.dropna()

def plot_rolling_returns(fund_data, benchmark_data, period_name, period_years, scheme_name):
    """Plot rolling returns for a given period using Plotly"""
    # Calculate rolling returns for the fund
    fund_rolling_cagr = calculate_rolling_returns(
        fund_data.rename(columns={'nav': 'date', 'value': 'value'}),
        period_years
    )
    # Calculate rolling returns for the benchmark
    benchmark_rolling_cagr = calculate_benchmark_rolling_returns(
        benchmark_data,
        period_years
    )

    # Create Plotly figure
    fig = go.Figure()
    
    # Add fund trace with a distinct color (blue)
    fig.add_trace(go.Scatter(
        x=fund_rolling_cagr.index,
        y=fund_rolling_cagr * 100,
        name=f'{scheme_name} Rolling CAGR',
        line=dict(color='#1f77b4', width=2),  # Blue color
        mode='lines'
    ))
    
    # Add benchmark trace with a contrasting color (orange)
    fig.add_trace(go.Scatter(
        x=benchmark_rolling_cagr.index,
        y=benchmark_rolling_cagr * 100,
        name='Benchmark Rolling CAGR',
        line=dict(color='#ff7f0e', width=2),  # Orange color
        mode='lines'
    ))
    
    # Update layout
    fig.update_layout(
        title=f'{period_name} Rolling Returns: {scheme_name} vs Benchmark',
        xaxis_title='Start Date of Rolling Period',
        yaxis_title='CAGR (%)',
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500,
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_multiple_funds_rolling_returns(fund_data_list, benchmark_data, period_name, period_years):
    """Plot rolling returns for multiple funds vs benchmark using Plotly"""
    # Create Plotly figure
    fig = go.Figure()
    
    # Define a color palette with distinct colors
    colors = [
        '#1f77b4',  # Blue
        '#2ca02c',  # Green
        '#d62728',  # Red
        '#9467bd',  # Purple
        '#8c564b',  # Brown
        '#e377c2',  # Pink
        '#7f7f7f',  # Gray
        '#bcbd22',  # Olive
        '#17becf'   # Teal
    ]
    
    # Add benchmark trace first (orange)
    benchmark_rolling_cagr = calculate_benchmark_rolling_returns(benchmark_data, period_years)
    fig.add_trace(go.Scatter(
        x=benchmark_rolling_cagr.index,
        y=benchmark_rolling_cagr * 100,
        name='Benchmark Rolling CAGR',
        line=dict(color='#ff7f0e', width=2),  # Orange color
        mode='lines'
    ))
    
    # Add traces for each fund with distinct colors
    for idx, (fund_data, fund_name) in enumerate(fund_data_list):
        fund_rolling_cagr = calculate_rolling_returns(
            fund_data.rename(columns={'nav': 'date', 'value': 'value'}),
            period_years
        )
        fig.add_trace(go.Scatter(
            x=fund_rolling_cagr.index,
            y=fund_rolling_cagr * 100,
            name=f'{fund_name} Rolling CAGR',
            line=dict(color=colors[idx % len(colors)], width=2),
            mode='lines'
        ))
    
    # Update layout
    fig.update_layout(
        title=f'{period_name} Rolling Returns: Funds vs Benchmark',
        xaxis_title='Start Date of Rolling Period',
        yaxis_title='CAGR (%)',
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500,
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def main():
    st.set_page_config(page_title='Mutual Fund Rolling Returns Analyzer', layout='wide')
    st.title('Mutual Fund Rolling Returns Analyzer')

    # Create tabs
    tab1, tab2 = st.tabs(["Single Fund vs Benchmark", "Compare 3 Funds vs Benchmark"])

    with tab1:
        st.header("Single Fund vs Benchmark")
        # Sidebar for inputs
        st.sidebar.header('Select Fund')
        fund_list = get_fund_list()
        selected_fund = st.sidebar.selectbox('Select Mutual Fund', fund_list['scheme_name'], key='single_fund')
        selected_code = fund_list[fund_list['scheme_name'] == selected_fund]['code'].values[0]

        analyze_button = st.sidebar.button('Analyze Fund', key='analyze_single')

        if analyze_button:
            with st.spinner('Analyzing fund and benchmark rolling returns...'):
                # Fetch historical NAV data for the selected fund
                fund_nav_data = get_fund_nav_data(selected_code, selected_fund)
                if fund_nav_data.empty:
                    st.error('No data found for the selected fund.')
                    return

                # Fetch historical benchmark data
                benchmark_data = get_benchmark_data()
                if benchmark_data.empty:
                    st.error('No benchmark data found.')
                    return

                # Define rolling periods (only 1, 3, 5, 7, and 10 years)
                periods = {
                    '1 Year': 1,
                    '3 Years': 3,
                    '5 Years': 5,
                    '7 Years': 7,
                    '10 Years': 10
                }

                # Calculate and plot rolling returns for each period
                for period_name, period_years in periods.items():
                    plot_rolling_returns(fund_nav_data, benchmark_data, period_name, period_years, selected_fund)

    with tab2:
        st.header("Compare 3 Funds vs Benchmark")
        # Sidebar for inputs
        st.sidebar.header('Select Funds')
        fund_list = get_fund_list()
        selected_funds = st.sidebar.multiselect(
            'Select up to 3 Mutual Funds', 
            fund_list['scheme_name'], 
            default=None, 
            key='multi_fund',
            max_selections=3
        )

        analyze_button = st.sidebar.button('Analyze Funds', key='analyze_multi')

        if analyze_button and len(selected_funds) > 0:
            with st.spinner('Analyzing funds and benchmark rolling returns...'):
                # Fetch historical benchmark data
                benchmark_data = get_benchmark_data()
                if benchmark_data.empty:
                    st.error('No benchmark data found.')
                    return

                # Define rolling periods (only 1, 3, 5, 7, and 10 years)
                periods = {
                    '1 Year': 1,
                    '3 Years': 3,
                    '5 Years': 5,
                    '7 Years': 7,
                    '10 Years': 10
                }

                # Fetch historical NAV data for the selected funds
                fund_data_list = []
                for fund_name in selected_funds:
                    fund_code = fund_list[fund_list['scheme_name'] == fund_name]['code'].values[0]
                    fund_nav_data = get_fund_nav_data(fund_code, fund_name)
                    if fund_nav_data.empty:
                        st.error(f'No data found for the selected fund: {fund_name}.')
                        return
                    fund_data_list.append((fund_nav_data, fund_name))

                # Calculate and plot rolling returns for each period
                for period_name, period_years in periods.items():
                    plot_multiple_funds_rolling_returns(fund_data_list, benchmark_data, period_name, period_years)

if __name__ == "__main__":
    main()