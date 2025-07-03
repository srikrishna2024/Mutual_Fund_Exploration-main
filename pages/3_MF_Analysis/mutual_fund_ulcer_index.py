import streamlit as st
import pandas as pd
import numpy as np
import psycopg
from datetime import datetime

# Database connection parameters
DB_PARAMS = {
    'dbname': 'postgres',
    'user': 'postgres',
    'password': 'admin123',
    'host': 'localhost',
    'port': '5432'
}

def connect_to_db():
    """Create database connection"""
    return psycopg.connect(**DB_PARAMS)

def get_categories():
    """Fetch unique scheme categories"""
    with connect_to_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT DISTINCT scheme_category 
                FROM mutual_fund_master_data 
                ORDER BY scheme_category;
            """)
            return [row[0] for row in cur.fetchall()]

def calculate_ulcer_index(nav_series, period=14):
    """Calculate Ulcer Index for a series of NAV values"""
    if len(nav_series) < period:
        return None
        
    # Calculate running maximum
    rolling_max = pd.Series(nav_series).expanding().max()
    
    # Calculate percentage drawdown
    drawdown = 100 * (nav_series - rolling_max) / rolling_max
    
    # Square the drawdowns (negative values only)
    squared_drawdown = drawdown.clip(upper=0) ** 2
    
    # Calculate Ulcer Index
    ulcer_index = np.sqrt(squared_drawdown.mean())
    
    return ulcer_index

def calculate_cagr(start_value, end_value, start_date, end_date):
    """
    Calculate CAGR given start value, end value, start date, and end date.
    """
    time_period = (end_date - start_date).days / 365.25
    cagr = (end_value / start_value) ** (1 / time_period) - 1
    return cagr * 100  # Convert to percentage

def get_fund_nav_data(category, lookback_days=90):
    """Fetch NAV data for funds in the selected category"""
    with connect_to_db() as conn:
        # First, let's check for the date range in the database
        date_range_query = """
        SELECT MIN(nav::date), MAX(nav::date)
        FROM mutual_fund_nav;
        """
        date_range_df = pd.read_sql(date_range_query, conn)
        st.write(f"Database NAV date range: {date_range_df.iloc[0, 0]} to {date_range_df.iloc[0, 1]}")
        
        # Main query for NAV data - get all available historical data
        query = """
        SELECT 
            m.scheme_name,
            n.nav::date as date,
            n.value::float as nav_value
        FROM mutual_fund_master_data m
        JOIN mutual_fund_nav n ON m.code = n.code
        WHERE m.scheme_category = %s
        AND n.value::float > 0
        ORDER BY m.scheme_name, n.nav::date;
        """
        
        df = pd.read_sql(query, conn, params=(category,))
        
        # Debug information
        st.write(f"Number of unique funds: {df['scheme_name'].nunique()}")
        st.write(f"Total NAV records: {len(df)}")
        
        if not df.empty:
            st.write(f"Data date range: {df['date'].min()} to {df['date'].max()}")
        
        return df

def main():
    st.set_page_config(page_title='Mutual Fund Ulcer Index Calculator', layout='wide')
    st.title('Mutual Fund Ulcer Index Calculator')

    # Sidebar inputs
    st.sidebar.header('Parameters')
    categories = get_categories()
    selected_category = st.sidebar.selectbox('Select Mutual Fund Category', categories)
    
    lookback_period = st.sidebar.slider(
        'Lookback Period (days)',
        min_value=30,
        max_value=365,
        value=90,
        step=30,
        help='Period for calculating Ulcer Index'
    )
    
    ui_period = st.sidebar.slider(
        'Ulcer Index Period (days)',
        min_value=14,
        max_value=90,
        value=14,
        step=1,
        help='Rolling period for Ulcer Index calculation'
    )
    
    calculate_button = st.sidebar.button('Calculate Ulcer Index')

    if calculate_button:
        with st.spinner('Calculating Ulcer Index...'):
            # Get NAV data
            df = get_fund_nav_data(selected_category, lookback_period)
            
            if df.empty:
                st.error('No data found for the selected category.')
                return
            
            # Calculate Ulcer Index and CAGR for each fund
            results = []
            for fund in df['scheme_name'].unique():
                fund_data = df[df['scheme_name'] == fund].copy()
                fund_data = fund_data.sort_values('date')
                
                # Use all available data points
                nav_values = fund_data['nav_value'].values
                dates = fund_data['date'].values
                
                if len(nav_values) >= ui_period:
                    ui = calculate_ulcer_index(nav_values, period=ui_period)
                    latest_nav = nav_values[-1]
                    initial_nav = nav_values[0]
                    start_date = dates[0]
                    end_date = dates[-1]
                    
                    # Calculate Absolute Return
                    absolute_return = ((latest_nav - initial_nav) / initial_nav) * 100
                    
                    # Calculate CAGR
                    cagr = calculate_cagr(initial_nav, latest_nav, start_date, end_date)
                    
                    results.append({
                        'Scheme Name': fund,
                        'Ulcer Index': ui,
                        'Latest NAV': latest_nav,
                        'Absolute Return (%)': absolute_return,
                        'CAGR (%)': cagr,
                        'Data Points': len(nav_values)
                    })
            
            if not results:
                st.error('No funds have sufficient data for the selected parameters.')
                return
                
            results_df = pd.DataFrame(results)
            
            # Format for display
            display_df = pd.DataFrame({
                'Scheme Name': results_df['Scheme Name'],
                'Ulcer Index': results_df['Ulcer Index'].apply(
                    lambda x: f"{x:.2f}" if pd.notnull(x) else "Insufficient data"
                ),
                'Latest NAV': results_df['Latest NAV'].apply(
                    lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A"
                ),
                'Absolute Return (%)': results_df['Absolute Return (%)'].apply(
                    lambda x: f"{x:.2f}%" if pd.notnull(x) else "N/A"
                ),
                'CAGR (%)': results_df['CAGR (%)'].apply(
                    lambda x: f"{x:.2f}%" if pd.notnull(x) else "N/A"
                ),
                'Data Points': results_df['Data Points']
            })
            
            # Display results
            st.subheader(f'Ulcer Index Analysis for {selected_category}')
            st.table(display_df)

            # Add download to CSV button
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name=f"ulcer_index_results_{selected_category}.csv",
                mime="text/csv",
            )

if __name__ == "__main__":
    main()