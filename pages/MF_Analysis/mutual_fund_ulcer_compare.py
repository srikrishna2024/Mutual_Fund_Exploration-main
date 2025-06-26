import streamlit as st
import pandas as pd
import numpy as np
import psycopg
import plotly.express as px
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
    return psycopg.connect(**DB_PARAMS)

def get_funds():
    """Fetch unique mutual fund names."""
    with connect_to_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT DISTINCT scheme_name FROM mutual_fund_master_data ORDER BY scheme_name;
            """)
            return [row[0] for row in cur.fetchall()]

def get_nav_data(fund_name):
    """Fetch NAV data for the selected fund."""
    with connect_to_db() as conn:
        query = """
        SELECT n.nav::date as date, n.value::float as nav_value
        FROM mutual_fund_nav n
        JOIN mutual_fund_master_data m ON n.code = m.code
        WHERE m.scheme_name = %s
        ORDER BY n.nav::date;
        """
        df = pd.read_sql(query, conn, params=(fund_name,))
        return df

def calculate_ulcer_index(nav_series, period=14):
    """Calculate rolling Ulcer Index."""
    rolling_max = nav_series.expanding().max()
    drawdown = 100 * (nav_series - rolling_max) / rolling_max
    squared_drawdown = drawdown.clip(upper=0) ** 2
    ulcer_index = np.sqrt(squared_drawdown.rolling(window=period).mean())
    return ulcer_index

def main():
    st.set_page_config(page_title='Ulcer Index Trend', layout='wide')
    st.title('Mutual Fund Ulcer Index Trend')
    
    # Sidebar user inputs
    funds = get_funds()
    selected_funds = st.sidebar.multiselect('Select Up to 3 Mutual Funds', funds, max_selections=3)
    ui_period = st.sidebar.slider('Ulcer Index Period (days)', min_value=14, max_value=100, value=50, step=1)
    analyze_button = st.sidebar.button('Analyze')
    
    if analyze_button and len(selected_funds) == 3:
        with st.spinner('Fetching data and calculating Ulcer Index...'):
            data_frames = []
            for fund in selected_funds:
                df = get_nav_data(fund)
                if not df.empty:
                    df['Ulcer Index'] = calculate_ulcer_index(df['nav_value'], period=ui_period)
                    df['Scheme Name'] = fund
                    data_frames.append(df)
            
            if not data_frames:
                st.error('No NAV data found for the selected funds.')
                return
            
            # Merge all data frames and find the common date range
            merged_df = pd.concat(data_frames)
            common_dates = merged_df.groupby('date').filter(lambda x: len(x) == 3)['date'].unique()
            merged_df = merged_df[merged_df['date'].isin(common_dates)]
            
            # Plot with Plotly
            fig = px.line(merged_df, x='date', y='Ulcer Index', color='Scheme Name', title='Ulcer Index Trend Comparison',
                          width=1000, height=600)
            fig.update_layout(legend=dict(orientation='h', yanchor='bottom', y=-0.3, xanchor='center', x=0.5))
            st.plotly_chart(fig)
            
            # Interpretation of Ulcer Index
            st.markdown("""
            **Ulcer Index Interpretation:**
            - **Low Values (Below 5):** Indicates low drawdowns and less volatility.
            - **Moderate Values (5-10):** Moderate drawdowns, meaning some fluctuations.
            - **High Values (Above 10):** Indicates high downside risk and significant drawdowns.
            """)
            
            # Downloadable data
            csv = merged_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Data as CSV",
                data=csv,
                file_name="ulcer_index_comparison.csv",
                mime="text/csv",
            )
    elif analyze_button:
        st.error("Please select exactly 3 mutual funds for comparison.")

if __name__ == "__main__":
    main()
