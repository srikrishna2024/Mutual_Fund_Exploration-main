import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import psycopg2
from psycopg2 import sql

# Database connection parameters
DB_PARAMS = {
    'dbname': 'postgres',
    'user': 'postgres',
    'password': 'admin123',
    'host': 'localhost',
    'port': '5432'
}

# Define ideal antifragile metric ranges
IDEAL_RANGES = {
    'Volatility': '< 10%',
    'Max Drawdown': '< -15%',
    'Recovery Time': '< 180 days',
    'Gain/Loss Ratio': '> 1.5'
}

# UI
st.title("Antifragile Mutual Fund Analyzer")
st.markdown("""
Select a mutual fund category and time period to analyze antifragile metrics. The top 5 funds will be recommended based on resilience to downside and recovery ability.
""")

def get_db_connection():
    """Establish connection to PostgreSQL database"""
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        return conn
    except psycopg2.Error as e:
        st.error(f"Database connection failed: {e}")
        return None

def get_available_categories(conn):
    """Get distinct fund categories from the database"""
    try:
        query = "SELECT DISTINCT scheme_category FROM mutual_fund_master_data ORDER BY scheme_category"
        categories_df = pd.read_sql_query(query, conn)
        return categories_df['scheme_category'].tolist()
    except Exception as e:
        st.error(f"Error fetching categories: {e}")
        return []

# Connect to database and get available categories
conn = get_db_connection()
if conn is None:
    st.stop()

# Get all available categories from database
categories = get_available_categories(conn)
if not categories:
    st.error("No fund categories found in the database.")
    conn.close()
    st.stop()

# UI Elements - Now shows all available categories from database
category = st.selectbox("Select Fund Category", categories)
time_period = st.selectbox("Select Time Period", ["YTD", "1Y", "2Y", "3Y", "5Y", "10Y", "Max"])

if st.button("Analyze"):
    try:
        # First get all funds in the selected category from master data
        query = """
            SELECT code, scheme_name 
            FROM mutual_fund_master_data 
            WHERE scheme_category = %s
        """
        master_df = pd.read_sql_query(query, conn, params=(category,))
        
        if master_df.empty:
            st.warning(f"No funds found in the {category} category")
            conn.close()
            st.stop()

        # Get NAV data for these funds
        fund_codes = tuple(master_df['code'].unique())
        if len(fund_codes) == 1:
            # Handle case with single fund
            query = """
                SELECT code, scheme_name, nav as date, value as nav 
                FROM mutual_fund_nav 
                WHERE code = %s
            """
            df = pd.read_sql_query(query, conn, params=(fund_codes[0],))
        else:
            query = """
                SELECT code, scheme_name, nav as date, value as nav 
                FROM mutual_fund_nav 
                WHERE code IN %s
            """
            df = pd.read_sql_query(query, conn, params=(fund_codes,))
        
        # Merge with master data to get category info
        df = pd.merge(df, master_df, on=['code', 'scheme_name'])
        
        # Convert dates
        df['date'] = pd.to_datetime(df['date'])
        latest_date = df['date'].max()

        # Time filtering
        def filter_period(df, period):
            if period == "YTD":
                start = datetime(latest_date.year, 1, 1)
            elif period == "1Y":
                start = latest_date - timedelta(days=365)
            elif period == "2Y":
                start = latest_date - timedelta(days=730)
            elif period == "3Y":
                start = latest_date - timedelta(days=1095)
            elif period == "5Y":
                start = latest_date - timedelta(days=1825)
            elif period == "10Y":
                start = latest_date - timedelta(days=3650)
            else:
                return df
            return df[df['date'] >= start]

        df = filter_period(df, time_period)

        # Antifragile Metrics Calculations
        metrics = []
        for fund in df['scheme_name'].unique():
            fund_df = df[df['scheme_name'] == fund].sort_values("date")
            nav = fund_df['nav'].values

            # Volatility
            daily_returns = np.diff(nav) / nav[:-1]
            vol = np.std(daily_returns) * np.sqrt(252) * 100

            # Max Drawdown
            peak = nav[0]
            max_dd = 0
            drawdowns = []
            for val in nav:
                if val > peak:
                    peak = val
                dd = (val - peak) / peak
                drawdowns.append(dd)
                max_dd = min(max_dd, dd)

            # Recovery Time
            drawdown_df = pd.DataFrame({"date": fund_df['date'].values, "dd": drawdowns})
            drawdown_df['recovered'] = drawdown_df['dd'] == 0
            recovery_periods = drawdown_df[drawdown_df['recovered']].index.to_series().diff().dropna()
            avg_recovery = recovery_periods.mean() if not recovery_periods.empty else np.nan

            # Gain/Loss Ratio
            gains = daily_returns[daily_returns > 0]
            losses = -daily_returns[daily_returns < 0]
            gl_ratio = gains.mean() / losses.mean() if losses.mean() != 0 else np.nan

            metrics.append({
                "Fund": fund,
                "Volatility": vol,
                "Max Drawdown": max_dd * 100,
                "Recovery Time (days)": avg_recovery if not np.isnan(avg_recovery) else -1,
                "Gain/Loss Ratio": gl_ratio
            })

        result_df = pd.DataFrame(metrics)
        result_df = result_df.sort_values(by=["Max Drawdown", "Recovery Time (days)", "Volatility"], ascending=[False, True, True])

        st.subheader("Top 5 Recommended Funds")
        st.dataframe(result_df.head(5).style.format({
            "Volatility": "{:.2f}%",
            "Max Drawdown": "{:.2f}%",
            "Recovery Time (days)": "{:.0f}",
            "Gain/Loss Ratio": "{:.2f}"
        }))

        st.subheader("All Fund Metrics")
        st.dataframe(result_df.style.format({
            "Volatility": "{:.2f}%",
            "Max Drawdown": "{:.2f}%",
            "Recovery Time (days)": "{:.0f}",
            "Gain/Loss Ratio": "{:.2f}"
        }))

        st.subheader("Ideal Metric Ranges (Antifragile Reference)")
        st.write(IDEAL_RANGES)

    except Exception as e:
        st.error(f"An error occurred during analysis: {e}")
    finally:
        conn.close()