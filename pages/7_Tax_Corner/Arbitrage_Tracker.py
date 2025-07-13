import streamlit as st
import pandas as pd
import psycopg
import plotly.express as px
from datetime import datetime, timedelta

st.set_page_config(page_title="Arbitrage Cash Flow Tracker", layout="wide")
st.title("📊 Arbitrage Mutual Fund - Monthly Cash Flow & LTCG Tracker")

# DB connection
def connect_to_db():
    return psycopg.connect(
        dbname='postgres', user='postgres', password='admin123', host='localhost', port='5432'
    )

# Fetch Arbitrage Funds from goals table
@st.cache_data
def get_arbitrage_funds_from_goals():
    with connect_to_db() as conn:
        query = """
        SELECT DISTINCT m.code, m.scheme_name
        FROM mutual_fund_master_data m
        JOIN goals g ON m.code = g.scheme_code
        WHERE m.scheme_category = 'Hybrid Scheme - Arbitrage Fund' AND g.goal_name = 'Regular Cash'
        ORDER BY m.scheme_name
        """
        df = pd.read_sql(query, conn)
        return df

funds_df = get_arbitrage_funds_from_goals()

if funds_df.empty:
    st.error("❌ No arbitrage funds found under the 'Regular Cash' goal. Please check your data.")
    st.stop()

selected_fund = st.selectbox("Select Arbitrage Fund", funds_df['scheme_name'].tolist())
selected_code_row = funds_df.loc[funds_df['scheme_name'] == selected_fund]
if selected_code_row.empty:
    st.error("❌ Selected fund not found in the data. Please try again.")
    st.stop()
selected_code = selected_code_row['code'].values[0]

# NAV History
@st.cache_data
def get_nav_history(code):
    with connect_to_db() as conn:
        query = """
        SELECT nav AS nav_date, value AS nav_value
        FROM mutual_fund_nav
        WHERE code = %s
        ORDER BY nav
        """
        df = pd.read_sql(query, conn, params=[code])
        df['nav_date'] = pd.to_datetime(df['nav_date'])
        return df.set_index('nav_date')

# Portfolio Units from portfolio_data
@st.cache_data
def get_units_from_portfolio(code):
    with connect_to_db() as conn:
        query = """
        SELECT transaction_date, transaction_type, units, amount
        FROM portfolio_data
        WHERE code = %s
        ORDER BY transaction_date
        """
        df = pd.read_sql(query, conn, params=[code])
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
        return df

nav_data = get_nav_history(selected_code)
portfolio_df = get_units_from_portfolio(selected_code)

# Filter by FY or Month
st.sidebar.subheader("🔎 Filter Data")
fy_options = portfolio_df['transaction_date'].dt.to_period('Y').astype(str).unique()
fiscal_filter = st.sidebar.multiselect("Select Financial Year(s)", options=fy_options, default=list(fy_options))
portfolio_df = portfolio_df[portfolio_df['transaction_date'].dt.to_period('Y').astype(str).isin(fiscal_filter)]

# Separate investments and redemptions
investments = portfolio_df[portfolio_df['transaction_type'].isin(['invest', 'switch_in'])].copy()
redemptions = portfolio_df[portfolio_df['transaction_type'].isin(['redeem', 'switch_out'])].copy()

# Preprocess investment lots
lots = []
for _, row in investments.iterrows():
    dt = row['transaction_date']
    amount = row['amount']
    units = row['units']
    nav_value = nav_data.loc[nav_data.index.get_indexer([dt], method='nearest')[0], 'nav_value']
    ltcg_date = dt + pd.DateOffset(months=12)
    today = datetime.today()
    is_ltcg = today >= ltcg_date
    redeem_date = ltcg_date if is_ltcg else dt
    redemption_nav = nav_data.loc[nav_data.index.get_indexer([redeem_date], method='nearest')[0], 'nav_value']
    current_value = units * redemption_nav
    gain = current_value - amount
    lots.append({
        'Date': dt.date(),
        'Amount Invested': amount,
        'Units': units,
        'NAV': nav_value,
        'LTCG Eligibility': ltcg_date.date(),
        'Gain Type': 'LTCG' if is_ltcg else 'STCG',
        'Current Value': current_value,
        'Capital Gain': gain,
        'Redeemable': is_ltcg
    })

lots_df = pd.DataFrame(lots)

# Adjust for redemptions (FIFO)
for _, row in redemptions.iterrows():
    remaining_units = row['units']
    for i in range(len(lots_df)):
        if lots_df.loc[i, 'Units'] > 0:
            offset = min(lots_df.loc[i, 'Units'], remaining_units)
            lots_df.loc[i, 'Units'] -= offset
            lots_df.loc[i, 'Amount Invested'] *= (lots_df.loc[i, 'Units'] / (lots_df.loc[i, 'Units'] + offset))
            remaining_units -= offset
        if remaining_units <= 0:
            break
lots_df = lots_df[lots_df['Units'] > 0]

# LTCG scheduler (limit to ₹1.25L gains per FY)
lots_df['FY'] = lots_df['LTCG Eligibility'].apply(lambda d: f"{d.year}-{d.year+1}")
lots_df['Capital Gain'] = lots_df['Capital Gain'].round(2)

scheduler = []
fy_groups = lots_df[lots_df['Gain Type'] == 'LTCG'].groupby('FY')
for fy, group in fy_groups:
    total_gain = 0
    selected_lots = []
    for _, row in group.sort_values('Capital Gain').iterrows():
        if total_gain + row['Capital Gain'] <= 125000:
            total_gain += row['Capital Gain']
            selected_lots.append(row)
        else:
            break
    for lot in selected_lots:
        scheduler.append(lot)

scheduler_df = pd.DataFrame(scheduler)

# Display Summary
st.subheader("📌 LTCG Scheduler - Tax Efficient Withdrawals")
if not scheduler_df.empty:
    st.dataframe(scheduler_df[['Date', 'Amount Invested', 'Units', 'Current Value', 'Capital Gain', 'FY']], use_container_width=True)
    st.download_button("⬇️ Download Withdrawal Plan (CSV)", data=scheduler_df.to_csv(index=False), file_name="ltcg_withdrawal_schedule.csv", mime="text/csv")
    st.success("✅ You can withdraw from the above lots without exceeding ₹1.25L LTCG limit per FY")
else:
    st.warning("⚠️ No LTCG-eligible units available within the ₹1.25L exemption limit")

st.subheader("📈 Remaining Units Over Time")
lots_df['YearMonth'] = pd.to_datetime(lots_df['Date']).dt.to_period('M').astype(str)
units_by_month = lots_df.groupby('YearMonth')['Units'].sum().reset_index()
fig = px.line(units_by_month, x='YearMonth', y='Units', title='Remaining Units Over Time')
st.plotly_chart(fig, use_container_width=True)

st.subheader("📦 Actual Arbitrage Holdings from Portfolio")
st.dataframe(lots_df[['Date', 'Amount Invested', 'Units', 'NAV', 'Current Value', 'Capital Gain', 'Gain Type', 'LTCG Eligibility']], use_container_width=True)

st.subheader("📤 Redemption History")
if not redemptions.empty:
    redemptions_display = redemptions[['transaction_date', 'units', 'amount']].copy()
    redemptions_display.columns = ['Date', 'Units Redeemed', 'Amount Received']
    st.dataframe(redemptions_display, use_container_width=True)
else:
    st.info("No redemptions found in portfolio data.")

st.caption("Data sourced from portfolio_data and mutual_fund_nav tables. Redemptions reduce available units. Scheduler avoids LTCG tax by staying under ₹1.25L/year.")
st.sidebar.markdown("""
    <style>
        .sidebar .sidebar-content {
            background-color: #f0f2f5;
        }
    </style>
""", unsafe_allow_html=True)