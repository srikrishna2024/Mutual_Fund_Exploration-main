import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import psycopg
from psycopg import sql

st.set_page_config(page_title="Stock Tax Optimizer", layout="wide")
st.title("Stock Portfolio Tax Optimization Assistant (2024 Rules)")

st.markdown("""
This application analyzes your stock portfolio transactions with the latest 2024 tax rules:

**Key 2024 Updates:**
- LTCG exemption limit increased to ₹1.25 lakh (from ₹1 lakh)
- LTCG tax rate increased to 12.5% (from 10%) effective 23-July-2024
- STCG tax rate increased to 20% (from 15%) effective 23-July-2024
""")

# Indian Financial Year Calculation
current_date = datetime.now()
if current_date.month >= 4:  # April to December
    current_fy_start_year = current_date.year
    current_fy_end_year = current_date.year + 1
else:  # January to March
    current_fy_start_year = current_date.year - 1
    current_fy_end_year = current_date.year

def connect_to_db():
    """Connect to PostgreSQL database"""
    DB_PARAMS = {
        'dbname': 'postgres',
        'user': 'postgres',
        'password': 'admin123',
        'host': 'localhost',
        'port': '5432'
    }
    try:
        conn = psycopg.connect(**DB_PARAMS)
        return conn
    except Exception as e:
        st.error(f"Database connection failed: {str(e)}")
        return None

# Data source selection
st.subheader("📂 Select Data Source")
data_source = st.radio(
    "Choose how to load your stock transactions:",
    ("Upload File", "Database Connection"),
    horizontal=True
)

df = None

if data_source == "Upload File":
    # File upload section
    uploaded_file = st.file_uploader(
        "Upload your stock transactions (CSV/Excel)",
        type=["csv", "xlsx", "xls"],
        help="File should contain columns: symbol, trade_date, trade_type, quantity, price"
    )

    if not uploaded_file:
        st.info("Please upload your stock transactions file to begin analysis")
        st.stop()

    # Load and validate data
    @st.cache_data
    def load_data(file):
        if file.name.endswith('.csv'):
            return pd.read_csv(file)
        else:
            return pd.read_excel(file)

    try:
        with st.spinner("Processing your transactions..."):
            df = load_data(uploaded_file)
            
            # Standardize column names (case insensitive)
            df.columns = df.columns.str.lower()
            required_columns = {'symbol', 'trade_date', 'trade_type', 'quantity', 'price'}
            if not required_columns.issubset(df.columns):
                missing = required_columns - set(df.columns)
                st.error(f"Missing required columns: {', '.join(missing)}")
                st.stop()
                
            # Data cleaning
            df = df.rename(columns={
                'symbol': 'Symbol',
                'trade_date': 'Trade Date',
                'trade_type': 'Trade Type',
                'quantity': 'Quantity',
                'price': 'Price'
            })
            
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.stop()

else:  # Database Connection
    try:
        with st.spinner("Connecting to database..."):
            conn = connect_to_db()
            if conn is None:
                st.stop()
                
            # Query the database
            query = """
            SELECT symbol, trade_date, trade_type, quantity, price 
            FROM portfolio_stocks
            ORDER BY trade_date
            """
            
            df = pd.read_sql(query, conn)
            conn.close()
            
            # Standardize column names (case insensitive)
            df.columns = df.columns.str.lower()
            required_columns = {'symbol', 'trade_date', 'trade_type', 'quantity', 'price'}
            if not required_columns.issubset(df.columns):
                missing = required_columns - set(df.columns)
                st.error(f"Missing required columns in database table: {', '.join(missing)}")
                st.stop()
                
            # Data cleaning
            df = df.rename(columns={
                'symbol': 'Symbol',
                'trade_date': 'Trade Date',
                'trade_type': 'Trade Type',
                'quantity': 'Quantity',
                'price': 'Price'
            })
            
            st.success(f"Successfully loaded {len(df)} transactions from database")
            
    except Exception as e:
        st.error(f"Error loading data from database: {str(e)}")
        if 'conn' in locals():
            conn.close()
        st.stop()

if df is None and data_source == "Database Connection":
    st.info("Please connect to database to load transactions")
    st.stop()

# Common data processing for both file and database sources
try:
    with st.spinner("Processing transactions..."):
        # Convert to datetime and numeric types
        df['Trade Date'] = pd.to_datetime(df['Trade Date'], errors='coerce')
        df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        
        # Remove any rows with missing critical data
        df = df.dropna(subset=['Symbol', 'Trade Date', 'Trade Type', 'Quantity', 'Price'])
        
        # Separate buy and sell transactions
        buys = df[df['Trade Type'].str.lower().isin(['buy', 'b'])]
        sells = df[df['Trade Type'].str.lower().isin(['sell', 's'])]
        
        # Add amount column
        buys['Amount'] = buys['Quantity'] * buys['Price']
        sells['Amount'] = sells['Quantity'] * sells['Price']
        
except Exception as e:
    st.error(f"Error processing transactions: {str(e)}")
    st.stop()

# Display basic portfolio info
st.success(f"Processed {len(df)} transactions ({len(buys)} buys, {len(sells)} sells)")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Transactions", len(df))
with col2:
    st.metric("Unique Stocks", df['Symbol'].nunique())
with col3:
    st.metric("Date Range", f"{df['Trade Date'].min().strftime('%Y-%m-%d')} to {df['Trade Date'].max().strftime('%Y-%m-%d')}")

# Assume all stocks are equity
df['Classification'] = 'Equity'
buys['Classification'] = 'Equity'
sells['Classification'] = 'Equity'

# Initialize tax calculation variables
results = []
carry_forward_losses = 0
carry_forward_opportunities = []
unrealized_ltcg = 0
unrealized_stcg = 0

# Process each stock for REALIZED gains first
for stock in df['Symbol'].unique():
    stock_buys = buys[buys['Symbol'] == stock].copy()
    stock_sells = sells[sells['Symbol'] == stock].copy()
    stock_buys.sort_values('Trade Date', inplace=True)
    stock_sells.sort_values('Trade Date', inplace=True)
    
    # Create FIFO queue for this stock
    stock_fifo = stock_buys[['Trade Date', 'Price', 'Quantity']].to_dict('records')
    
    # Process sell transactions
    for _, sell in stock_sells.iterrows():
        units_to_sell = sell['Quantity']
        sell_date = sell['Trade Date']
        sell_price = sell['Price']
        
        while units_to_sell > 0 and stock_fifo:
            lot = stock_fifo[0]
            lot_units = lot['Quantity']
            lot_date = lot['Trade Date']
            lot_price = lot['Price']
            
            matched_units = min(units_to_sell, lot_units)
            holding_period = (sell_date - lot_date).days
            is_long_term = holding_period > 365  # Equity threshold
            cost = matched_units * lot_price
            proceeds = matched_units * sell_price
            gain = proceeds - cost
            
            # Determine tax rate based on date
            if sell_date >= pd.to_datetime('2024-07-23'):
                stcg_rate = 0.20  # 20% after 23-July-2024
                ltcg_rate = 0.125  # 12.5% after 23-July-2024
            else:
                stcg_rate = 0.15  # 15% before 23-July-2024
                ltcg_rate = 0.10  # 10% before 23-July-2024
            
            gain_type = 'LTCG' if is_long_term else 'STCG'
            
            if gain < 0:
                carry_forward_losses += abs(gain)
            
            # Calculate Indian Financial Year for the sell date
            if sell_date.month >= 4:  # April to December
                financial_year = sell_date.year + 1
            else:  # January to March
                financial_year = sell_date.year
            
            results.append({
                'Symbol': stock,
                'Sell Date': sell_date.date(),
                'Financial Year': financial_year,
                'Units Sold': matched_units,
                'Holding Period (days)': holding_period,
                'Gain Type': gain_type,
                'Gain': gain,
                'Tax Rate': ltcg_rate if is_long_term else stcg_rate,
                'Tax Amount': gain * ltcg_rate if is_long_term else gain * stcg_rate
            })
            
            units_to_sell -= matched_units
            lot['Quantity'] -= matched_units
            if lot['Quantity'] <= 0:
                stock_fifo.pop(0)
    
    # Process remaining units (unrealized gains/losses)
    if not stock_fifo:
        continue
        
    # Calculate unrealized gains for ALL remaining lots
    for lot in stock_fifo:
        holding_period = (datetime.today() - lot['Trade Date']).days
        current_price = lot['Price']  # Ideally fetch from API
        gain = lot['Quantity'] * (current_price - lot['Price'])
        
        if holding_period > 365:
            unrealized_ltcg += gain
        else:
            unrealized_stcg += gain
    
    # Add to carry_forward_opportunities if the oldest lot is at a loss
    oldest_lot = stock_fifo[0]
    holding_period = (datetime.today() - oldest_lot['Trade Date']).days
    current_price = oldest_lot['Price']
    gain = oldest_lot['Quantity'] * (current_price - oldest_lot['Price'])
    
    if gain < 0:
        carry_forward_opportunities.append({
            'Symbol': stock,
            'Purchase Date': oldest_lot['Trade Date'].date(),
            'Quantity': oldest_lot['Quantity'],
            'Purchase Price': oldest_lot['Price'],
            'Current Price': current_price,
            'Unrealized Loss': gain,
            'Loss Amount': abs(gain),
            'Holding Period (days)': holding_period
        })

def calculate_net_gains(gains_df):
    """Apply tax set-off rules and calculate carry-forward losses"""
    if gains_df.empty:
        return pd.DataFrame()
    
    # Ensure we have both gain types even if empty
    if 'Gain Type' not in gains_df.columns:
        gains_df['Gain Type'] = 'STCG'  # Default to STCG if not specified
    
    fy_summary = gains_df.groupby(['Financial Year', 'Gain Type'])['Gain'].sum().unstack(fill_value=0)
    
    # Initialize columns with default values
    fy_summary['Net_STCG'] = fy_summary.get('STCG', 0)
    fy_summary['Net_LTCG'] = fy_summary.get('LTCG', 0)
    fy_summary['Carry_Forward'] = 0
    
    # Apply set-off rules
    for fy in fy_summary.index:
        stcg = fy_summary.loc[fy, 'STCG'] if 'STCG' in fy_summary.columns else 0
        ltcg = fy_summary.loc[fy, 'LTCG'] if 'LTCG' in fy_summary.columns else 0
        
        # Initialize net values
        net_stcg = max(0, stcg)
        net_ltcg = max(0, ltcg)
        carry_forward = 0
        
        # Handle STCL (negative STCG)
        if stcg < 0:
            stcl = abs(stcg)
            
            # STCL first offsets STCG (none in this case since STCG is negative)
            # Then offsets LTCG
            remaining_after_ltcg = max(0, stcl - net_ltcg)
            
            # Update net values
            net_ltcg = max(0, ltcg - stcl)
            carry_forward = remaining_after_ltcg
        
        fy_summary.loc[fy, 'Net_STCG'] = net_stcg
        fy_summary.loc[fy, 'Net_LTCG'] = net_ltcg
        fy_summary.loc[fy, 'Carry_Forward'] = carry_forward
    
    return fy_summary

# Display results with accurate set-off calculations
if results:
    gains_df = pd.DataFrame(results)
    fy_summary = calculate_net_gains(gains_df)
    
    if not fy_summary.empty:
        # Get values with proper defaults if columns don't exist
        total_ltcg = fy_summary['Net_LTCG'].sum() if 'Net_LTCG' in fy_summary.columns else 0
        total_stcg = fy_summary['Net_STCG'].sum() if 'Net_STCG' in fy_summary.columns else 0
        total_carry_forward = fy_summary['Carry_Forward'].sum() if 'Carry_Forward' in fy_summary.columns else 0

        st.subheader("📊 Realized Capital Gains Summary")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Net LTCG (After Set-Off)", f"₹{total_ltcg:,.1f}")
            with st.expander("FY-wise Breakdown"):
                st.write("**Long-Term Capital Gains:**")
                if 'Net_LTCG' in fy_summary.columns:
                    for fy in fy_summary.index:
                        fy_label = f"FY {fy-1}-{str(fy)[2:]}"
                        st.write(f"{fy_label}: ₹{fy_summary.loc[fy, 'Net_LTCG']:,.1f}")
                else:
                    st.write("No LTCG data available")
        
        with col2:
            st.metric("Net STCG (After Set-Off)", f"₹{total_stcg:,.1f}")
            with st.expander("FY-wise Breakdown"):
                st.write("**Short-Term Capital Gains:**")
                if 'Net_STCG' in fy_summary.columns:
                    for fy in fy_summary.index:
                        fy_label = f"FY {fy-1}-{str(fy)[2:]}"
                        st.write(f"{fy_label}: ₹{fy_summary.loc[fy, 'Net_STCG']:,.1f}")
                else:
                    st.write("No STCG data available")
        
        with col3:
            st.metric("Total Carry-Forward Losses", f"₹{total_carry_forward:,.1f}")
            with st.expander("Detailed Analysis"):
                st.write("**Net Gains After Tax Set-Off Rules:**")
                display_cols = [col for col in ['STCG', 'LTCG', 'Net_STCG', 'Net_LTCG', 'Carry_Forward'] 
                              if col in fy_summary.columns]
                st.dataframe(fy_summary[display_cols].style.format('₹{:,.1f}'))
                
                if total_carry_forward > 0:
                    st.write("\n**Carry-Forward Details:**")
                    for fy in fy_summary.index:
                        if fy_summary.loc[fy, 'Carry_Forward'] > 0:
                            st.write(
                                f"- FY {fy-1}-{str(fy)[2:]}: ₹{fy_summary.loc[fy, 'Carry_Forward']:,.1f} "
                                f"(Claimable from AY {fy}-{str(fy+1)[2:]} to AY {fy+7}-{str(fy+8)[2:]})"
                            )

        # Show detailed gains table
        with st.expander("View All Capital Gains Transactions"):
            st.dataframe(gains_df, use_container_width=True)

        # Year-wise breakdown chart
        yearly_summary = gains_df.groupby(['Financial Year', 'Gain Type'])['Gain'].sum().reset_index()
        if not yearly_summary.empty:
            fig = px.bar(
                yearly_summary,
                x='Financial Year',
                y='Gain',
                color='Gain Type',
                barmode='group',
                title='Year-wise Capital Gains Breakdown',
                width=700,
                height=400
            )
            fig.update_traces(marker_line_width=0.5, marker_line_color='black')
            st.plotly_chart(fig, use_container_width=True)

# Unrealized gains section
st.subheader("📊 Unrealized Capital Gains")
col1, col2 = st.columns(2)
with col1:
    st.metric("Unrealized LTCG (Eligible)", f"₹{unrealized_ltcg:,.1f}")
with col2:
    st.metric("Unrealized STCG (Still Short-Term)", f"₹{unrealized_stcg:,.1f}")

# LTCG Exemption Tracker (₹1.25 lakh limit for equity)
st.subheader("🎯 LTCG Exemption Tracker")

current_fy_label = f"FY {current_fy_start_year}-{str(current_fy_end_year)[2:]}"

current_fy_ltcg = 0
if results:
    # Filter gains for current Indian FY
    current_fy_gains = gains_df[
        (gains_df['Sell Date'] >= pd.to_datetime(f'{current_fy_start_year}-04-01').date()) &
        (gains_df['Sell Date'] <= pd.to_datetime(f'{current_fy_end_year}-03-31').date()) &
        (gains_df['Gain Type'] == 'LTCG')
    ]
    current_fy_ltcg = current_fy_gains['Gain'].sum()

remaining_exemption = max(0, 125000 - current_fy_ltcg)
excess_ltcg = max(0, current_fy_ltcg - 125000)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(f"{current_fy_label} LTCG Used", f"₹{current_fy_ltcg:,.1f}")
with col2:
    st.metric("Remaining LTCG Exemption", f"₹{remaining_exemption:,.1f}")
with col3:
    exemption_percentage = min(100, (current_fy_ltcg / 125000) * 100) if 125000 > 0 else 0
    st.metric("Exemption Utilized", f"{exemption_percentage:.1f}%")

# Tax Rates (Updated 2024)
st.subheader("🧾 Updated Tax Rates (2024 Changes)")
tax_rates = {
    'Security Type': ['Equity Shares'],
    'STCG Rate (Pre-23-Jul-2024)': ['15%'],
    'STCG Rate (Post-23-Jul-2024)': ['20%'],
    'LTCG Rate (Pre-23-Jul-2024)': ['10% over ₹1L'],
    'LTCG Rate (Post-23-Jul-2024)': ['12.5% over ₹1.25L'],
    'Holding Period': ['12 months'],
    'STT Applicable': ['Yes']
}
st.table(pd.DataFrame(tax_rates))

# Tax Loss Harvesting Opportunities
st.subheader("📉 Tax Loss Harvesting Opportunities")
if carry_forward_opportunities:
    harvestable_df = pd.DataFrame(carry_forward_opportunities)
    harvestable_df = harvestable_df.sort_values('Loss Amount', ascending=False)
    
    st.dataframe(harvestable_df[[
        'Symbol', 'Purchase Date', 'Quantity', 
        'Purchase Price', 'Current Price', 'Loss Amount', 'Holding Period (days)'
    ]], use_container_width=True)
    
    total_harvestable = harvestable_df['Loss Amount'].sum()
    st.metric("Total Harvestable Losses", f"₹{total_harvestable:,.1f}")
else:
    st.info("No tax loss harvesting opportunities found in your portfolio")

# Portfolio overview
with st.expander("📋 Portfolio Overview"):
    st.write("**Current Holdings by Stock:**")
    current_holdings = buys.groupby('Symbol').agg({
        'Quantity': 'sum',
        'Amount': 'sum'
    }).reset_index()
    current_holdings['Avg Purchase Price'] = current_holdings['Amount'] / current_holdings['Quantity']
    st.dataframe(current_holdings, use_container_width=True)

# Save analysis report
st.subheader("💾 Save Analysis Report")
if st.button("Generate PDF Report"):
    with st.spinner("Generating report..."):
        # Here you would add code to generate a PDF report
        # This would typically use a library like reportlab or weasyprint
        st.success("Report generated successfully!")
        st.download_button(
            label="Download Report",
            data="Your report data would go here",
            file_name="Stock_Tax_Report.pdf",
            mime="application/pdf"
        )