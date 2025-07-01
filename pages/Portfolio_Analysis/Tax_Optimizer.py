import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import psycopg

st.set_page_config(page_title="MF Tax Optimizer", layout="wide")
st.title("Mutual Fund Tax Optimization Assistant")

st.markdown("""
This application analyzes your mutual fund portfolio data directly from the database to provide tax optimization insights.

**Portfolio Data Analysis:**
- Reads transaction data from `portfolio_data` table
- Calculates realized and unrealized capital gains
- Identifies tax loss harvesting opportunities
- Shows upcoming long-term eligibility dates
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
    DB_PARAMS = {
        'dbname': 'postgres',
        'user': 'postgres',
        'password': 'admin123',
        'host': 'localhost',
        'port': '5432'
    }
    return psycopg.connect(**DB_PARAMS)

def get_portfolio_data():
    """Fetch portfolio data from database"""
    with connect_to_db() as conn:
        query = """
            SELECT 
                id,
                date,
                scheme_name,
                code,
                transaction_type,
                value as nav,
                units,
                amount,
                created_at
            FROM portfolio_data
            ORDER BY scheme_name, date
        """
        return pd.read_sql(query, conn)

def get_latest_nav_from_db():
    """Get latest NAV values for each scheme"""
    with connect_to_db() as conn:
        query = """
            SELECT code, value AS nav_value
            FROM mutual_fund_nav
            WHERE (code, nav) IN (
                SELECT code, MAX(nav) AS nav_date
                FROM mutual_fund_nav
                GROUP BY code
            )
        """
        return pd.read_sql(query, conn)

# Load data from database
try:
    with st.spinner("Loading portfolio data from database..."):
        df = get_portfolio_data()
    
    if df.empty:
        st.error("No portfolio data found in the database.")
        st.stop()
    
    st.success(f"Loaded {len(df)} transactions from portfolio_data table")
    
    # Display basic portfolio info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Transactions", len(df))
    with col2:
        st.metric("Unique Schemes", df['scheme_name'].nunique())
    with col3:
        st.metric("Date Range", f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")

except Exception as e:
    st.error(f"Error connecting to database: {str(e)}")
    st.stop()

# Data preprocessing
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.rename(columns={
    'scheme_name': 'Fund Name',
    'transaction_type': 'Transaction Type',
    'nav': 'NAV',
    'units': 'Units',
    'amount': 'Amount',
    'date': 'Date',
    'code': 'Code'
})

# Get latest NAV data
try:
    latest_nav_df = get_latest_nav_from_db()
    latest_nav_map = dict(zip(latest_nav_df['code'], latest_nav_df['nav_value']))
except Exception as e:
    st.warning(f"Could not fetch latest NAV data: {str(e)}")
    latest_nav_map = {}

df = df.sort_values(by=['Fund Name', 'Date'])

# Separate buy and sell transactions
buys = df[df['Transaction Type'].str.lower().isin(['invest', 'switch_in'])].copy()
sells = df[df['Transaction Type'].str.lower().isin(['redeem', 'switch_out'])].copy()

# Initialize all tax calculation variables
results = []
carry_forward_losses = 0
carry_forward_opportunities = []
unrealized_ltcg = 0
unrealized_stcg = 0

st.subheader("📈 Tax Optimization Analysis")

# Process each fund
for fund in df['Fund Name'].unique():
    fund_buys = buys[buys['Fund Name'] == fund].copy()
    fund_sells = sells[sells['Fund Name'] == fund].copy()
    fund_buys.sort_values('Date', inplace=True)
    fund_sells.sort_values('Date', inplace=True)

    # Create FIFO queue for this fund
    available_cols = [col for col in ['Date', 'NAV', 'Units', 'Code'] if col in fund_buys.columns]
    fund_fifo = fund_buys[available_cols].to_dict('records')

    # Process sell transactions
    for _, sell in fund_sells.iterrows():
        units_to_sell = sell['Units']
        sell_date = sell['Date']
        sell_nav = sell['NAV']

        while units_to_sell > 0 and fund_fifo:
            lot = fund_fifo[0]
            lot_units = lot['Units']
            lot_date = lot['Date']
            lot_nav = lot['NAV']

            matched_units = min(units_to_sell, lot_units)
            holding_period = (sell_date - lot_date).days
            is_long_term = holding_period > 365
            cost = matched_units * lot_nav
            proceeds = matched_units * sell_nav
            gain = proceeds - cost

            gain_type = 'LTCG' if is_long_term else 'STCG'

            if gain < 0:
                carry_forward_losses += abs(gain)

            # Calculate Indian Financial Year for the sell date
            if sell_date.month >= 4:  # April to December
                financial_year = sell_date.year + 1
            else:  # January to March
                financial_year = sell_date.year

            results.append({
                'Fund Name': fund,
                'Sell Date': sell_date.date(),
                'Financial Year': financial_year,
                'Units Sold': matched_units,
                'Holding Period (days)': holding_period,
                'Gain Type': gain_type,
                'Gain': gain,
            })

            units_to_sell -= matched_units
            lot['Units'] -= matched_units
            if lot['Units'] <= 0:
                fund_fifo.pop(0)

    # Process remaining units (unrealized gains/losses)
    if not fund_fifo:
        continue
        
    for lot in fund_fifo:
        holding_period = (datetime.today() - lot['Date']).days
        code = lot.get('Code', None)
        current_nav = latest_nav_map.get(code, lot['NAV'])
        gain = lot['Units'] * (current_nav - lot['NAV'])
        
        if gain < 0:
            carry_forward_opportunities.append({
                'Fund Name': fund,
                'Purchase Date': lot['Date'].date(),
                'Units': lot['Units'],
                'NAV at Purchase': lot['NAV'],
                'Current NAV': current_nav,
                'Unrealized Loss': gain,
                'Loss Amount': abs(gain)
            })
        
        if holding_period > 365:
            unrealized_ltcg += gain
        else:
            unrealized_stcg += gain

# Corrected set-off calculations
def calculate_net_gains(gains_df):
    """Apply tax set-off rules and calculate carry-forward losses"""
    fy_summary = gains_df.groupby(['Financial Year', 'Gain Type'])['Gain'].sum().unstack(fill_value=0)
    
    # Initialize columns
    fy_summary['Net_STCG'] = fy_summary.get('STCG', 0)
    fy_summary['Net_LTCG'] = fy_summary.get('LTCG', 0)
    fy_summary['Carry_Forward'] = 0
    
    # Apply corrected set-off rules (STCL first offsets STCG, then LTCG)
    for fy in fy_summary.index:
        stcg = fy_summary.loc[fy, 'STCG']
        ltcg = fy_summary.loc[fy, 'LTCG']
        
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
    total_ltcg = fy_summary['Net_LTCG'].sum()
    total_stcg = fy_summary['Net_STCG'].sum()
    total_carry_forward = fy_summary['Carry_Forward'].sum()

    st.subheader("📊 Realized Capital Gains Summary")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Net LTCG (After Set-Off)", f"₹{total_ltcg:,.1f}")
        with st.expander("FY-wise Breakdown"):
            st.write("**Long-Term Capital Gains:**")
            for fy in fy_summary.index:
                fy_label = f"FY {fy-1}-{str(fy)[2:]}"
                st.write(f"{fy_label}: ₹{fy_summary.loc[fy, 'Net_LTCG']:,.1f}")
    
    with col2:
        st.metric("Net STCG (After Set-Off)", f"₹{total_stcg:,.1f}")
        with st.expander("FY-wise Breakdown"):
            st.write("**Short-Term Capital Gains:**")
            for fy in fy_summary.index:
                fy_label = f"FY {fy-1}-{str(fy)[2:]}"
                st.write(f"{fy_label}: ₹{fy_summary.loc[fy, 'Net_STCG']:,.1f}")
    
    with col3:
        st.metric("Total Carry-Forward Losses", f"₹{total_carry_forward:,.1f}")
        with st.expander("Detailed Analysis"):
            st.write("**Net Gains After Tax Set-Off Rules:**")
            st.dataframe(fy_summary.style.format({
                'STCG': '₹{:,.1f}',
                'LTCG': '₹{:,.1f}',
                'Net_STCG': '₹{:,.1f}',
                'Net_LTCG': '₹{:,.1f}',
                'Carry_Forward': '₹{:,.1f}'
            }))
            
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

# [Rest of your original code remains unchanged...]
# Continue with all remaining sections exactly as in your original script
# Including Unrealized Gains, Upcoming Long-Term Eligibility, Tax Loss Harvesting, etc.

# Unrealized gains section
st.subheader("📊 Unrealized Capital Gains")
col1, col2 = st.columns(2)
with col1:
    st.metric("Unrealized LTCG (Eligible)", f"₹{unrealized_ltcg:,.1f}")
with col2:
    st.metric("Unrealized STCG (Still Short-Term)", f"₹{unrealized_stcg:,.1f}")

# [Continue with all other sections exactly as in your original script]
# Upcoming long-term eligibility (FIFO-aware)
st.subheader("🕒 Upcoming Long-Term Eligibility")
today = datetime.today()

# Calculate FIFO-aware available units for each fund
fifo_available_units = []

for fund in df['Fund Name'].unique():
    fund_buys = buys[buys['Fund Name'] == fund].copy()
    fund_sells = sells[sells['Fund Name'] == fund].copy()
    fund_buys.sort_values('Date', inplace=True)
    fund_sells.sort_values('Date', inplace=True)

    # Create FIFO queue for this fund (similar to main calculation)
    available_cols = [col for col in ['Date', 'NAV', 'Units', 'Code'] if col in fund_buys.columns]
    fund_fifo = fund_buys[available_cols].to_dict('records')

    # Process all sell transactions to get remaining FIFO queue
    total_units_sold = fund_sells['Units'].sum() if not fund_sells.empty else 0
    remaining_units_to_deduct = total_units_sold

    # Remove sold units from FIFO queue
    fifo_queue_copy = fund_fifo.copy()
    for lot in fifo_queue_copy:
        if remaining_units_to_deduct <= 0:
            break
        
        if lot['Units'] <= remaining_units_to_deduct:
            remaining_units_to_deduct -= lot['Units']
            fund_fifo.remove(lot)
        else:
            lot['Units'] -= remaining_units_to_deduct
            remaining_units_to_deduct = 0

    # Now check remaining FIFO units for LTCG eligibility
    for lot in fund_fifo:
        holding_period = (today - lot['Date']).days
        days_to_ltcg = 365 - holding_period
        
        # Check if nearing LTCG (between 335-365 days, i.e., within 30 days of becoming LTCG)
        if 0 < days_to_ltcg <= 30:  # Still STCG but becoming LTCG soon
            fifo_available_units.append({
                'Fund Name': fund,
                'Purchase Date': lot['Date'].date(),
                'Available Units': lot['Units'],
                'Purchase NAV': lot['NAV'],
                'Holding Period (days)': holding_period,
                'Days to LTCG': days_to_ltcg,
                'Current Status': 'Next in FIFO Queue'
            })

if fifo_available_units:
    st.warning("⚠️ **FIFO-Aware Alert:** These units are next in line for sale and nearing 1-year LTCG eligibility:")
    st.info("💡 These are the actual units that would be sold if you redeem from these funds (following FIFO method)")
    
    near_lt_df = pd.DataFrame(fifo_available_units)
    near_lt_df = near_lt_df.sort_values('Days to LTCG')
    st.dataframe(near_lt_df, use_container_width=True)
    
    # Summary metrics
    col1, col2 = st.columns(2)
    with col1:
        total_units_near_ltcg = near_lt_df['Available Units'].sum()
        st.metric("Total Units Nearing LTCG", f"{total_units_near_ltcg:,.1f}")
    with col2:
        unique_funds = near_lt_df['Fund Name'].nunique()
        st.metric("Funds Affected", unique_funds)
    
    # Specific recommendations
    st.markdown(f"""
    **⚠️ Important FIFO Considerations:**
    - These units are **next in line** to be sold if you redeem from these funds
    - **Wait {near_lt_df['Days to LTCG'].min()} more days** for LTCG treatment to save on taxes
    - **STCG Tax**: 20% vs **LTCG Tax**: 12.5% (+ ₹1.25L exemption)
    - If you must sell, consider **different funds** where LTCG-eligible units are available
    """)
    
    # Fund-wise FIFO status
    with st.expander("📋 Fund-wise FIFO Status"):
        st.markdown("**Understanding FIFO Impact:**")
        for fund in near_lt_df['Fund Name'].unique():
            fund_data = near_lt_df[near_lt_df['Fund Name'] == fund]
            earliest_ltcg_date = fund_data['Days to LTCG'].min()
            st.write(f"- **{fund}**: Wait {earliest_ltcg_date} days before selling to get LTCG benefit")

else:
    # Check if there are any available units at all
    any_available_units = False
    for fund in df['Fund Name'].unique():
        fund_buys_total = buys[buys['Fund Name'] == fund]['Units'].sum()
        fund_sells_total = sells[sells['Fund Name'] == fund]['Units'].sum() if not sells[sells['Fund Name'] == fund].empty else 0
        
        if fund_buys_total > fund_sells_total:
            any_available_units = True
            break
    
    if any_available_units:
        st.success("✅ No units in FIFO queue are nearing LTCG eligibility in the next 30 days.")
        st.info("💡 Your next-in-line units are either already LTCG eligible or still have more than 30 days to go.")
    else:
        st.info("ℹ️ All purchased units have been sold. No units available for LTCG analysis.")

# Tax loss harvesting opportunities
if carry_forward_opportunities:
    st.subheader("📉 Tax Loss Harvesting Opportunities")
    st.info("💡 These units are currently at a loss. Consider redeeming and re-investing to book losses for tax benefit:")
    opportunities_df = pd.DataFrame(carry_forward_opportunities)
    opportunities_df = opportunities_df.sort_values('Loss Amount', ascending=False)
    st.dataframe(opportunities_df, use_container_width=True)
    
    total_loss_opportunity = opportunities_df['Loss Amount'].sum()
    st.metric("Total Loss Harvesting Potential", f"₹{total_loss_opportunity:,.1f}")
else:
    st.success("✅ No current tax loss harvesting opportunities found.")

# LTCG Exemption Tracker (₹1.25 lakh limit)
st.subheader("🎯 LTCG Exemption Tracker")

current_fy_label = f"FY {current_fy_start_year}-{str(current_fy_end_year)[2:]}"

current_fy_ltcg = 0
if results:
    # Filter gains for current Indian FY
    current_fy_gains = gains_df[
        (gains_df['Sell Date'] >= pd.to_datetime(f'{current_fy_start_year}-04-01').date()) &
        (gains_df['Sell Date'] <= pd.to_datetime(f'{current_fy_end_year}-03-31').date())
    ]
    current_fy_ltcg = current_fy_gains[current_fy_gains['Gain Type'] == 'LTCG']['Gain'].sum()

remaining_exemption = max(0, 125000 - current_fy_ltcg)
excess_ltcg = max(0, current_fy_ltcg - 125000)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(f"{current_fy_label} LTCG Used", f"₹{current_fy_ltcg:,.1f}")
with col2:
    st.metric("Remaining LTCG Exemption", f"₹{remaining_exemption:,.1f}")
with col3:
    exemption_percentage = min(100, (current_fy_ltcg / 125000) * 100)
    st.metric("Exemption Utilized", f"{exemption_percentage:.1f}%")

# Smart Recommendations based on LTCG status
if current_fy_ltcg == 0:
    st.success("✅ **No LTCG realized this FY** - You have full ₹1.25L exemption available!")
    st.info("💡 **Recommendation:** Consider booking some LTCG profits up to ₹1.25L to rebalance portfolio tax-free.")
    
elif remaining_exemption > 0:
    st.info(f"💡 **Good News:** You can still realize ₹{remaining_exemption:,.1f} in LTCG without paying tax this financial year!")
    
    # Find suitable units for tax-free booking
    if not buys.empty:
        profitable_units = []
        for _, row in buys.iterrows():
            if (datetime.now() - row['Date']).days > 365:  # LTCG eligible
                code = row['Code']
                current_nav = latest_nav_map.get(code, row['NAV'])
                if current_nav > row['NAV']:  # Profitable
                    gain_per_unit = current_nav - row['NAV']
                    units_for_exemption = min(row['Units'], remaining_exemption / gain_per_unit)
                    if units_for_exemption > 0:
                        profitable_units.append({
                            'Fund Name': row['Fund Name'],
                            'Purchase Date': row['Date'].strftime('%Y-%m-%d'),
                            'Units Available': row['Units'],
                            'Purchase NAV': row['NAV'],
                            'Current NAV': current_nav,
                            'Gain per Unit': gain_per_unit,
                            'Units to Sell (Tax-Free)': min(row['Units'], units_for_exemption),
                            'Tax-Free Gain': min(row['Units'], units_for_exemption) * gain_per_unit
                        })
        
        if profitable_units:
            st.success("🎯 **Tax-Free Profit Booking Opportunities:**")
            profitable_df = pd.DataFrame(profitable_units)
            profitable_df = profitable_df.sort_values('Tax-Free Gain', ascending=False)
            st.dataframe(profitable_df, use_container_width=True)
            
            total_tax_free_gain = profitable_df['Tax-Free Gain'].sum()
            st.metric("Total Available Tax-Free Gains", f"₹{min(total_tax_free_gain, remaining_exemption):,.1f}")

elif excess_ltcg > 0:
    tax_paid = excess_ltcg * 0.125  # 12.5% LTCG tax
    st.error(f"⚠️ **LTCG Limit Exceeded!** You've exceeded the exemption by ₹{excess_ltcg:,.1f}")
    st.error(f"💸 **Tax Impact:** ₹{tax_paid:,.1f} LTCG tax paid/payable (12.5% on excess amount)")
    
    # Strategic recommendations for over-limit scenarios
    st.warning("🚫 **STOP SELLING RECOMMENDATION:**")
    st.markdown(f"""
    **Immediate Actions:**
    - 🛑 **Avoid further LTCG realizations** this financial year unless absolutely necessary
    - 📅 **Defer planned redemptions** to next FY (April {current_fy_end_year} onwards) 
    - 🔄 **Consider tax loss harvesting** to offset some gains
    """)
    
    # Calculate potential tax loss harvesting benefit
    if carry_forward_opportunities:
        harvestable_losses = sum([abs(opp['Unrealized Loss']) for opp in carry_forward_opportunities])
        potential_offset = min(harvestable_losses, excess_ltcg)
        tax_savings = potential_offset * 0.125
        
        st.info(f"💡 **Tax Loss Harvesting Opportunity:** You can offset ₹{potential_offset:,.1f} of excess LTCG")
        st.info(f"**Potential Tax Savings:** ₹{tax_savings:,.1f}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Excess LTCG", f"₹{excess_ltcg:,.1f}")
        with col2:
            st.metric("Offsettable via Loss Harvesting", f"₹{potential_offset:,.1f}")
    
    # Future planning recommendations
    with st.expander("📋 Future Planning Strategy"):
        st.markdown(f"""
        **Next Financial Year Strategy (FY {current_fy_end_year}-{str(current_fy_end_year + 1)[2:]}):**
        - 🎯 Plan redemptions across multiple years to utilize annual ₹1.25L exemption
        - 📊 Consider SWP (Systematic Withdrawal Plan) instead of lump-sum redemptions
        - ⚖️ Spread large redemptions over 2-3 financial years
        - 📈 Book profits strategically within exemption limits each year
        
        **Long-term Optimization:**
        - 🔄 Regular portfolio rebalancing within exemption limits
        - 📅 Calendar-based profit booking (March each year)
        - 💰 Utilize both spouse's exemption limits if applicable
        """)

# Calculate remaining months in current FY
fy_end_date = datetime(current_fy_end_year, 3, 31)
days_remaining_in_fy = (fy_end_date - current_date).days
months_remaining_in_fy = max(0, days_remaining_in_fy // 30)

if months_remaining_in_fy > 0 and remaining_exemption > 0:
    monthly_exemption = remaining_exemption / max(1, months_remaining_in_fy)
    st.info(f"📅 **Monthly Planning:** You can realize ₹{monthly_exemption:,.1f} LTCG per month for remaining ~{months_remaining_in_fy} months in FY")
elif months_remaining_in_fy > 0 and excess_ltcg > 0:
    st.warning(f"⏰ **Time Alert:** Only ~{months_remaining_in_fy} months left in current FY - Plan next year's redemptions carefully!")
elif days_remaining_in_fy <= 0:
    st.info(f"🗓️ **New FY Started:** Welcome to {current_fy_label}! Fresh ₹1.25L LTCG exemption available.")

# Tax-Efficient Rebalancing Suggestions
st.subheader("⚖️ Tax-Efficient Rebalancing")
if not buys.empty:
    # Calculate current portfolio value
    portfolio_value = {}
    for _, row in buys.iterrows():
        fund = row['Fund Name']
        code = row['Code']
        current_nav = latest_nav_map.get(code, row['NAV'])
        current_value = row['Units'] * current_nav
        
        if fund in portfolio_value:
            portfolio_value[fund] += current_value
        else:
            portfolio_value[fund] = current_value
    
    if portfolio_value:
        total_portfolio = sum(portfolio_value.values())
        rebalancing_suggestions = []
        
        for fund, value in portfolio_value.items():
            weight = (value / total_portfolio) * 100
            if weight > 30:  # Overweight threshold
                rebalancing_suggestions.append({
                    'Fund': fund,
                    'Current Weight': f"{weight:.1f}%",
                    'Current Value': f"₹{value:,.1f}",
                    'Suggestion': 'Consider reducing allocation'
                })
        
        if rebalancing_suggestions:
            st.warning("⚠️ Portfolio Concentration Risk Detected:")
            st.dataframe(pd.DataFrame(rebalancing_suggestions), use_container_width=True)

# Systematic Withdrawal Plan (SWP) Tax Optimizer
st.subheader("📊 SWP Tax Optimization")
st.info("Systematic Withdrawal Plans can be more tax-efficient than lump-sum redemptions")

if not buys.empty:
    with st.expander("Calculate Optimal SWP Amount"):
        col1, col2 = st.columns(2)
        with col1:
            monthly_need = st.number_input("Monthly Income Needed (₹)", min_value=1000, value=25000, step=1000)
        with col2:
            swp_duration = st.selectbox("Duration (months)", [12, 24, 36, 48, 60])
        
        if st.button("Calculate SWP Tax Impact"):
            annual_withdrawal = monthly_need * 12
            total_withdrawal = monthly_need * swp_duration
            
            # Estimate tax on SWP vs lump sum
            st.write(f"**Annual Withdrawal:** ₹{annual_withdrawal:,.1f}")
            st.write(f"**Total Withdrawal:** ₹{total_withdrawal:,.1f}")
            
            if annual_withdrawal <= 125000:
                st.success("✅ Annual withdrawals stay within LTCG exemption limit!")
            else:
                excess_ltcg = annual_withdrawal - 125000
                annual_tax = excess_ltcg * 0.125  # 12.5% LTCG tax
                st.warning(f"⚠️ Annual LTCG tax: ₹{annual_tax:,.1f}")

# Debt Fund Tax Alert (Post-2023 Rules)
st.subheader("⚠️ Debt Fund Tax Alert")
debt_keywords = ['debt', 'bond', 'credit', 'liquid', 'ultra short', 'short term', 'medium term', 'gilt']
debt_funds = df[df['Fund Name'].str.lower().str.contains('|'.join(debt_keywords), na=False)]

if not debt_funds.empty:
    st.warning("📢 Important: Debt fund taxation changed from April 1, 2023")
    debt_fund_names = debt_funds['Fund Name'].unique()
    
    for fund in debt_fund_names:
        fund_data = debt_funds[debt_funds['Fund Name'] == fund]
        pre_april_2023 = fund_data[fund_data['Date'] < '2023-04-01']
        post_april_2023 = fund_data[fund_data['Date'] >= '2023-04-01']
        
        if not pre_april_2023.empty and not post_april_2023.empty:
            st.info(f"📋 {fund}: Mixed taxation (pre & post April 2023 investments)")

# ELSS Lock-in Period Tracker
st.subheader("🔒 ELSS Lock-in Tracker")
elss_funds = df[df['Fund Name'].str.lower().str.contains('elss|tax saver|equity linked', na=False)]
if not elss_funds.empty:
    elss_summary = []
    for _, row in elss_funds.iterrows():
        lock_in_end = row['Date'] + pd.DateOffset(years=3)
        days_remaining = (lock_in_end - datetime.now()).days
        
        if days_remaining > 0:
            elss_summary.append({
                'Fund Name': row['Fund Name'],
                'Investment Date': row['Date'].date(),
                'Lock-in Ends': lock_in_end.date(),
                'Days Remaining': days_remaining,
                'Amount': row['Amount']
            })
    
    if elss_summary:
        elss_df = pd.DataFrame(elss_summary)
        elss_df = elss_df.sort_values('Days Remaining')
        st.dataframe(elss_df, use_container_width=True)
        
        upcoming_unlock = elss_df[elss_df['Days Remaining'] <= 90]
        if not upcoming_unlock.empty:
            st.success(f"🎉 {len(upcoming_unlock)} ELSS investments will unlock within 90 days!")

# Tax Loss Harvesting Calendar
st.subheader("📅 Tax Planning Calendar")
if carry_forward_opportunities:
    st.info("💡 Best months for tax loss harvesting: December-January (before financial year end)")
    
    # Calculate potential tax savings
    total_harvestable_loss = sum([abs(opp['Unrealized Loss']) for opp in carry_forward_opportunities])
    potential_tax_benefit = min(total_harvestable_loss, current_fy_ltcg) * 0.125
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Harvestable Losses", f"₹{total_harvestable_loss:,.1f}")
    with col2:
        st.metric("Potential Tax Savings", f"₹{potential_tax_benefit:,.1f}")

# Fund Category Tax Efficiency
st.subheader("📈 Fund Category Analysis")
equity_keywords = ['equity', 'large cap', 'mid cap', 'small cap', 'multi cap', 'flexi cap', 'index', 'etf']
hybrid_keywords = ['hybrid', 'balanced', 'aggressive', 'conservative', 'arbitrage']

fund_categories = []
for fund in df['Fund Name'].unique():
    fund_lower = fund.lower()
    if any(keyword in fund_lower for keyword in equity_keywords):
        category = 'Equity'
    elif any(keyword in fund_lower for keyword in hybrid_keywords):
        category = 'Hybrid'
    elif any(keyword in fund_lower for keyword in debt_keywords):
        category = 'Debt'
    elif 'elss' in fund_lower or 'tax saver' in fund_lower:
        category = 'ELSS'
    else:
        category = 'Others'
    
    fund_categories.append({'Fund Name': fund, 'Category': category})

if fund_categories:
    category_df = pd.DataFrame(fund_categories)
    category_summary = category_df['Category'].value_counts()
    
    fig_pie = px.pie(
        values=category_summary.values,
        names=category_summary.index,
        title='Portfolio Distribution by Fund Category',
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    st.plotly_chart(fig_pie, use_container_width=True)
    
    # Tax efficiency recommendations
    st.info("""
    **Tax Efficiency by Category:**
    - 🟢 **Equity Funds**: Most tax-efficient (12.5% LTCG after 1 year + ₹1.25L exemption)
    - 🟡 **ELSS**: Tax deduction + same as equity post 3-year lock-in
    - 🟠 **Hybrid Funds**: Taxed as equity if >65% equity allocation
    - 🔴 **Debt Funds**: Least efficient (taxed at slab rate from April 2023)
    """)

# Portfolio overview
with st.expander("📋 Portfolio Overview"):
    st.write("**Current Holdings by Fund:**")
    current_holdings = buys.groupby('Fund Name').agg({
        'Units': 'sum',
        'Amount': 'sum'
    }).reset_index()
    current_holdings['Avg Purchase Price'] = current_holdings['Amount'] / current_holdings['Units']
    st.dataframe(current_holdings, use_container_width=True)

# Interpretation Guide
st.subheader("📚 How to Interpret These Metrics")
st.markdown("""
### Understanding Key Tax Metrics:

1. **Carry-Forward Losses Calculation:**
   - **Formula:** `Carry-Forward Loss = Sum of all (Sale Price - Purchase Price) where result is negative`
   - Only actualized losses (from sold units) are considered
   - Calculated separately for each financial year (April-March)
   - Must be claimed in your tax return (ITR) to carry forward
   - **Key Rules:**
     - Can be carried forward for 8 assessment years
     - Can offset against both STCG and LTCG in future years
     - Must maintain the same head of income (capital gains)

2. **Example Calculation:**
   - Transaction 1: ₹50,000 (Purchase) → ₹45,000 (Sale) = **-₹5,000**  
   - Transaction 2: ₹30,000 → ₹40,000 = +₹10,000  
   - Transaction 3: ₹20,000 → ₹18,000 = **-₹2,000**  
   **Total Carry-Forward Loss = (-5,000) + (-2,000) = ₹7,000**

3. **Realized Capital Gains:**
   - **LTCG (Long-Term Capital Gains):** Profits from investments held >1 year (taxed at 12.5% above ₹1.25L exemption)
   - **STCG (Short-Term Capital Gains):** Profits from investments held ≤1 year (taxed at 20%)

4. **Unrealized Gains:**
   - **LTCG Eligible:** Gains that would qualify for lower tax rate if sold today
   - **STCG (Still Short-Term):** Gains that would be taxed higher if sold today

5. **Tax Loss Harvesting:**
   - Strategy to sell losing positions to offset capital gains
   - Can reduce current or future tax liability
   - Must wait 31 days before repurchasing to avoid "wash sale" rules

6. **LTCG Exemption:**
   - First ₹1.25 lakh of LTCG each financial year is tax-free
   - Plan redemptions strategically to stay within this limit
   - Consider spreading large redemptions across multiple years

7. **FIFO (First-In-First-Out):**
   - Mutual fund redemptions follow FIFO accounting
   - The oldest units are sold first when you redeem
   - Important for tax planning as it affects holding period calculation

### Smart Tax Planning Tips:
- **For Gains:** Wait until investments qualify for LTCG (365+ days) before selling
- **For Losses:** Consider harvesting losses in December/January to offset gains
- **Rebalancing:** Do it gradually within LTCG exemption limits
- **SWP:** Systematic withdrawals can be more tax-efficient than lump sums
- **ELSS:** Remember 3-year lock-in period for tax-saving funds
""")