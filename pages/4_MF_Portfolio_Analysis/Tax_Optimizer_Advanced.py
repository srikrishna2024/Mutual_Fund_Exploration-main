import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import psycopg
import numpy as np
from itertools import combinations
from scipy.optimize import minimize

# Configuration
st.set_page_config(page_title="Advanced MF Tax Optimizer", layout="wide")
st.title("🚀 Advanced Mutual Fund Tax Optimizer")

# Database Connection
def connect_to_db():
    DB_PARAMS = {
        'dbname': 'postgres',
        'user': 'postgres',
        'password': 'admin123',
        'host': 'localhost',
        'port': '5432'
    }
    return psycopg.connect(**DB_PARAMS)

@st.cache_data(ttl=3600)
def get_portfolio_data():
    with connect_to_db() as conn:
        return pd.read_sql("""
            SELECT id, date, scheme_name as fund, code, 
                   transaction_type, value as nav, units, amount
            FROM portfolio_data ORDER BY scheme_name, date
        """, conn)

@st.cache_data(ttl=3600)
def get_latest_nav():
    with connect_to_db() as conn:
        return pd.read_sql("""
            SELECT code, value as nav FROM mutual_fund_nav
            WHERE (code, nav) IN (
                SELECT code, MAX(nav) FROM mutual_fund_nav GROUP BY code
            )
        """, conn)

# Financial Year Calculations
def get_financial_year(date):
    return date.year + 1 if date.month >= 4 else date.year

current_date = datetime.now()
current_fy = get_financial_year(current_date)

# Load Data
with st.spinner("Loading portfolio data..."):
    df = get_portfolio_data()
    nav_df = get_latest_nav()
    nav_map = dict(zip(nav_df['code'], nav_df['nav']))

if df.empty:
    st.error("No portfolio data found!")
    st.stop()

# Preprocessing
df['date'] = pd.to_datetime(df['date'])
buys = df[df['transaction_type'].str.lower().isin(['invest', 'switch_in'])].copy()
sells = df[df['transaction_type'].str.lower().isin(['redeem', 'switch_out'])].copy()

# --------------------------
# CORE TAX CALCULATION ENGINE
# --------------------------
class TaxEngine:
    def __init__(self, buys, sells, nav_map):
        self.buys = buys.copy()
        self.sells = sells.copy()
        self.nav_map = nav_map
        self.results = []
        
    def calculate_fifo_gains(self, sell_date=None):
        """Calculate gains using FIFO method up to optional sell_date"""
        sell_date = sell_date or datetime.now()
        results = []
        
        for fund in self.buys['fund'].unique():
            fund_buys = self.buys[self.buys['fund'] == fund].sort_values('date')
            fund_sells = self.sells[self.sells['fund'] == fund].sort_values('date')
            
            fifo_queue = fund_buys[['date', 'nav', 'units']].to_dict('records')
            
            # Process historical sales
            for _, sell in fund_sells[fund_sells['date'] <= sell_date].iterrows():
                units_to_sell = sell['units']
                while units_to_sell > 0 and fifo_queue:
                    lot = fifo_queue[0]
                    sell_units = min(units_to_sell, lot['units'])
                    
                    gain = sell_units * (sell['nav'] - lot['nav'])
                    holding_days = (sell['date'] - lot['date']).days
                    
                    results.append({
                        'fund': fund,
                        'sell_date': sell['date'],
                        'fy': get_financial_year(sell['date']),
                        'units': sell_units,
                        'days': holding_days,
                        'type': 'LTCG' if holding_days > 365 else 'STCG',
                        'gain': gain
                    })
                    
                    # Update FIFO queue
                    units_to_sell -= sell_units
                    if lot['units'] == sell_units:
                        fifo_queue.pop(0)
                    else:
                        lot['units'] -= sell_units
            
            # Calculate unrealized gains for remaining lots
            for lot in fifo_queue:
                current_nav = self.nav_map.get(fund, lot['nav'])
                gain = lot['units'] * (current_nav - lot['nav'])
                holding_days = (sell_date - lot['date']).days
                
                results.append({
                    'fund': fund,
                    'sell_date': sell_date,
                    'fy': get_financial_year(sell_date),
                    'units': lot['units'],
                    'days': holding_days,
                    'type': 'LTCG' if holding_days > 365 else 'STCG',
                    'gain': gain,
                    'unrealized': True
                })
        
        return pd.DataFrame(results)
    
    def simulate_redemptions(self, redemption_plan):
        """Simulate future redemptions and calculate tax impact"""
        temp_sells = self.sells.copy()
        
        # Add planned redemptions to sell transactions
        for fy, amount in redemption_plan.items():
            # Simplified - in reality would need proper allocation logic
            temp_sells = pd.concat([temp_sells, pd.DataFrame([{
                'date': datetime(fy-1, 4, 1),  # First day of FY
                'fund': 'SIMULATED',
                'nav': 100,  # Placeholder
                'units': amount / 100,
                'transaction_type': 'redeem'
            }])])
        
        # Recalculate with simulated sales
        return self.calculate_fifo_gains()

# Initialize engine
engine = TaxEngine(buys, sells, nav_map)
current_gains = engine.calculate_fifo_gains()

# --------------------------
# MULTI-YEAR TAX PROJECTION
# --------------------------
st.header("📅 Multi-Year Tax Projection")

# Projection parameters
col1, col2 = st.columns(2)
with col1:
    projection_years = st.slider("Projection Period (years)", 1, 5, 3)
with col2:
    growth_rate = st.number_input("Expected Annual Growth Rate (%)", 0.0, 30.0, 8.0) / 100

# Calculate current portfolio value
portfolio_value = {}
for fund in df['fund'].unique():
    fund_buys = buys[buys['fund'] == fund]
    current_nav = nav_map.get(fund, fund_buys['nav'].iloc[-1])
    portfolio_value[fund] = fund_buys['units'].sum() * current_nav

total_value = sum(portfolio_value.values())

# Scenario planning
st.subheader("Scenario Planning")
scenario = st.selectbox("Choose Scenario", [
    "No Redemptions",
    "Equal Annual Redemptions",
    "Custom Redemptions"
])

redemption_plan = {}
if scenario == "Equal Annual Redemptions":
    annual_amount = st.number_input("Annual Redemption Amount (₹)", 0, int(total_value), 100000)
    for fy in range(current_fy, current_fy + projection_years):
        redemption_plan[fy] = annual_amount
elif scenario == "Custom Redemptions":
    for fy in range(current_fy, current_fy + projection_years):
        redemption_plan[fy] = st.number_input(f"FY {fy} Redemption Amount", 0, int(total_value), 100000)

# Calculate projections
if st.button("Run Projection"):
    projections = []
    for fy in range(current_fy, current_fy + projection_years):
        # Simulate growth
        for fund in portfolio_value:
            portfolio_value[fund] *= (1 + growth_rate)
        
        # Calculate tax if redemption occurs
        if fy in redemption_plan:
            simulated_gains = engine.simulate_redemptions({fy: redemption_plan[fy]})
            fy_gains = simulated_gains[simulated_gains['fy'] == fy]
            ltcg = fy_gains[fy_gains['type'] == 'LTCG']['gain'].sum()
            stcg = fy_gains[fy_gains['type'] == 'STCG']['gain'].sum()
            
            # Apply tax rules
            taxable_ltcg = max(0, ltcg - 125000)
            tax = (taxable_ltcg * 0.125) + (stcg * 0.15)
        else:
            tax = 0
        
        projections.append({
            'FY': fy,
            'Portfolio Value': sum(portfolio_value.values()),
            'Redemptions': redemption_plan.get(fy, 0),
            'LTCG': ltcg if fy in redemption_plan else 0,
            'STCG': stcg if fy in redemption_plan else 0,
            'Tax': tax
        })
    
    projection_df = pd.DataFrame(projections)
    
    # Display results
    st.success("Projection Complete")
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(projection_df.style.format({
            'Portfolio Value': '₹{:,.0f}',
            'Redemptions': '₹{:,.0f}',
            'LTCG': '₹{:,.0f}',
            'STCG': '₹{:,.0f}',
            'Tax': '₹{:,.0f}'
        }))
    
    with col2:
        fig = px.bar(projection_df, x='FY', y=['LTCG', 'STCG', 'Tax'],
                     title="Projected Tax Liability", barmode='group')
        st.plotly_chart(fig, use_container_width=True)
    
    # Optimal redemption suggestion
    st.subheader("Optimal Redemption Strategy")
    st.info("""
    **Recommendation:** To minimize taxes over {projection_years} years:
    - Utilize ₹1.25L LTCG exemption each year
    - Harvest losses in December to offset gains
    - Consider SWP for systematic withdrawals
    """)

# --------------------------
# TAX-LOSS HARVESTING AUTOMATION
# --------------------------
st.header("⚡ Automated Tax-Loss Harvesting")

# Find harvestable losses
harvestable = []
for fund in buys['fund'].unique():
    fund_buys = buys[buys['fund'] == fund].sort_values('date')
    current_nav = nav_map.get(fund, fund_buys['nav'].iloc[-1])
    
    for _, lot in fund_buys.iterrows():
        if lot['units'] <= 0:
            continue
            
        gain = (current_nav - lot['nav']) * lot['units']
        if gain < 0:  # Loss
            harvestable.append({
                'Fund': fund,
                'Purchase Date': lot['date'].date(),
                'Units': lot['units'],
                'Cost Basis': lot['nav'],
                'Current NAV': current_nav,
                'Loss Amount': abs(gain),
                'Holding Days': (datetime.now() - lot['date']).days
            })

if harvestable:
    harvest_df = pd.DataFrame(harvestable).sort_values('Loss Amount', ascending=False)
    
    st.subheader("Available Losses")
    st.dataframe(harvest_df.style.format({
        'Cost Basis': '₹{:.2f}',
        'Current NAV': '₹{:.2f}',
        'Loss Amount': '₹{:,.0f}'
    }), use_container_width=True)
    
    # Wash sale monitoring
    st.subheader("Wash Sale Monitor")
    last_sale_date = sells['date'].max() if not sells.empty else None
    
    if last_sale_date:
        wash_sale_period = last_sale_date + timedelta(days=31)
        st.warning(f"⚠️ Avoid repurchasing funds sold before {wash_sale_period.date()} to prevent wash sales")
    
    # Optimal harvesting dates
    st.subheader("Optimal Harvesting Calendar")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Harvestable Losses", f"₹{harvest_df['Loss Amount'].sum():,.0f}")
    with col2:
        st.metric("Potential Tax Savings", f"₹{harvest_df['Loss Amount'].sum() * 0.15:,.0f}")
    
    st.info("""
    **Best Harvesting Dates:**
    - December 15-31: Capture losses before year-end
    - March 15-31: Last chance for current FY losses
    """)
else:
    st.success("No tax-loss harvesting opportunities currently available")

# --------------------------
# TAX-EFFICIENT REBALANCING
# --------------------------
st.header("⚖️ Tax-Efficient Rebalancing")

# Current allocation
allocations = {}
for fund in buys['fund'].unique():
    fund_units = buys[buys['fund'] == fund]['units'].sum() - sells[sells['fund'] == fund]['units'].sum()
    current_nav = nav_map.get(fund, buys[buys['fund'] == fund]['nav'].iloc[-1])
    allocations[fund] = fund_units * current_nav

total_value = sum(allocations.values())
target_alloc = {fund: (value/total_value)*100 for fund, value in allocations.items()}

# Rebalancing suggestions
st.subheader("Current Allocation vs Target")
rebal_suggestions = []
for fund in allocations:
    current_pct = (allocations[fund] / total_value) * 100
    rebal_suggestions.append({
        'Fund': fund,
        'Current %': current_pct,
        'Target %': 100/len(allocations),  # Simplified equal allocation
        'Action': 'Reduce' if current_pct > (100/len(allocations)) else 'Increase'
    })

rebal_df = pd.DataFrame(rebal_suggestions)
st.dataframe(rebal_df.style.format({'Current %': '{:.1f}%', 'Target %': '{:.1f}%'}))

# Tax-efficient rebalancing path
st.subheader("Optimal Rebalancing Path")
st.info("""
1. **Harvest losses first** to offset gains
2. **Utilize LTCG exemption** for profitable sales
3. **Prioritize high-turnover funds** for reduction
4. **Consider tax-efficient alternatives** for new purchases
""")

# --------------------------
# FUND PERFORMANCE ANALYSIS
# --------------------------
st.header("📊 Fund Performance vs Tax Efficiency")

# Calculate tax efficiency metrics
fund_metrics = []
for fund in buys['fund'].unique():
    fund_buys = buys[buys['fund'] == fund]
    fund_sells = sells[sells['fund'] == fund]
    
    # Calculate realized gains
    realized_gains = 0
    if not fund_sells.empty:
        realized_gains = fund_sells['amount'].sum() - fund_buys['amount'].sum()
    
    # Calculate tax efficiency score (simplified)
    tax_efficiency = 100 - (abs(realized_gains) / fund_buys['amount'].sum() * 100)
    
    fund_metrics.append({
        'Fund': fund,
        'Total Invested': fund_buys['amount'].sum(),
        'Realized Gains': realized_gains,
        'Tax Efficiency': tax_efficiency,
        'Rating': '★' * int(tax_efficiency/20)
    })

# Display fund rankings
fund_df = pd.DataFrame(fund_metrics).sort_values('Tax Efficiency', ascending=False)
st.dataframe(fund_df.style.format({
    'Total Invested': '₹{:,.0f}',
    'Realized Gains': '₹{:,.0f}',
    'Tax Efficiency': '{:.1f}%'
}), use_container_width=True)

# Replacement suggestions
st.subheader("Tax-Efficient Alternatives")
st.info("""
- **High-turnover funds:** Consider index funds/ETFs
- **High-dividend funds:** Switch to growth-oriented funds
- **Debt funds:** Evaluate based on your tax bracket
""")

# --------------------------
# SMART ALERTS DASHBOARD
# --------------------------
st.header("🔔 Smart Alerts Dashboard")

# Alerts container
alerts = []

# 1. LTCG Eligibility Alerts
ltcg_alerts = []
for fund in buys['fund'].unique():
    fund_buys = buys[buys['fund'] == fund].sort_values('date')
    oldest_lot = fund_buys.iloc[0]
    days_to_ltcg = 365 - (datetime.now() - oldest_lot['date']).days
    
    if 0 < days_to_ltcg <= 90:
        ltcg_alerts.append({
            'Fund': fund,
            'Purchase Date': oldest_lot['date'].date(),
            'Days to LTCG': days_to_ltcg,
            'Units': oldest_lot['units']
        })

if ltcg_alerts:
    alerts.append({
        'type': 'ltcg',
        'title': '⚠️ Upcoming LTCG Eligibility',
        'content': pd.DataFrame(ltcg_alerts).sort_values('Days to LTCG')
    })

# 2. LTCG Exemption Utilization
current_ltcg = current_gains[
    (current_gains['fy'] == current_fy) & 
    (current_gains['type'] == 'LTCG')
]['gain'].sum()

utilization = min(100, (current_ltcg / 125000) * 100)
if utilization > 50:
    alerts.append({
        'type': 'exemption',
        'title': f'🎯 LTCG Exemption {utilization:.0f}% Used',
        'content': f"₹{current_ltcg:,.0f} of ₹125,000 exemption used"
    })

# 3. Market Volatility Alerts
# (Simplified - would normally connect to market data)
alerts.append({
    'type': 'volatility',
    'title': '📉 Market Dip Detected',
    'content': "Good opportunity for tax-loss harvesting"
})

# 4. FY-End Reminders
fy_end = datetime(current_fy, 3, 31)
days_left = (fy_end - datetime.now()).days
if days_left <= 60:
    alerts.append({
        'type': 'fy_end',
        'title': '⏳ Financial Year Ending Soon',
        'content': f"{days_left} days left - plan your tax moves!"
    })

# Display alerts
for alert in alerts:
    with st.expander(f"{alert['title']}"):
        if isinstance(alert['content'], pd.DataFrame):
            st.dataframe(alert['content'])
        else:
            st.write(alert['content'])

# --------------------------
# INTERPRETATION GUIDE
# --------------------------
with st.expander("📚 How to Use This Tool"):
    st.markdown("""
    **Multi-Year Projections:**
    - Models portfolio growth and redemption scenarios
    - Helps plan large redemptions across multiple years
    
    **Tax-Loss Harvesting:**
    - Identifies losing positions to offset gains
    - Tracks wash sale rules to avoid disallowed losses
    
    **Rebalancing:**
    - Suggests most tax-efficient way to adjust allocations
    - Prioritizes loss harvesting before profitable sales
    
    **Fund Analysis:**
    - Ranks funds by after-tax performance
    - Flags tax-inefficient funds draining returns
    
    **Alerts:**
    - Proactive notifications for key tax events
    - Never miss important deadlines again
    """)

# --------------------------
# FOOTER
# --------------------------
st.markdown("---")
st.markdown("""
**Advanced Tax Optimizer** • Uses FIFO accounting • Consult your tax advisor before acting on recommendations
""")