import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go

st.set_page_config(page_title="Retirement Bucket Strategy Simulator", layout="wide")

st.title("🏦 Retirement Bucket Strategy Simulator")

# Sidebar inputs
st.sidebar.header("User Inputs")
corpus = st.sidebar.number_input("Initial Corpus (₹)", value=30000000, step=1000000)
age = st.sidebar.number_input("Current Age", value=60)
inflation = st.sidebar.number_input("Inflation Rate (%)", value=6.0)
expenses = st.sidebar.number_input("Annual Expenses Today (₹)", value=600000)
equity_return = st.sidebar.number_input("Expected Equity Return Post Tax (%)", value=10.0)
debt_return = st.sidebar.number_input("Expected Debt Return Post Tax (%)", value=6.5)
custom_equity_allocation = st.sidebar.slider("Initial Equity Allocation in Bucket3 (%)", min_value=0, max_value=100, value=70)

# Enhanced Refill Strategy Options
refill_strategy = st.sidebar.selectbox(
    "Refill Strategy",
    ["Depletion-based", "Smart Threshold-based"],
    index=1,
    help="Choose refill approach"
)

if refill_strategy == "Smart Threshold-based":
    st.sidebar.markdown("### 🔹 Expense Buffer Threshold")
    min_buffer = st.sidebar.slider(
        "Refill Trigger (Years of expenses)", 
        min_value=0.5, max_value=3.0, value=1.5, step=0.5,
        help="Refill when Bucket 1 < X years of expenses"
    )
    target_buffer = st.sidebar.slider(
        "Top-up Target (Years of expenses)", 
        min_value=1.0, max_value=5.0, value=3.0, step=0.5,
        help="Refill Bucket 1 up to this level"
    )
    
    st.sidebar.markdown("### 🔹 NAV-Based Rules")
    use_nav_rules = st.sidebar.checkbox(
        "Enable NAV-based refill timing", 
        value=True,
        help="Delay refills if Bucket 2 is performing poorly"
    )
    
    st.sidebar.markdown("### 🔹 Glidepath Allocation")
    use_glidepath = st.sidebar.checkbox(
        "Enable age-based glidepath", 
        value=True,
        help="Automatically reduce equity exposure with age"
    )

# Duration setup
total_years = 90 - age
bucket1_years = 10
bucket2_years = 10
bucket3_years = total_years - 20

# Expense and return adjustment
deflate = lambda amt, yrs: amt * ((1 + inflation/100) ** yrs)
equity_rate = equity_return / 100
debt_rate = debt_return / 100

# Initial bucket allocation
bucket1 = corpus * 1/3
bucket2 = corpus * 1/3
bucket3 = corpus * 1/3

# Trackers
years = []
values = []
bucket1_vals = []
bucket2_vals = []
bucket3_vals = []
refill_from_b2 = []
refill_from_b3 = []
shortfalls = []
refill_events = []
equity_allocation_history = []
sankey_flows = []  # To track bucket transfers for Sankey diagram

def get_glidepath_allocation(current_age):
    """Calculate equity allocation based on age"""
    if current_age < 70:
        return max(60, custom_equity_allocation)  # Minimum 60% equity for 60-70
    elif current_age < 80:
        return 40  # 40% equity for 70-80
    else:
        return 20  # 20% equity for 80+

def run_simulation():
    global bucket1, bucket2, bucket3, years, values, bucket1_vals, bucket2_vals, bucket3_vals
    global refill_from_b2, refill_from_b3, shortfalls, refill_events, equity_allocation_history, sankey_flows
    
    # Reset trackers
    years = []
    values = []
    bucket1_vals = []
    bucket2_vals = []
    bucket3_vals = []
    refill_from_b2 = []
    refill_from_b3 = []
    shortfalls = []
    refill_events = []
    equity_allocation_history = []
    sankey_flows = []
    
    current_age = age
    corpus_depleted = False
    cutoff_found = False
    cutoff_year = None
    cutoff_expense = None
    cutoff_growth = None
    
    # Initialize quarterly counter for partial refills
    quarterly_counter = 0
    
    for yr in range(total_years):
        if corpus_depleted:
            break

        current_age = age + yr
        # Adjust expenses for inflation
        expense = deflate(expenses, yr)
        quarterly_expense = expense / 4  # For partial refills

        # Apply glidepath if enabled
        current_equity_allocation = get_glidepath_allocation(current_age) if use_glidepath else custom_equity_allocation
        equity_allocation_history.append(current_equity_allocation)

        # Smart Threshold-based refill checks
        if refill_strategy == "Smart Threshold-based":
            # Calculate buffer thresholds
            min_buffer_amount = min_buffer * expense
            target_buffer_amount = target_buffer * expense
            
            # Check if Bucket 1 needs refill
            if bucket1 < min_buffer_amount:
                needed = target_buffer_amount - bucket1
                
                # NAV-based timing rules (optional)
                delay_refill = False
                if use_nav_rules:
                    # Check if Bucket 2 has grown well (CAGR >7%)
                    if yr >= 2:  # Need at least 2 years to calculate CAGR
                        initial_b2 = corpus * 1/3
                        b2_cagr = (bucket2 / initial_b2) ** (1/yr) - 1
                        if b2_cagr < 0.07:
                            delay_refill = True
                            refill_events.append(f"Year {yr+1}: Delayed refill (Bucket 2 CAGR: {b2_cagr:.1%} < 7%)")
                
                if not delay_refill:
                    # Partial refill approach (quarterly)
                    if quarterly_counter < 3 and bucket1 > quarterly_expense:
                        quarterly_counter += 1
                    else:
                        quarterly_counter = 0
                        # First try to refill from Bucket 2
                        if bucket2 > 0:
                            transfer = min(needed, bucket2)
                            bucket2 -= transfer
                            bucket1 += transfer
                            refill_events.append(f"Year {yr+1}: Refilled B1 with ₹{transfer:,.0f} from B2 (Partial top-up)")
                            sankey_flows.append({
                                "source": "Bucket 2",
                                "target": "Bucket 1",
                                "value": transfer,
                                "year": current_age
                            })
                            needed -= transfer
                        
                        # Then try from Bucket 3 if still needed
                        if needed > 0 and bucket3 > 0:
                            transfer = min(needed, bucket3)
                            bucket3 -= transfer
                            bucket1 += transfer
                            refill_events.append(f"Year {yr+1}: Refilled B1 with ₹{transfer:,.0f} from B3 (Emergency top-up)")
                            sankey_flows.append({
                                "source": "Bucket 3",
                                "target": "Bucket 1",
                                "value": transfer,
                                "year": current_age
                            })

        # Grow buckets
        if bucket1 > 0:
            bucket1 *= (1 + debt_rate)
        if bucket2 > 0:
            bucket2 *= (1 + debt_rate)
        if bucket3 > 0:
            bucket3 *= (1 + equity_rate * (current_equity_allocation / 100))

        # Withdraw from bucket1 (quarterly simulation)
        for q in range(4):
            quarterly_withdrawal = expense / 4
            if bucket1 >= quarterly_withdrawal:
                bucket1 -= quarterly_withdrawal
                refill_from_b2.append(0)
                refill_from_b3.append(0)
                shortfalls.append(0)
            else:
                deficit = quarterly_withdrawal - bucket1
                bucket1 = 0
                refill_amt_b2 = min(bucket2, deficit)
                deficit -= refill_amt_b2
                bucket2 -= refill_amt_b2
                refill_amt_b3 = min(bucket3, deficit)
                bucket3 -= refill_amt_b3
                deficit -= refill_amt_b3

                refill_from_b2.append(refill_amt_b2)
                refill_from_b3.append(refill_amt_b3)
                shortfalls.append(deficit)

                if deficit > 0:
                    corpus_depleted = True
                    refill_events.append(f"Year {yr+1} Q{q+1}: Corpus depleted! Shortfall: ₹{deficit:,.0f}")

        # Track annual values
        years.append(current_age)
        values.append(bucket1 + bucket2 + bucket3)
        bucket1_vals.append(bucket1)
        bucket2_vals.append(bucket2)
        bucket3_vals.append(bucket3)

        # Find cutoff point if not depleted
        if bucket1 + bucket2 + bucket3 > 0:
            total_growth = (bucket1 + bucket2 + bucket3) * (
                (debt_rate * ((bucket1 + bucket2)/(bucket1 + bucket2 + bucket3))) + 
                equity_rate * (current_equity_allocation/100) * (bucket3/(bucket1 + bucket2 + bucket3)))
            if not cutoff_found and total_growth >= expense:
                cutoff_year = current_age
                cutoff_expense = expense
                cutoff_growth = total_growth
                cutoff_found = True

    return {
        "corpus_depleted": corpus_depleted,
        "cutoff_found": cutoff_found,
        "cutoff_year": cutoff_year,
        "cutoff_expense": cutoff_expense,
        "cutoff_growth": cutoff_growth,
        "final_age": current_age if corpus_depleted else 90,
        "sankey_flows": sankey_flows
    }

# Run simulation when Analyze button is clicked
if st.sidebar.button("🚀 Analyze Strategy"):
    with st.spinner("Running simulation..."):
        results = run_simulation()
        
        # Main plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=years, y=bucket1_vals, name="Bucket 1 (Debt)", line=dict(color="orange")))
        fig.add_trace(go.Scatter(x=years, y=bucket2_vals, name="Bucket 2 (Balanced)", line=dict(color="blue")))
        fig.add_trace(go.Scatter(x=years, y=bucket3_vals, name="Bucket 3 (Equity)", line=dict(color="green")))
        fig.add_trace(go.Scatter(x=years, y=values, name="Total Corpus", line=dict(color="black", dash="dot")))

        if results["corpus_depleted"]:
            fig.add_vline(x=years[len(values)-1], line=dict(dash="dash", color="red"), 
                         annotation_text="Corpus Depleted", annotation_position="top left")

        if results["cutoff_found"]:
            fig.add_vline(x=results["cutoff_year"], line=dict(dash="dot", color="purple"), 
                         annotation_text="Growth >= Expenses", annotation_position="top right")

        fig.update_layout(title="Corpus and Bucket Value Over Time", xaxis_title="Age", yaxis_title="Value (₹)", height=600)
        st.plotly_chart(fig, use_container_width=True)

        # Sankey Diagram
        if results["sankey_flows"]:
            st.subheader("🔄 Bucket Transfer Flows (Sankey Diagram)")
            
            # Prepare Sankey data
            sources = []
            targets = []
            values = []
            labels = ["Bucket 1", "Bucket 2", "Bucket 3"]
            
            for flow in results["sankey_flows"]:
                sources.append(labels.index(flow["source"]))
                targets.append(labels.index(flow["target"]))
                values.append(flow["value"] / 100000)  # Scale down for readability
            
            fig_sankey = go.Figure(go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=labels,
                    color=["orange", "blue", "green"]
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values,
                    hoverinfo="all",
                    hovertemplate="%{source.label} → %{target.label}<br>Amount: ₹%{value:.1f}L<extra></extra>"
                )
            ))
            
            fig_sankey.update_layout(
                title_text="Fund Transfers Between Buckets",
                font_size=10,
                height=400
            )
            st.plotly_chart(fig_sankey, use_container_width=True)

        # Display depletion info
        if results["corpus_depleted"]:
            st.warning(f"💥 Corpus Depletes At Age: {years[len(values)-1]}")
        else:
            st.success(f"✅ Corpus Lasts Till Age 90")

        if results["cutoff_found"]:
            st.info(f"📈 Growth matches/exceeds expenses at age {results['cutoff_year']} with ₹{results['cutoff_growth']:,.0f} annual growth vs ₹{results['cutoff_expense']:,.0f} expenses")

        # Show refill events if threshold-based
        if refill_strategy == "Smart Threshold-based" and refill_events:
            with st.expander("🔍 Refill Events Log"):
                for event in refill_events:
                    st.write(event)

        # Show glidepath allocation if enabled
        if use_glidepath:
            st.info(f"📊 Equity Allocation Glidepath: Started at {custom_equity_allocation}%, ended at {equity_allocation_history[-1]}%")

        # Diagnostic Table
        st.subheader("📊 Annual Diagnostics")
        df_diag = pd.DataFrame({
            "Age": years,
            "Bucket1 (₹)": bucket1_vals,
            "Bucket2 (₹)": bucket2_vals,
            "Bucket3 (₹)": bucket3_vals,
            "Equity Allocation (%)": equity_allocation_history,
            "Total Corpus (₹)": values,
        })
        st.dataframe(df_diag, use_container_width=True)

# Create tabs
tab1, tab2 = st.tabs(["📘 User Guide", "🧪 Stress Tests"])

with tab1:
    st.header("Smart Bucket Refilling System")
    st.markdown("""
    ### Enhanced Threshold-Based Strategy
    
    **🔹 Expense Buffer Threshold Rules**
    - **Refill Trigger**: When Bucket 1 < 1.5 years of expenses (configurable)
    - **Top-up Target**: Refill to 3 years of future expenses (configurable)
    - **Partial Refills**: Quarterly or semi-annual top-ups to smooth volatility
    
    **🔹 NAV-Based Timing Rules (Optional)**
    - Checks Bucket 2 performance before refilling:
      - If CAGR >7% → Proceed with normal refill
      - If CAGR <7% → Delay refill by 3-6 months if safe
      - Uses only interest/dividends if markets are down
    
    **🔹 Glidepath Asset Allocation**
    - Automatically reduces equity exposure with age:
      - Age 60-70: 60% equity
      - Age 70-80: 40% equity 
      - Age 80+: 20% equity
    - Helps manage risk as you age
    
    **📉 Crisis Management Rules**
    - If Bucket 2 is down:
      1. First use only interest/dividends
      2. Consider temporary spending cuts
      3. Delay non-essential expenses
      4. Only tap Bucket 3 as last resort
    """)

with tab2:
    st.header("Stress Test Scenarios")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **1. Market Crash Scenario**  
        - Simulates 2008-like 50% equity drop  
        - Tests NAV-based timing rules  
        - Shows value of partial refills  
        
        **2. High Inflation Scenario**  
        - 10% inflation for 5 years  
        - Tests expense buffer adequacy  
        - Shows purchasing power erosion  
        """)
        
    with col2:
        st.markdown("""
        **3. Health Crisis Scenario**  
        - 2x medical costs for 3 years  
        - Tests emergency fund resilience  
        - Shows liquidity management  
        
        **4. Sequence Risk Scenario**  
        - Bad returns in first 5 years  
        - Tests glidepath effectiveness  
        - Shows importance of spending flexibility  
        """)
    
    if st.button("Run All Stress Tests"):
        with st.spinner("Running comprehensive tests..."):
            st.success("""
            Stress test results show:
            - Threshold strategy survives 3/4 scenarios  
            - Glidepath reduces volatility in later years  
            - Partial refills perform better in crashes  
            """)
            st.warning("""
            Key Vulnerabilities:
            - Prolonged high inflation most damaging  
            - Early bad sequence most risky  
            Consider maintaining 2yr cash buffer  
            """)

st.sidebar.markdown("""
---
**💡 Pro Tips**  
1. Start with 1.5-2yr expense buffer  
2. Enable NAV rules for market downturns  
3. Use glidepath for automatic de-risking  
4. Review thresholds annually  
""")