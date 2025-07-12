# Complete Goal-Based Financial Planner with All Functions
import streamlit as st
import pandas as pd
import numpy as np
import psycopg
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import re
import json
from pathlib import Path

# -------------------- DATABASE CONFIG --------------------
DB_PARAMS = {
    'dbname': 'postgres',
    'user': 'postgres',
    'password': 'admin123',
    'host': 'localhost',
    'port': '5432'
}

def get_db_connection():
    return psycopg.connect(**DB_PARAMS)

# -------------------- DATA FETCHING FUNCTIONS --------------------
def get_goal_mappings():
    """Retrieve all goals from database except Retirement"""
    with get_db_connection() as conn:
        query = """
            SELECT DISTINCT goal_name 
            FROM goals 
            WHERE goal_name != 'Retirement'
            ORDER BY goal_name
        """
        return pd.read_sql(query, conn)

def get_goal_investments(goal_name):
    """Get current investments for a specific goal"""
    with get_db_connection() as conn:
        try:
            # Get equity investments
            equity_query = f"""
            SELECT COALESCE(SUM(current_value), 0) as equity_value
            FROM goals
            WHERE goal_name = '{goal_name}'
            AND investment_type = 'Equity'
            """
            equity_value = pd.read_sql(equity_query, conn).iloc[0,0]
            
            # Get debt investments
            debt_query = f"""
            SELECT COALESCE(SUM(current_value), 0) as debt_value
            FROM goals
            WHERE goal_name = '{goal_name}'
            AND investment_type = 'Debt'
            """
            debt_value = pd.read_sql(debt_query, conn).iloc[0,0]
            
            return equity_value, debt_value
        except Exception as e:
            st.error(f"Error fetching investments: {str(e)}")
            return 0, 0

# -------------------- UTILITY FUNCTIONS --------------------
def format_indian_currency(amount):
    """Format numbers in Indian style (lakhs, crores)"""
    if pd.isna(amount) or amount == 0:
        return "₹0"
    
    amount = float(amount)
    if amount < 100000:
        return f"₹{amount:,.0f}"
    elif amount < 10000000:
        lakhs = amount / 100000
        return f"₹{lakhs:,.1f} L"
    else:
        crores = amount / 10000000
        return f"₹{crores:,.1f} Cr"

def parse_indian_currency(formatted_str):
    """Convert formatted Indian currency string back to float"""
    if formatted_str == "₹0":
        return 0.0
    
    # Remove currency symbol and commas
    clean_str = formatted_str.replace('₹', '').replace(',', '')
    
    # Handle lakhs (L)
    if ' L' in clean_str:
        return float(clean_str.replace(' L', '')) * 100000
    # Handle crores (Cr)
    elif ' Cr' in clean_str:
        return float(clean_str.replace(' Cr', '')) * 10000000
    else:
        return float(clean_str)

def create_goal_speedometer(current, target, title):
    """Create a speedometer-style gauge chart with Indian formatting"""
    def format_for_gauge(value):
        if value < 100000:
            return f"₹{value:,.0f}"
        elif value < 10000000:
            return f"₹{value/100000:,.1f}L"
        else:
            return f"₹{value/10000000:,.1f}Cr"
    
    current_formatted = format_for_gauge(current)
    target_formatted = format_for_gauge(target)
    progress_pct = min(100, (current/target)*100) if target > 0 else 0
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=current,
        number={'valueformat': ",.0f"},
        domain={'x': [0, 1], 'y': [0, 1]},
        title={
            'text': f"{title}<br>{current_formatted} of {target_formatted}",
            'font': {'size': 14}
        },
        gauge={
            'axis': {
                'range': [None, target],
                'tickformat': ",.0f",
                'tickprefix': "₹"
            },
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, target*0.5], 'color': "lightgray"},
                {'range': [target*0.5, target*0.8], 'color': "gray"},
                {'range': [target*0.8, target], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': current
            }
        }
    ))
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=80, b=20)
    )
    return fig

def save_to_json(data, filename="goal_planner_inputs.json"):
    """Save user inputs to a JSON file"""
    try:
        filepath = Path(filename)
        if filepath.exists():
            with open(filepath, 'r') as f:
                existing_data = json.load(f)
            if not isinstance(existing_data, list):
                existing_data = [existing_data]
            existing_data.append(data)
        else:
            existing_data = [data]
        
        with open(filepath, 'w') as f:
            json.dump(existing_data, f, indent=4)
        st.success("Inputs saved successfully!")
    except Exception as e:
        st.error(f"Error saving inputs: {str(e)}")

def calculate_future_value(current_value, years, equity_return, debt_return, inflation, equity_increase_pct, debt_increase_pct, equity_allocation=0.8):
    """Calculate future value of a goal considering all parameters"""
    if years <= 0:
        return current_value
    
    future_value = current_value
    equity_amount = current_value * equity_allocation
    debt_amount = current_value * (1 - equity_allocation)
    
    for year in range(1, years + 1):
        # Calculate returns
        equity_growth = equity_amount * equity_return
        debt_growth = debt_amount * debt_return
        
        # Apply inflation adjustment
        inflation_adjustment = 1 + inflation
        
        # Calculate new values
        equity_amount = (equity_amount + equity_growth) * inflation_adjustment * (1 + equity_increase_pct)
        debt_amount = (debt_amount + debt_growth) * inflation_adjustment * (1 + debt_increase_pct)
        
        future_value = equity_amount + debt_amount
    
    return future_value

# -------------------- CORE CALCULATION FUNCTIONS --------------------
def calculate_goal_corpus(current_value, target_amount, years_to_goal, equity_return, debt_return, current_age, goal_name, equity_increase_pct=0, debt_increase_pct=0):
    """Calculate required investments with sequence risk management"""
    if years_to_goal <= 0:
        return target_amount - current_value, None, None
    
    # Special handling for Emergency Fund
    if goal_name.lower() == "emergency fund":
        equity_return = 0
        equity_pct = 0
        debt_pct = 1
        equity_increase_pct = 0
    
    future_value_needed = max(0, target_amount - current_value)
    projection_data = []
    monthly_investments = []
    
    current_equity = current_value * (0 if goal_name.lower() == "emergency fund" else 0.8)
    current_debt = current_value * (1 if goal_name.lower() == "emergency fund" else 0.2)
    
    for year in range(1, years_to_goal + 1):
        age = current_age + year
        years_remaining = years_to_goal - year
        
        # Dynamic asset allocation
        if goal_name.lower() == "emergency fund":
            equity_pct = 0
            debt_pct = 1
        else:
            equity_pct = max(0.1, 0.8 - (0.7 * (year / years_to_goal)))
            debt_pct = 1 - equity_pct
        
        # Calculate growth with sequence risk adjustment
        equity_growth = current_equity * equity_return * (1 - 0.1 * (year/years_to_goal))
        debt_growth = current_debt * debt_return
        
        current_equity += equity_growth
        current_debt += debt_growth
        total_value = current_equity + current_debt
        
        if year < years_to_goal:
            fv_needed = future_value_needed - total_value
            effective_return = equity_pct*equity_return + debt_pct*debt_return
            annuity_factor = ((1 + effective_return) ** years_remaining - 1) / effective_return
            base_required = max(0, fv_needed / annuity_factor)
            
            equity_investment = (base_required * equity_pct) * (1 + equity_increase_pct) ** (year-1)
            debt_investment = (base_required * debt_pct) * (1 + debt_increase_pct) ** (year-1)
            required_investment = equity_investment + debt_investment
        else:
            required_investment = 0
        
        current_equity += equity_investment
        current_debt += debt_investment
        
        projection_data.append({
            'Year': datetime.now().year + year,
            'Age': age,
            'Years to Goal': years_remaining,
            'Equity Allocation %': equity_pct * 100,
            'Debt Allocation %': debt_pct * 100,
            'Equity Value': current_equity,
            'Debt Value': current_debt,
            'Total Value': current_equity + current_debt,
            'Annual Investment': required_investment,
            'Monthly Investment': required_investment / 12
        })
        
        for month in range(1, 13):
            monthly_investments.append({
                'Year': datetime.now().year + year,
                'Month': month,
                'Equity Investment': equity_investment / 12,
                'Debt Investment': debt_investment / 12,
                'Total Investment': (equity_investment + debt_investment) / 12,
                'Suggested Allocation': f"{equity_pct*100:.0f}% Equity, {debt_pct*100:.0f}% Debt"
            })
    
    return future_value_needed, pd.DataFrame(projection_data), pd.DataFrame(monthly_investments)

# -------------------- GOAL PLANNER TAB --------------------
def create_goal_planner_tab():
    """Create a tab for goal planning calculations"""
    st.header("📈 Goal Planner Calculator")
    
    # Get all goals except Retirement
    goals_df = get_goal_mappings()
    goal_names = goals_df['goal_name'].unique().tolist()
    
    col1, col2 = st.columns(2)
    with col1:
        goal_name = st.selectbox("Select Goal", goal_names, key="goal_planner_select")
        current_value = st.number_input("Current Value (₹)", min_value=0, value=100000, step=10000, key="goal_planner_current_value")
        years_to_goal = st.number_input("Years to Goal", min_value=1, max_value=50, value=10, key="goal_planner_years")
        
    with col2:
        equity_return = st.slider("Equity Return % (post-tax)", 0.0, 20.0, 10.0, 0.1, key="goal_planner_equity_return") / 100
        debt_return = st.slider("Debt Return % (post-tax)", 0.0, 15.0, 6.0, 0.1, key="goal_planner_debt_return") / 100
        inflation = st.slider("Inflation %", 0.0, 15.0, 5.0, 0.1, key="goal_planner_inflation") / 100
    
    col3, col4 = st.columns(2)
    with col3:
        equity_increase_pct = st.slider("Annual Equity Increase %", 0.0, 20.0, 5.0, 0.1, key="goal_planner_equity_increase") / 100
    with col4:
        debt_increase_pct = st.slider("Annual Debt Increase %", 0.0, 20.0, 3.0, 0.1, key="goal_planner_debt_increase") / 100
    
    # Calculate future value
    future_value = calculate_future_value(
        current_value, years_to_goal, equity_return, debt_return, 
        inflation, equity_increase_pct, debt_increase_pct
    )
    
    st.subheader("📊 Projected Future Value")
    st.metric("Future Value of Your Goal", format_indian_currency(future_value))
    
    # Show allocation chart
    allocation_data = {
        'Asset': ['Equity', 'Debt'],
        'Value': [current_value * 0.8, current_value * 0.2]
    }
    fig = px.pie(allocation_data, names='Asset', values='Value', 
                 title="Current Allocation (80% Equity, 20% Debt)")
    st.plotly_chart(fig, use_container_width=True)
    
    # Save to JSON functionality
    st.subheader("💾 Save Your Inputs")
    if st.button("Save Inputs to JSON", key="goal_planner_save_button"):
        input_data = {
            "goal_name": goal_name,
            "current_value": current_value,
            "years_to_goal": years_to_goal,
            "equity_return": equity_return,
            "debt_return": debt_return,
            "inflation": inflation,
            "equity_increase_pct": equity_increase_pct,
            "debt_increase_pct": debt_increase_pct,
            "calculated_future_value": future_value,
            "calculation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        save_to_json(input_data)

# -------------------- GOAL TAB FUNCTION --------------------
def create_goal_tab(goal_name):
    """Create a tab for a specific financial goal"""
    st.header(f"🎯 {goal_name} Planner")
    
    # Get current investments
    current_equity, current_debt = get_goal_investments(goal_name)
    current_total = current_equity + current_debt
    
    with st.sidebar:
        st.subheader(f"🔧 {goal_name} Settings")
        
        # Target inputs
        target_amount = st.number_input("Target Amount (₹)", min_value=10000, value=1000000, step=10000, key=f"{goal_name}_target_amount")
        target_year = st.number_input("Target Year", 
                                    min_value=datetime.now().year + 1,
                                    max_value=datetime.now().year + 50,
                                    value=datetime.now().year + 10,
                                    key=f"{goal_name}_target_year")
        
        # Hide equity-related inputs for Emergency Fund
        if goal_name.lower() != "emergency fund":
            equity_return = st.slider("Equity Return %", 0.0, 20.0, 10.0, 0.1, key=f"{goal_name}_equity_return") / 100
            equity_increase_pct = st.slider("Annual Equity Increase %", 0.0, 20.0, 5.0, 0.1, key=f"{goal_name}_equity_increase") / 100
        else:
            equity_return = 0
            equity_increase_pct = 0
        
        debt_return = st.slider("Debt Return %", 0.0, 15.0, 6.0, 0.1, key=f"{goal_name}_debt_return") / 100
        debt_increase_pct = st.slider("Annual Debt Increase %", 0.0, 20.0, 3.0, 0.1, key=f"{goal_name}_debt_increase") / 100
    
    # Show speedometer chart
    st.subheader("📊 Goal Progress")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.plotly_chart(create_goal_speedometer(current_total, target_amount, goal_name), use_container_width=True)
    with col2:
        st.metric("Target Amount", format_indian_currency(target_amount))
        st.metric("Current Value", format_indian_currency(current_total))
        st.metric("Years Remaining", max(0, target_year - datetime.now().year))
    
    # Calculate and show investment plan
    corpus_needed, projection_df, monthly_investments = calculate_goal_corpus(
        current_total, target_amount, target_year - datetime.now().year,
        equity_return, debt_return, 30, goal_name, equity_increase_pct, debt_increase_pct
    )
    
    st.subheader("📅 Yearly Investment Plan")
    display_df = projection_df[['Year', 'Annual Investment', 'Equity Allocation %', 'Debt Allocation %']].copy()
    display_df['Annual Investment'] = display_df['Annual Investment'].apply(format_indian_currency)
    display_df['Monthly Equivalent'] = (display_df['Annual Investment']
                                      .apply(lambda x: parse_indian_currency(x)) / 12).apply(format_indian_currency)
    
    st.dataframe(
        display_df.rename(columns={
            'Annual Investment': 'Annual (₹)',
            'Monthly Equivalent': 'Monthly (₹)',
            'Equity Allocation %': 'Equity %',
            'Debt Allocation %': 'Debt %'
        }),
        hide_index=True,
        use_container_width=True
    )
    
    with st.expander("📝 View Detailed Monthly Plan"):
        monthly_display = monthly_investments.copy()
        monthly_display['Total Investment'] = monthly_display['Total Investment'].apply(format_indian_currency)
        st.dataframe(
            monthly_display[['Year', 'Month', 'Total Investment', 'Suggested Allocation']],
            hide_index=True,
            use_container_width=True
        )
    
    # Special advice for Emergency Fund
    if goal_name.lower() == "emergency fund":
        st.warning("""
        **Emergency Fund Special Considerations:**
        - 100% allocated to debt instruments for capital preservation
        - Maintain high liquidity (liquid funds, short-term FDs)
        - Keep 3-6 months of expenses
        - Replenish immediately after withdrawals
        """)

# -------------------- MAIN APP --------------------
def main():
    st.set_page_config(
        page_title="Goal-Based Financial Planner",
        page_icon="💰",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Get all goals from database
    goals_df = get_goal_mappings()
    
    # Create tabs - first the Goal Planner, then individual goal tabs
    tab_titles = ["📈 Goal Planner"] + [f"🎯 {goal}" for goal in goals_df['goal_name'].unique()]
    tabs = st.tabs(tab_titles)
    
    # Goal Planner tab
    with tabs[0]:
        create_goal_planner_tab()
    
    # Individual goal tabs
    for i, goal_name in enumerate(goals_df['goal_name'].unique(), start=1):
        with tabs[i]:
            create_goal_tab(goal_name)

if __name__ == "__main__":
    main()