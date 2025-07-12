# Other Goals Financial Planner with Emergency Fund Special Handling
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import psycopg
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os

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
        return f"₹{lakhs:,.2f} L"
    else:
        crores = amount / 10000000
        return f"₹{crores:,.2f} Cr"

def create_speedometer(current, target, title):
    """Create a speedometer-style gauge chart with values in Lakhs or Crores"""
    if target < 10000000:  # Less than 1 Cr
        current_display = current / 100000
        target_display = target / 100000
        suffix = " L"
    else:
        current_display = current / 10000000
        target_display = target / 10000000
        suffix = " Cr"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = current_display,
        number = {'suffix': suffix, 'valueformat': ".2f"},
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title},
        gauge = {
            'axis': {
                'range': [None, target_display],
                'tickformat': ".1f",
                'ticksuffix': suffix
            },
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, target_display*0.5], 'color': "lightgray"},
                {'range': [target_display*0.5, target_display*0.8], 'color': "gray"},
                {'range': [target_display*0.8, target_display], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': current_display
            }
        }
    ))
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

# -------------------- GOAL PLANNING FUNCTIONS --------------------
def get_goals():
    """Get all goals except retirement from the database"""
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
        equity_query = f"""
        SELECT COALESCE(SUM(pd.units * mf.value), 0) as equity_value
        FROM portfolio_data pd
        JOIN mutual_fund_nav mf ON pd.code = mf.code
        JOIN goals g ON pd.code = g.scheme_code
        WHERE g.goal_name = '{goal_name}'
        AND g.investment_type = 'Equity'
        AND mf.nav = (SELECT MAX(nav) FROM mutual_fund_nav WHERE code = pd.code)
        AND pd.transaction_type IN ('invest', 'switch_in')
        """
        equity_value = pd.read_sql(equity_query, conn).iloc[0,0]
        
        debt_query = f"""
        SELECT COALESCE(SUM(current_value), 0) as debt_value
        FROM goals
        WHERE goal_name = '{goal_name}'
        AND investment_type = 'Debt'
        """
        debt_value = pd.read_sql(debt_query, conn).iloc[0,0]
        
        return equity_value, debt_value

def calculate_emergency_fund(
    monthly_expenses,
    months_needed,
    inflation,
    debt_return=0.05
):
    """
    Calculate emergency fund corpus needed (debt only)
    Returns corpus_needed, monthly_investments
    """
    # Calculate corpus needed with inflation adjustment
    corpus_needed = monthly_expenses * months_needed * (1 + inflation) ** (months_needed/12)
    
    # Calculate monthly investments needed (simple calculation for debt only)
    monthly_investment = corpus_needed / months_needed
    
    # Create simple projection
    monthly_investment_data = []
    corpus_projection = []
    current_corpus = 0
    
    for month in range(1, months_needed + 1):
        # Apply monthly return
        monthly_return = (1 + debt_return) ** (1/12) - 1
        current_corpus = current_corpus * (1 + monthly_return) + monthly_investment
        
        monthly_investment_data.append({
            'Month': month,
            'Investment': monthly_investment,
            'Projected Corpus': current_corpus
        })
    
    corpus_projection.append({
        'Total Corpus Needed': corpus_needed,
        'Monthly Investment': monthly_investment,
        'Months': months_needed
    })
    
    return corpus_needed, monthly_investment, pd.DataFrame(monthly_investment_data), pd.DataFrame(corpus_projection)

def calculate_goal_corpus(
    current_goal_value,
    years_to_goal,
    current_equity=0,
    current_debt=0,
    equity_return=0.10,
    debt_return=0.06,
    inflation=0.06,
    annual_increase=0.05
):
    """Calculate corpus needed and investment plan for regular goals"""
    corpus_needed = current_goal_value * (1 + inflation) ** years_to_goal
    
    projection_data = []
    monthly_investment_data = []
    
    current_equity_value = current_equity
    current_debt_value = current_debt
    
    for year in range(1, years_to_goal + 1):
        years_remaining = years_to_goal - year
        
        # Smooth glide path from 80% to 30% equity
        equity_pct = max(0.3, 0.8 - (0.5 * (year / years_to_goal)))
        debt_pct = 1 - equity_pct
        
        equity_growth = current_equity_value * equity_return
        debt_growth = current_debt_value * debt_return
        
        current_equity_value += equity_growth
        current_debt_value += debt_growth
        total_corpus = current_equity_value + current_debt_value
        
        if year < years_to_goal:
            fv_needed = corpus_needed - total_corpus
            
            effective_return = equity_pct * equity_return + debt_pct * debt_return
            if effective_return != annual_increase:
                annuity_factor = (((1 + effective_return) ** years_remaining - 
                                 (1 + annual_increase) ** years_remaining) / 
                                (effective_return - annual_increase))
            else:
                annuity_factor = years_remaining * (1 + effective_return) ** (years_remaining - 1)
            
            required_investment = max(0, fv_needed / annuity_factor)
        else:
            required_investment = 0
        
        equity_investment = required_investment * equity_pct
        debt_investment = required_investment * debt_pct
        
        current_equity_value += equity_investment
        current_debt_value += debt_investment
        total_corpus = current_equity_value + current_debt_value
        
        projection_data.append({
            'Year': year,
            'Years Remaining': years_remaining,
            'Equity Allocation %': equity_pct * 100,
            'Debt Allocation %': debt_pct * 100,
            'Equity Value': current_equity_value,
            'Debt Value': current_debt_value,
            'Total Corpus': total_corpus,
            'Annual Investment': required_investment,
            'Equity Investment': equity_investment,
            'Debt Investment': debt_investment,
            'Inflated Goal': corpus_needed
        })
        
        for month in range(1, 13):
            monthly_investment_data.append({
                'Year': year,
                'Month': month,
                'Equity Investment': equity_investment / 12,
                'Debt Investment': debt_investment / 12,
                'Total Investment': (equity_investment + debt_investment) / 12
            })
    
    projection_df = pd.DataFrame(projection_data)
    monthly_investments = pd.DataFrame(monthly_investment_data)
    
    if not projection_df.empty:
        avg_annual_investment = projection_df['Annual Investment'].mean()
    else:
        avg_annual_investment = 0
    
    annual_investments = {
        'total': avg_annual_investment,
        'equity': avg_annual_investment * projection_df['Equity Allocation %'].iloc[0] / 100,
        'debt': avg_annual_investment * projection_df['Debt Allocation %'].iloc[0] / 100
    }
    
    return corpus_needed, annual_investments, projection_df, monthly_investments, current_goal_value

def plot_glide_path(projection_df):
    """Plot asset allocation glide path"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=projection_df['Year'],
        y=projection_df['Equity Allocation %'],
        name='Equity %',
        line=dict(color='blue', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=projection_df['Year'],
        y=projection_df['Debt Allocation %'],
        name='Debt %',
        line=dict(color='orange', width=3)
    ))
    
    fig.update_layout(
        title='Asset Allocation Glide Path',
        xaxis_title='Year',
        yaxis_title='Allocation %',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    return fig

def plot_cashflow_plan(monthly_investments, corpus_needed, equity_return=None, debt_return=None):
    """Plot the cashflow plan for the goal"""
    # Check which type of investments we're dealing with
    if 'Investment' in monthly_investments.columns:  # Emergency fund case
        monthly_investments['Cumulative Investment'] = monthly_investments['Investment'].cumsum()
        investment_col = 'Investment'
    else:  # Regular goal case
        monthly_investments['Cumulative Investment'] = monthly_investments['Total Investment'].cumsum()
        investment_col = 'Total Investment'
    
    start_date = pd.Timestamp(datetime.now().date()).replace(day=1)
    monthly_investments['Date'] = pd.date_range(
        start=start_date,
        periods=len(monthly_investments),
        freq='MS'
    )
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(go.Scatter(
        x=monthly_investments['Date'],
        y=monthly_investments[investment_col],
        name='Monthly Investment',
        line=dict(color='blue', width=2),
        hovertemplate='₹%{y:,.0f}<extra></extra>'
    ), secondary_y=False)
    
    if debt_return:
        monthly_debt_return = (1 + debt_return) ** (1/12) - 1
        monthly_investments['Projected Corpus'] = 0.0
        
        for i in range(len(monthly_investments)):
            if i == 0:
                monthly_investments.loc[i, 'Projected Corpus'] = monthly_investments.loc[i, investment_col]
            else:
                monthly_investments.loc[i, 'Projected Corpus'] = (
                    monthly_investments.loc[i-1, 'Projected Corpus'] * (1 + monthly_debt_return) + 
                    monthly_investments.loc[i, investment_col]
                )
        
        fig.add_trace(go.Scatter(
            x=monthly_investments['Date'],
            y=monthly_investments['Projected Corpus'],
            name='Projected Corpus',
            line=dict(color='green', width=3),
            hovertemplate='₹%{y:,.0f}<extra></extra>'
        ), secondary_y=True)
    
    fig.add_hline(
        y=corpus_needed,
        line_dash="dash",
        line_color="purple",
        annotation_text=f"Goal Amount: {format_indian_currency(corpus_needed)}",
        annotation_position="bottom right",
        secondary_y=True
    )
    
    fig.update_layout(
        title='Monthly Investment Plan',
        xaxis_title='Date',
        yaxis_title='Monthly Investment (₹)',
        yaxis2_title='Projected Corpus (₹)',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    fig.update_yaxes(
        title_text="Monthly Investment (₹)",
        secondary_y=False,
        tickprefix="₹"
    )
    
    fig.update_yaxes(
        title_text="Projected Corpus (₹)",
        secondary_y=True,
        tickprefix="₹"
    )
    
    return fig

# -------------------- USER PREFERENCES --------------------
def load_user_prefs(goal_name):
    """Load user preferences for a specific goal"""
    prefs_file = f"goal_prefs_{goal_name.lower().replace(' ', '_')}.json"
    if os.path.exists(prefs_file):
        with open(prefs_file, 'r') as f:
            return json.load(f)
    return {}

def save_user_prefs(goal_name, prefs):
    """Save user preferences for a specific goal"""
    prefs_file = f"goal_prefs_{goal_name.lower().replace(' ', '_')}.json"
    with open(prefs_file, 'w') as f:
        json.dump(prefs, f, indent=4)

# -------------------- STREAMLIT UI --------------------
def goal_selection_tab():
    st.header("🎯 Select Your Financial Goals")
    
    goals_df = get_goals()
    
    if goals_df.empty:
        st.warning("No financial goals found in database (excluding Retirement).")
        st.info("Please add goals in the database to use this planner.")
        return
    
    st.subheader("Available Financial Goals")
    selected_goals = []
    
    for goal in goals_df['goal_name'].unique():
        if st.checkbox(goal, key=f"goal_{goal}"):
            selected_goals.append(goal)
    
    if st.button("Create Planners for Selected Goals", type="primary"):
        if not selected_goals:
            st.warning("Please select at least one goal")
        else:
            st.session_state['selected_goals'] = selected_goals
            st.session_state['current_goal_index'] = 0
            st.rerun()

def goal_planner_tab(goal_name):
    st.header(f"📊 {goal_name} Goal Planner")
    
    user_prefs = load_user_prefs(goal_name)
    current_equity, current_debt = get_goal_investments(goal_name)
    
    with st.sidebar:
        st.subheader(f"🔢 {goal_name} Details")
        
        if goal_name.lower() == "emergency fund":
            # Special inputs for Emergency Fund
            monthly_expenses = st.number_input(
                "Current Monthly Expenses (₹)", 
                min_value=10000, 
                value=int(user_prefs.get('monthly_expenses', 50000)), 
                step=5000
            )
            
            months_needed = st.number_input(
                "Months of Expenses to Cover", 
                min_value=3, 
                max_value=24, 
                value=int(user_prefs.get('months_needed', 6))
            )
            
            inflation = st.slider(
                "Expected Inflation %", 
                min_value=0.0, 
                max_value=10.0, 
                value=float(user_prefs.get('inflation', 6.0)), 
                step=0.1
            ) / 100
            
            debt_return = st.slider(
                "Expected Debt Return %", 
                min_value=0.0, 
                max_value=8.0, 
                value=float(user_prefs.get('debt_return', 5.0)), 
                step=0.1,
                help="Conservative return expectation for debt instruments"
            ) / 100
            
            auto_load = st.checkbox(
                "Auto-load emergency fund investments", 
                value=user_prefs.get('auto_load', True)
            )
            
            if auto_load:
                st.metric("Current Emergency Fund", format_indian_currency(current_debt))
            else:
                current_debt = st.number_input(
                    "Current Emergency Fund (₹)", 
                    min_value=0, 
                    value=int(user_prefs.get('current_debt', current_debt)), 
                    step=10000
                )
            
            if st.button("💾 Save Settings", key=f"save_{goal_name}"):
                user_prefs.update({
                    'monthly_expenses': monthly_expenses,
                    'months_needed': months_needed,
                    'inflation': inflation * 100,
                    'debt_return': debt_return * 100,
                    'current_debt': current_debt,
                    'auto_load': auto_load
                })
                save_user_prefs(goal_name, user_prefs)
                st.success("Settings saved!")
        else:
            # Standard inputs for other goals
            current_goal_value = st.number_input(
                "Current Value of Goal (₹)", 
                min_value=10000, 
                value=int(user_prefs.get('current_goal_value', 1000000)), 
                step=10000
            )
            
            years_to_goal = st.number_input(
                "Years to Goal", 
                min_value=1, 
                max_value=50, 
                value=int(user_prefs.get('years_to_goal', 5))
            )
            
            st.subheader("📈 Expected Returns")
            equity_return = st.slider(
                "Post-Tax Equity Return %", 
                min_value=0.0, 
                max_value=20.0, 
                value=float(user_prefs.get('equity_return', 10.0)), 
                step=0.1
            ) / 100
            
            debt_return = st.slider(
                "Post-Tax Debt Return %", 
                min_value=0.0, 
                max_value=15.0, 
                value=float(user_prefs.get('debt_return', 6.0)), 
                step=0.1
            ) / 100
            
            st.subheader("💰 Current Investments")
            auto_load = st.checkbox(
                "Auto-load goal investments", 
                value=user_prefs.get('auto_load', True)
            )
            
            if auto_load:
                st.metric("Current Equity", format_indian_currency(current_equity))
                st.metric("Current Debt", format_indian_currency(current_debt))
            else:
                current_equity = st.number_input(
                    "Current Equity Investments (₹)", 
                    min_value=0, 
                    value=int(user_prefs.get('current_equity', current_equity)), 
                    step=10000
                )
                current_debt = st.number_input(
                    "Current Debt Investments (₹)", 
                    min_value=0, 
                    value=int(user_prefs.get('current_debt', current_debt)), 
                    step=10000
                )
            
            st.subheader("📅 Other Parameters")
            inflation = st.slider(
                "Expected Inflation %", 
                min_value=0.0, 
                max_value=10.0, 
                value=float(user_prefs.get('inflation', 6.0)), 
                step=0.1
            ) / 100
            
            annual_increase = st.slider(
                "Expected Annual Increase in Investments %", 
                min_value=0.0, 
                max_value=20.0, 
                value=float(user_prefs.get('annual_increase', 5.0)), 
                step=0.1
            ) / 100
            
            if st.button("💾 Save Settings", key=f"save_{goal_name}"):
                user_prefs.update({
                    'current_goal_value': current_goal_value,
                    'years_to_goal': years_to_goal,
                    'equity_return': equity_return * 100,
                    'debt_return': debt_return * 100,
                    'current_equity': current_equity,
                    'current_debt': current_debt,
                    'inflation': inflation * 100,
                    'annual_increase': annual_increase * 100,
                    'auto_load': auto_load
                })
                save_user_prefs(goal_name, user_prefs)
                st.success("Settings saved!")
    
    if st.button("Calculate Goal Plan", type="primary"):
        with st.spinner(f"Calculating {goal_name} goal plan..."):
            if goal_name.lower() == "emergency fund":
                corpus_needed, monthly_investment, monthly_investments, corpus_projection = calculate_emergency_fund(
                    monthly_expenses=monthly_expenses,
                    months_needed=months_needed,
                    inflation=inflation,
                    debt_return=debt_return
                )
                
                st.session_state[f'{goal_name}_plan'] = {
                    'corpus_needed': corpus_needed,
                    'monthly_investment': monthly_investment,
                    'monthly_investments': monthly_investments,
                    'corpus_projection': corpus_projection,
                    'debt_return': debt_return,
                    'current_debt': current_debt,
                    'is_emergency_fund': True
                }
            else:
                corpus_needed, annual_investments, projection_df, monthly_investments, current_goal_value = calculate_goal_corpus(
                    current_goal_value=current_goal_value,
                    years_to_goal=years_to_goal,
                    current_equity=current_equity,
                    current_debt=current_debt,
                    equity_return=equity_return,
                    debt_return=debt_return,
                    inflation=inflation,
                    annual_increase=annual_increase
                )
                
                st.session_state[f'{goal_name}_plan'] = {
                    'corpus_needed': corpus_needed,
                    'annual_investments': annual_investments,
                    'projection_df': projection_df,
                    'monthly_investments': monthly_investments,
                    'equity_return': equity_return,
                    'debt_return': debt_return,
                    'current_goal_value': current_goal_value,
                    'is_emergency_fund': False
                }
    
    if f'{goal_name}_plan' in st.session_state:
        plan = st.session_state[f'{goal_name}_plan']
        
        if plan.get('is_emergency_fund', False):
            # Display for Emergency Fund
            current_total = plan['current_debt']
            progress_pct = min(100, (current_total / plan['corpus_needed']) * 100)
            
            st.subheader("🛡️ Emergency Fund Progress")
            col1, col2 = st.columns([1, 2])
            with col1:
                st.plotly_chart(create_speedometer(
                    current=current_total,
                    target=plan['corpus_needed'],
                    title=f"Progress: {progress_pct:.1f}%"
                ), use_container_width=True)
                
                if progress_pct < 25:
                    message = "Just starting - build your safety net!"
                elif 25 <= progress_pct < 50:
                    message = "Good progress - keep adding to your safety net!"
                elif 50 <= progress_pct < 75:
                    message = "Halfway there - you're getting more secure!"
                elif 75 <= progress_pct < 90:
                    message = "Almost there - just a bit more to go!"
                elif 90 <= progress_pct < 100:
                    message = "Nearly complete - excellent financial safety!"
                else:
                    message = "Fully funded - great job protecting yourself!"
                
                st.markdown(f"<div style='text-align: center; margin-top: -20px;'>{message}</div>", unsafe_allow_html=True)
            
            st.subheader("💰 Emergency Fund Details")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Months Covered", f"{months_needed} months")
                st.metric("Monthly Expenses", format_indian_currency(monthly_expenses))
            with col2:
                st.metric("Required Corpus", format_indian_currency(plan['corpus_needed']))
                st.metric("Current Corpus", format_indian_currency(current_total))
            with col3:
                st.metric("Monthly Investment Needed", format_indian_currency(plan['monthly_investment']))
                st.metric("Projected Return", f"{plan['debt_return']*100:.1f}%")
            
            st.subheader("📅 Monthly Investment Plan")
            st.plotly_chart(plot_cashflow_plan(
                plan['monthly_investments'],
                plan['corpus_needed'],
                debt_return=plan['debt_return']
            ), use_container_width=True)
            
            st.subheader("💡 Emergency Fund Tips")
            st.markdown("""
            - Keep your emergency fund in highly liquid instruments (savings account, liquid funds)
            - Consider splitting between:
              - Immediate access (1-2 months in savings account)
              - Short-term (remainder in liquid/debt funds)
            - Replenish immediately after any withdrawals
            - Review amount annually or when expenses change significantly
            """)
        else:
            # Display for regular goals
            current_total = current_equity + current_debt
            progress_pct = min(100, (current_total / plan['corpus_needed']) * 100)
            
            st.subheader("🎯 Goal Progress")
            col1, col2 = st.columns([1, 2])
            with col1:
                st.plotly_chart(create_speedometer(
                    current=current_total,
                    target=plan['corpus_needed'],
                    title=f"Progress: {progress_pct:.1f}%"
                ), use_container_width=True)
                
                if progress_pct < 25:
                    message = "Just starting - every journey begins with a first step!"
                elif 25 <= progress_pct < 50:
                    message = "Making good progress - keep it up!"
                elif 50 <= progress_pct < 75:
                    message = "Halfway there - stay consistent!"
                elif 75 <= progress_pct < 90:
                    message = "Almost there - final push needed!"
                elif 90 <= progress_pct < 100:
                    message = "Nearly at your goal - well done!"
                else:
                    message = "Goal achieved - congratulations!"
                
                st.markdown(f"<div style='text-align: center; margin-top: -20px;'>{message}</div>", unsafe_allow_html=True)
            
            st.subheader("📊 Key Metrics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Goal Value", format_indian_currency(plan['current_goal_value']))
                st.metric("Inflated Goal Amount", format_indian_currency(plan['corpus_needed']))
            with col2:
                st.metric("Annual Investment Needed", format_indian_currency(plan['annual_investments']['total']))
                st.metric("Monthly Investment Needed", format_indian_currency(plan['annual_investments']['total'] / 12))
            with col3:
                st.metric("Equity Allocation Start", f"{plan['projection_df']['Equity Allocation %'].iloc[0]:.1f}%")
                st.metric("Debt Allocation End", f"{plan['projection_df']['Debt Allocation %'].iloc[-1]:.1f}%")
            
            st.subheader("📉 Asset Allocation Glide Path")
            st.plotly_chart(plot_glide_path(plan['projection_df']), use_container_width=True)
            
            st.subheader("💰 Investment Plan")
            st.plotly_chart(plot_cashflow_plan(
                plan['monthly_investments'],
                plan['corpus_needed'],
                plan['equity_return'],
                plan['debt_return']
            ), use_container_width=True)
            
            st.subheader("📅 Yearly Investment Plan")
            yearly_summary = plan['monthly_investments'].groupby('Year').agg({
                'Equity Investment': 'sum',
                'Debt Investment': 'sum',
                'Total Investment': 'sum'
            }).reset_index()
            
            display_summary = yearly_summary.copy()
            display_summary['Equity Investment'] = display_summary['Equity Investment'].apply(format_indian_currency)
            display_summary['Debt Investment'] = display_summary['Debt Investment'].apply(format_indian_currency)
            display_summary['Total Investment'] = display_summary['Total Investment'].apply(format_indian_currency)
            
            st.dataframe(display_summary, hide_index=True, use_container_width=True)

def main():
    st.set_page_config(
        page_title="Other Goals Financial Planner",
        page_icon="💰",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""
    <style>
    .metric-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 25px;
        background-color: #f0f2f6;
        border-radius: 10px 10px 0px 0px;
        font-weight: bold;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
    }
    </style>
    """, unsafe_allow_html=True)
    
    if 'selected_goals' not in st.session_state:
        st.session_state['selected_goals'] = []
    if 'current_goal_index' not in st.session_state:
        st.session_state['current_goal_index'] = 0
    
    if not st.session_state['selected_goals']:
        goal_selection_tab()
    else:
        tabs = st.tabs([f"📊 {goal}" for goal in st.session_state['selected_goals']])
        
        for i, tab in enumerate(tabs):
            with tab:
                goal_planner_tab(st.session_state['selected_goals'][i])
        
        if st.button("← Back to Goal Selection"):
            st.session_state['selected_goals'] = []
            st.session_state['current_goal_index'] = 0
            st.rerun()

if __name__ == "__main__":
    main()