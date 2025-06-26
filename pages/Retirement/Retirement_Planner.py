# Complete Integrated Retirement Planner with All Features
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import psycopg
from scipy.optimize import minimize
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import io
# Add to your imports if not already present
import plotly.express as px
from fpdf import FPDF  # For PDF generation

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

# -------------------- USER PREFERENCES --------------------
def load_user_prefs():
    """Load user preferences from JSON file"""
    prefs_file = "user_prefs.json"
    if os.path.exists(prefs_file):
        with open(prefs_file, 'r') as f:
            return json.load(f)
    return {}

def save_user_prefs(prefs):
    """Save user preferences to JSON file"""
    prefs_file = "user_prefs.json"
    with open(prefs_file, 'w') as f:
        json.dump(prefs, f, indent=4)

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

def reset_database():
    """Clear all retirement planning data"""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM retirement_progress")
            cur.execute("DELETE FROM retirement_plans")
            conn.commit()
    st.success("All retirement planning data has been reset!")
    st.rerun()

def create_speedometer(current, target, title):
    """Create a speedometer-style gauge chart with values in Crores"""
    current_cr = current / 10000000
    target_cr = target / 10000000
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = current_cr,
        number = {'suffix': " Cr", 'valueformat': ".2f"},
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title},
        gauge = {
            'axis': {
                'range': [None, target_cr],
                'tickformat': ".1f",
                'ticksuffix': " Cr"
            },
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, target_cr*0.5], 'color': "lightgray"},
                {'range': [target_cr*0.5, target_cr*0.8], 'color': "gray"},
                {'range': [target_cr*0.8, target_cr], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': current_cr
            }
        }
    ))
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

# -------------------- DATABASE SETUP --------------------
def initialize_database():
    """Create required tables if they don't exist"""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            # Retirement plans table
            cur.execute("""
            CREATE TABLE IF NOT EXISTS retirement_plans (
                plan_id SERIAL PRIMARY KEY,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                plan_type VARCHAR(20) CHECK (plan_type IN ('BASELINE', 'UPDATE')),
                current_age INTEGER,
                retirement_age INTEGER,
                life_expectancy INTEGER,
                current_monthly_expenses NUMERIC,
                annual_expenses_growth NUMERIC,
                recurring_annual_expenses NUMERIC,
                post_retirement_inflation NUMERIC,
                equity_return NUMERIC,
                debt_return NUMERIC,
                corpus_needed NUMERIC,
                notes TEXT
            )
            """)
            
            # Retirement progress table
            cur.execute("""
            CREATE TABLE IF NOT EXISTS retirement_progress (
                progress_id SERIAL PRIMARY KEY,
                plan_id INTEGER REFERENCES retirement_plans(plan_id),
                check_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                current_equity NUMERIC,
                current_debt NUMERIC,
                projected_equity NUMERIC,
                projected_debt NUMERIC,
                on_track BOOLEAN,
                variance NUMERIC,
                notes TEXT
            )
            """)
            conn.commit()

# -------------------- PORTFOLIO FUNCTIONS --------------------
def get_portfolio_data():
    """Retrieve all records from portfolio_data table"""
    with get_db_connection() as conn:
        query = """
            SELECT date, scheme_name, code, transaction_type, value, units, amount 
            FROM portfolio_data 
            ORDER BY date, scheme_name
        """
        return pd.read_sql(query, conn)

def get_latest_nav():
    """Retrieve the latest NAVs from mutual_fund_nav table"""
    with get_db_connection() as conn:
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

def get_goal_mappings():
    """Retrieve goal mappings from the goals table"""
    with get_db_connection() as conn:
        query = """
            SELECT goal_name, investment_type, scheme_name, scheme_code, current_value
            FROM goals
            ORDER BY goal_name, investment_type, scheme_name
        """
        return pd.read_sql(query, conn)

def get_retirement_investments():
    """Get current investments tagged for retirement goal"""
    with get_db_connection() as conn:
        # Get equity investments for retirement
        equity_query = """
        SELECT COALESCE(SUM(pd.units * mf.value), 0) as equity_value
        FROM portfolio_data pd
        JOIN mutual_fund_nav mf ON pd.code = mf.code
        JOIN goals g ON pd.code = g.scheme_code
        WHERE g.goal_name = 'Retirement'
        AND g.investment_type = 'Equity'
        AND mf.nav = (SELECT MAX(nav) FROM mutual_fund_nav WHERE code = pd.code)
        AND pd.transaction_type IN ('invest', 'switch_in')
        """
        equity_value = pd.read_sql(equity_query, conn).iloc[0,0]
        
        # Get debt investments for retirement
        debt_query = """
        SELECT COALESCE(SUM(current_value), 0) as debt_value
        FROM goals
        WHERE goal_name = 'Retirement'
        AND investment_type = 'Debt'
        """
        debt_value = pd.read_sql(debt_query, conn).iloc[0,0]
        
        return equity_value, debt_value

# -------------------- RETIREMENT PLANNING FUNCTIONS --------------------
def calculate_retirement_corpus(
    current_age,
    retirement_age,
    current_monthly_expenses,
    annual_expenses_growth,
    recurring_annual_expenses,
    post_retirement_inflation,
    equity_return,
    debt_return,
    current_equity=0,
    current_debt=0,
    life_expectancy=90
):
    """
    Calculate retirement corpus needed and annual investment requirements.
    Returns corpus_needed, annual_investments, corpus_projection, monthly_investments
    """
    # Validate inputs
    if retirement_age <= current_age:
        st.error("Retirement age must be greater than current age")
        return None, None, None, None
    
    years_to_retirement = retirement_age - current_age
    retirement_years = int(life_expectancy - retirement_age)
    
    # Calculate current annual expenses
    current_annual_expenses = current_monthly_expenses * 12 + recurring_annual_expenses
    
    # Project expenses at retirement (growing at annual_expenses_growth)
    retirement_annual_expenses = current_annual_expenses * (
        (1 + annual_expenses_growth) ** years_to_retirement
    )
    
    # Calculate corpus needed using present value of retirement expenses
    corpus_needed = 0
    for year in range(1, retirement_years + 1):
        year_expenses = retirement_annual_expenses * ((1 + post_retirement_inflation) ** (year - 1))
        # Discount back to retirement date using conservative post-retirement returns
        post_ret_return = 0.5 * equity_return + 0.5 * debt_return  # Balanced portfolio in retirement
        corpus_needed += year_expenses / ((1 + post_ret_return) ** year)
    
    # Add 10% buffer for unexpected expenses
    corpus_needed *= 1.1
    
    # Calculate annual investments needed with glide path
    projection_data = []
    monthly_investment_data = []
    current_equity_value = current_equity
    current_debt_value = current_debt
    
    for year in range(1, years_to_retirement + 1):
        age = current_age + year
        years_remaining = years_to_retirement - year
        
        # Calculate target allocation (glide path)
        progress = year / years_to_retirement
        equity_pct = 0.3 + (0.5 / (1 + np.exp(6 * (progress - 0.6))))  # Smoother transition
        equity_pct = max(0.3, min(0.8, equity_pct))  # Clamp between 30% and 80%
        debt_pct = 1 - equity_pct
        
        # Calculate investment growth
        equity_growth = current_equity_value * equity_return
        debt_growth = current_debt_value * debt_return
        
        # Update values with growth
        current_equity_value += equity_growth
        current_debt_value += debt_growth
        total_corpus = current_equity_value + current_debt_value
        
        # Calculate required annual investment to reach target
        if year < years_to_retirement:
            # Future value needed at retirement
            fv_needed = corpus_needed - total_corpus
            
            # Calculate required annual investment (growing with income growth)
            growth_rate = min(0.05, annual_expenses_growth)  # Assume salary grows with inflation but cap at 5%
            effective_return = equity_pct * equity_return + debt_pct * debt_return
            if effective_return != growth_rate:
                annuity_factor = (((1 + effective_return) ** years_remaining - 
                                 (1 + growth_rate) ** years_remaining) / 
                                (effective_return - growth_rate))
            else:
                annuity_factor = years_remaining * (1 + effective_return) ** (years_remaining - 1)
            
            required_investment = max(0, fv_needed / annuity_factor)
        else:
            required_investment = 0
        
        # Split investment by allocation
        equity_investment = required_investment * equity_pct
        debt_investment = required_investment * debt_pct
        
        # Update corpus with new investments
        current_equity_value += equity_investment
        current_debt_value += debt_investment
        total_corpus = current_equity_value + current_debt_value
        
        # Record projection data
        projection_data.append({
            'Year': age,
            'Age': age,
            'Years to Retirement': years_remaining,
            'Equity Allocation %': equity_pct * 100,
            'Debt Allocation %': debt_pct * 100,
            'Equity Value': current_equity_value,
            'Debt Value': current_debt_value,
            'Total Corpus': total_corpus,
            'Annual Investment': required_investment,
            'Equity Investment': equity_investment,
            'Debt Investment': debt_investment,
            'Projected Expenses': current_annual_expenses * ((1 + annual_expenses_growth) ** year)
        })
        
        # Calculate monthly investments for this year
        for month in range(1, 13):
            monthly_investment_data.append({
                'Year': age,
                'Month': month,
                'Equity Investment': equity_investment / 12,
                'Debt Investment': debt_investment / 12,
                'Total Investment': (equity_investment + debt_investment) / 12
            })
    
    corpus_projection = pd.DataFrame(projection_data)
    monthly_investments = pd.DataFrame(monthly_investment_data)
    
    # Calculate average annual investments
    if not corpus_projection.empty:
        avg_annual_investment = corpus_projection['Annual Investment'].mean()
    else:
        avg_annual_investment = 0
    
    annual_investments = {
        'total': avg_annual_investment,
        'equity': avg_annual_investment * 0.8,  # Start with 80% equity allocation
        'debt': avg_annual_investment * 0.2     # Start with 20% debt allocation
    }
    
    return corpus_needed, annual_investments, corpus_projection, monthly_investments

# -------------------- RETIREMENT TRACKING FUNCTIONS --------------------
def save_retirement_plan(plan_data, plan_type="BASELINE"):
    """Save retirement plan to database"""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
            INSERT INTO retirement_plans (
                plan_type,
                current_age,
                retirement_age,
                life_expectancy,
                current_monthly_expenses,
                annual_expenses_growth,
                recurring_annual_expenses,
                post_retirement_inflation,
                equity_return,
                debt_return,
                corpus_needed,
                notes
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            ) RETURNING plan_id
            """, (
                plan_type,
                plan_data['current_age'],
                plan_data['retirement_age'],
                plan_data['life_expectancy'],
                plan_data['current_monthly_expenses'],
                plan_data['annual_expenses_growth'],
                plan_data['recurring_annual_expenses'],
                plan_data['post_retirement_inflation'],
                plan_data['equity_return'],
                plan_data['debt_return'],
                plan_data['corpus_needed'],
                plan_data.get('notes', '')
            ))
            plan_id = cur.fetchone()[0]
            conn.commit()
            return plan_id

def save_progress_update(plan_id, progress_data):
    """Save progress update against a plan"""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
            INSERT INTO retirement_progress (
                plan_id,
                current_equity,
                current_debt,
                projected_equity,
                projected_debt,
                on_track,
                variance,
                notes
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s
            )
            """, (
                plan_id,
                progress_data['current_equity'],
                progress_data['current_debt'],
                progress_data['projected_equity'],
                progress_data['projected_debt'],
                progress_data['on_track'],
                progress_data['variance'],
                progress_data.get('notes', '')
            ))
            conn.commit()

def get_retirement_plans():
    """Get all retirement plans"""
    with get_db_connection() as conn:
        return pd.read_sql("""
        SELECT * FROM retirement_plans ORDER BY created_date DESC
        """, conn)

def get_plan_progress(plan_id):
    """Get progress updates for a specific plan"""
    with get_db_connection() as conn:
        return pd.read_sql("""
        SELECT * FROM retirement_progress 
        WHERE plan_id = %s 
        ORDER BY check_date
        """, conn, params=(plan_id,))

def plot_glide_path(projection_df):
    """Plot asset allocation glide path"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(projection_df['Year'], projection_df['Equity Allocation %'], label='Equity %', color='blue')
    ax.plot(projection_df['Year'], projection_df['Debt Allocation %'], label='Debt %', color='orange')
    ax.set_title('Asset Allocation Glide Path', fontweight='bold')
    ax.set_xlabel('Year')
    ax.set_ylabel('Allocation %')
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig

def plot_progress_comparison(progress_data):
    """Create interactive progress comparison chart"""
    progress_data['Total Actual'] = progress_data['current_equity'] + progress_data['current_debt']
    progress_data['Total Projected'] = progress_data['projected_equity'] + progress_data['projected_debt']
    
    fig = px.line(
        progress_data, 
        x='check_date', 
        y=['Total Actual', 'Total Projected'],
        labels={'value': 'Amount (₹)', 'check_date': 'Date'},
        title='Actual vs Projected Retirement Corpus'
    )
    
    # Add markers and improve styling
    fig.update_traces(
        mode='lines+markers',
        line=dict(width=2),
        marker=dict(size=8)
    )
    
    # Add variance annotations
    for i, row in progress_data.iterrows():
        variance = (row['Total Actual'] - row['Total Projected']) / row['Total Projected'] * 100
        fig.add_annotation(
            x=row['check_date'],
            y=row['Total Actual'],
            text=f"{variance:.1f}%",
            showarrow=True,
            arrowhead=1
        )
    
    fig.update_layout(
        hovermode='x unified',
        legend_title_text='Corpus Type',
        yaxis_tickprefix='₹',
        yaxis_tickformat=',.2r'
    )
    
    return fig

# -------------------- CASHFLOW PLANNER FUNCTIONS --------------------
def plot_cashflow_plan(monthly_investments, actual_equity, actual_debt, equity_return, debt_return, corpus_needed):
    """Plot the cashflow plan comparing suggested vs actual investments"""
    # Calculate cumulative investments for suggested plan
    monthly_investments['Cumulative Equity Suggested'] = monthly_investments['Equity Investment'].cumsum()
    monthly_investments['Cumulative Debt Suggested'] = monthly_investments['Debt Investment'].cumsum()
    
    # Calculate actual investments
    monthly_investments['Equity Investment Actual'] = actual_equity
    monthly_investments['Debt Investment Actual'] = actual_debt
    monthly_investments['Cumulative Equity Actual'] = monthly_investments['Equity Investment Actual'].cumsum()
    monthly_investments['Cumulative Debt Actual'] = monthly_investments['Debt Investment Actual'].cumsum()
    
    # Create a date column for plotting - fixed timestamp issue
    start_date = pd.Timestamp(datetime.now().date()).replace(day=1)
    monthly_investments['Date'] = pd.date_range(
        start=start_date,
        periods=len(monthly_investments),
        freq='MS'  # Month Start frequency
    )
    
    # Rest of the function remains the same...
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add suggested investments
    fig.add_trace(
        go.Scatter(
            x=monthly_investments['Date'],
            y=monthly_investments['Equity Investment'],
            name='Suggested Equity',
            line=dict(color='blue', width=2),
            stackgroup='suggested',
            hovertemplate='%{y:,.0f}'
        ),
        secondary_y=False
    )
    
    # ... rest of the plotting code ...
    
    fig.add_trace(
        go.Scatter(
            x=monthly_investments['Date'],
            y=monthly_investments['Debt Investment'],
            name='Suggested Debt',
            line=dict(color='orange', width=2),
            stackgroup='suggested',
            hovertemplate='%{y:,.0f}'
        ),
        secondary_y=False
    )
    
    # Add actual investments
    fig.add_trace(
        go.Scatter(
            x=monthly_investments['Date'],
            y=monthly_investments['Equity Investment Actual'],
            name='Actual Equity',
            line=dict(color='blue', width=2, dash='dot'),
            stackgroup='actual',
            hovertemplate='%{y:,.0f}'
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(
            x=monthly_investments['Date'],
            y=monthly_investments['Debt Investment Actual'],
            name='Actual Debt',
            line=dict(color='orange', width=2, dash='dot'),
            stackgroup='actual',
            hovertemplate='%{y:,.0f}'
        ),
        secondary_y=False
    )
    
    # Add cumulative totals on secondary axis
    fig.add_trace(
        go.Scatter(
            x=monthly_investments['Date'],
            y=monthly_investments['Cumulative Equity Suggested'] + monthly_investments['Cumulative Debt Suggested'],
            name='Suggested Total Corpus',
            line=dict(color='green', width=3),
            hovertemplate='%{y:,.0f}'
        ),
        secondary_y=True
    )
    
    # Calculate actual corpus growth with returns
    monthly_investments['Actual Equity Value'] = 0.0
    monthly_investments['Actual Debt Value'] = 0.0
    
    for i in range(len(monthly_investments)):
        if i == 0:
            monthly_investments.loc[i, 'Actual Equity Value'] = monthly_investments.loc[i, 'Equity Investment Actual']
            monthly_investments.loc[i, 'Actual Debt Value'] = monthly_investments.loc[i, 'Debt Investment Actual']
        else:
            # Apply monthly returns to previous value and add new investment
            monthly_equity_return = (1 + equity_return) ** (1/12) - 1
            monthly_debt_return = (1 + debt_return) ** (1/12) - 1
            
            monthly_investments.loc[i, 'Actual Equity Value'] = (
                monthly_investments.loc[i-1, 'Actual Equity Value'] * (1 + monthly_equity_return) + 
                monthly_investments.loc[i, 'Equity Investment Actual']
            )
            monthly_investments.loc[i, 'Actual Debt Value'] = (
                monthly_investments.loc[i-1, 'Actual Debt Value'] * (1 + monthly_debt_return) + 
                monthly_investments.loc[i, 'Debt Investment Actual']
            )
    
    fig.add_trace(
        go.Scatter(
            x=monthly_investments['Date'],
            y=monthly_investments['Actual Equity Value'] + monthly_investments['Actual Debt Value'],
            name='Projected Actual Corpus',
            line=dict(color='red', width=3),
            hovertemplate='%{y:,.0f}'
        ),
        secondary_y=True
    )
    
    # Add corpus needed line
    fig.add_hline(
        y=corpus_needed,
        line_dash="dash",
        line_color="purple",
        annotation_text=f"Corpus Needed: {format_indian_currency(corpus_needed)}",
        annotation_position="bottom right",
        secondary_y=True
    )
    
    # Find when actual corpus meets required corpus
    actual_corpus = monthly_investments['Actual Equity Value'] + monthly_investments['Actual Debt Value']
    meets_corpus = monthly_investments[actual_corpus >= corpus_needed]
    
    if not meets_corpus.empty:
        first_meet = meets_corpus.iloc[0]
        fig.add_vline(
            x=first_meet['Date'],
            line_dash="dot",
            line_color="red",
            annotation_text=f"Target met in {first_meet['Year']}",
            annotation_position="top right"
        )
    
    # Update layout
    fig.update_layout(
        title='Monthly Investment Plan: Suggested vs Actual',
        xaxis_title='Date',
        yaxis_title='Monthly Investment (₹)',
        yaxis2_title='Cumulative Corpus (₹)',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    fig.update_yaxes(
        title_text="Monthly Investment (₹)",
        secondary_y=False,
        tickprefix="₹"
    )
    
    fig.update_yaxes(
        title_text="Cumulative Corpus (₹)",
        secondary_y=True,
        tickprefix="₹"
    )
    
    return fig, monthly_investments

def calculate_early_retirement(monthly_investments, corpus_needed, equity_return, debt_return):
    """Calculate when the actual investments will meet the corpus needed"""
    # Calculate monthly returns
    monthly_equity_return = (1 + equity_return) ** (1/12) - 1
    monthly_debt_return = (1 + debt_return) ** (1/12) - 1
    
    # Initialize tracking variables
    equity_value = 0
    debt_value = 0
    months_to_retirement = 0
    
    for _, row in monthly_investments.iterrows():
        # Apply returns to existing corpus
        equity_value *= (1 + monthly_equity_return)
        debt_value *= (1 + monthly_debt_return)
        
        # Add new investments
        equity_value += row['Equity Investment Actual']
        debt_value += row['Debt Investment Actual']
        
        months_to_retirement += 1
        
        # Check if we've reached the target
        if (equity_value + debt_value) >= corpus_needed:
            break
    
    # If we didn't reach the target in the original timeframe
    if (equity_value + debt_value) < corpus_needed:
        # Continue projecting beyond original retirement date
        last_row = monthly_investments.iloc[-1]
        actual_equity = last_row['Equity Investment Actual']
        actual_debt = last_row['Debt Investment Actual']
        
        while (equity_value + debt_value) < corpus_needed:
            # Apply returns to existing corpus
            equity_value *= (1 + monthly_equity_return)
            debt_value *= (1 + monthly_debt_return)
            
            # Add new investments (using last known actual values)
            equity_value += actual_equity
            debt_value += actual_debt
            
            months_to_retirement += 1
    
    years = months_to_retirement // 12
    months = months_to_retirement % 12
    
    return years, months, equity_value + debt_value

def get_yearly_investment_summary(monthly_investments):
    """Aggregate monthly investments to yearly summary"""
    yearly_summary = monthly_investments.groupby('Year').agg({
        'Equity Investment': 'sum',
        'Debt Investment': 'sum',
        'Total Investment': 'sum'
    }).reset_index()
    
    # Format currency values
    yearly_summary['Equity Investment'] = yearly_summary['Equity Investment'].apply(format_indian_currency)
    yearly_summary['Debt Investment'] = yearly_summary['Debt Investment'].apply(format_indian_currency)
    yearly_summary['Total Investment'] = yearly_summary['Total Investment'].apply(format_indian_currency)
    
    return yearly_summary

# -------------------- STREAMLIT UI --------------------
def retirement_planner_tab():
    st.header("🏦 Retirement Planner with Progress Tracking")
    st.markdown("""
    <style>
        .milestone-container {
            width: 100%;
            overflow-x: auto;
            white-space: nowrap;
        }
        .milestone-line {
            display: inline-block;
            min-width: 100%;
            padding-right: 20px;
        }
    </style>
    """, unsafe_allow_html=True)

    # Initialize database
    initialize_database()
    
    # Load user preferences
    user_prefs = load_user_prefs()
    
    # Check for existing plans
    existing_plans = get_retirement_plans()
    has_baseline = not existing_plans.empty and 'BASELINE' in existing_plans['plan_type'].values
    
    # Sidebar inputs with saved preferences
    with st.sidebar:
        st.subheader("🔢 Personal Details")
        current_age = st.number_input("Your Current Age", min_value=18, max_value=70, 
                                    value=int(user_prefs.get('current_age', 30)))
        retirement_age = st.number_input("Planned Retirement Age", min_value=current_age+1, max_value=80, 
                                       value=int(user_prefs.get('retirement_age', 60)))
        life_expectancy = st.number_input("Life Expectancy", min_value=retirement_age+1, max_value=120, 
                                        value=int(user_prefs.get('life_expectancy', 90)))
        
        st.subheader("💰 Expenses")
        current_monthly_expenses = st.number_input("Current Monthly Living Expenses (₹)", 
                                                min_value=10000, 
                                                value=int(user_prefs.get('current_monthly_expenses', 50000)), 
                                                step=5000)
        recurring_annual_expenses = st.number_input("Annual Recurring Expenses (₹)", 
                                                 min_value=0, 
                                                 value=int(user_prefs.get('recurring_annual_expenses', 100000)), 
                                                 step=10000)
        annual_expenses_growth = st.slider("Expected Annual Expenses Growth %", 
                                        min_value=0.0, max_value=15.0, 
                                        value=float(user_prefs.get('annual_expenses_growth', 6.0)), 
                                        step=0.1) / 100
        post_retirement_inflation = st.slider("Post-Retirement Inflation %", 
                                           min_value=0.0, max_value=10.0, 
                                           value=float(user_prefs.get('post_retirement_inflation', 5.0)), 
                                           step=0.1) / 100
        
        st.subheader("📈 Expected Returns")
        equity_return = st.slider("Post-Tax Equity Return %", 
                               min_value=0.0, max_value=20.0, 
                               value=float(user_prefs.get('equity_return', 10.0)), 
                               step=0.1) / 100
        debt_return = st.slider("Post-Tax Debt Return %", 
                             min_value=0.0, max_value=15.0, 
                             value=float(user_prefs.get('debt_return', 6.0)), 
                             step=0.1) / 100
        
        st.subheader("💵 Current Investments")
        auto_load = st.checkbox("Auto-load retirement investments", 
                              value=user_prefs.get('auto_load', True))
        
        if auto_load:
            with st.spinner("Loading retirement investments..."):
                current_equity, current_debt = get_retirement_investments()
            st.metric("Retirement Equity", format_indian_currency(current_equity))
            st.metric("Retirement Debt", format_indian_currency(current_debt))
        else:
            current_equity = st.number_input("Current Equity Investments (₹)", 
                                          min_value=0, 
                                          value=int(user_prefs.get('current_equity', 500000)), 
                                          step=10000)
            current_debt = st.number_input("Current Debt Investments (₹)", 
                                        min_value=0, 
                                        value=int(user_prefs.get('current_debt', 300000)), 
                                        step=10000)
        
        # Save preferences when any input changes
        if st.button("Save Current Settings", type="secondary"):
            user_prefs.update({
                'current_age': current_age,
                'retirement_age': retirement_age,
                'life_expectancy': life_expectancy,
                'current_monthly_expenses': current_monthly_expenses,
                'recurring_annual_expenses': recurring_annual_expenses,
                'annual_expenses_growth': annual_expenses_growth * 100,  # Convert back to percentage
                'post_retirement_inflation': post_retirement_inflation * 100,
                'equity_return': equity_return * 100,
                'debt_return': debt_return * 100,
                'auto_load': auto_load,
                'current_equity': current_equity,
                'current_debt': current_debt,
                # Add cashflow values from session state if available
                'actual_equity': st.session_state.get('actual_equity_input', 0),
                'actual_debt': st.session_state.get('actual_debt_input', 0)
            })
            save_user_prefs(user_prefs)
            st.success("Settings saved for next session!")
        
        # Reset button
        if st.button("Reset All Data", type="secondary"):
            reset_database()
    
    # Main content
    if not has_baseline:
        st.warning("No baseline retirement plan found. Create your baseline plan first.")
        if st.button("Create Baseline Retirement Plan", type="primary"):
            with st.spinner("Creating your baseline retirement plan..."):
                corpus_needed, annual_investments, projection_df, monthly_investments = calculate_retirement_corpus(
                    current_age=current_age,
                    retirement_age=retirement_age,
                    current_monthly_expenses=current_monthly_expenses,
                    annual_expenses_growth=annual_expenses_growth,
                    recurring_annual_expenses=recurring_annual_expenses,
                    post_retirement_inflation=post_retirement_inflation,
                    equity_return=equity_return,
                    debt_return=debt_return,
                    current_equity=current_equity,
                    current_debt=current_debt,
                    life_expectancy=life_expectancy
                )
                
                if corpus_needed is not None:
                    # Save baseline plan
                    plan_id = save_retirement_plan({
                        'current_age': current_age,
                        'retirement_age': retirement_age,
                        'life_expectancy': life_expectancy,
                        'current_monthly_expenses': current_monthly_expenses,
                        'annual_expenses_growth': annual_expenses_growth,
                        'recurring_annual_expenses': recurring_annual_expenses,
                        'post_retirement_inflation': post_retirement_inflation,
                        'equity_return': equity_return,
                        'debt_return': debt_return,
                        'corpus_needed': corpus_needed,
                        'notes': 'Initial baseline plan'
                    }, plan_type="BASELINE")
                    
                    # Save initial progress
                    save_progress_update(plan_id, {
                        'current_equity': current_equity,
                        'current_debt': current_debt,
                        'projected_equity': current_equity,
                        'projected_debt': current_debt,
                        'on_track': True,
                        'variance': 0,
                        'notes': 'Baseline established'
                    })
                    
                    st.success("✅ Baseline retirement plan created successfully!")
                    st.rerun()
    else:
        baseline_plan = existing_plans[existing_plans['plan_type'] == 'BASELINE'].iloc[0]
        
        # Display baseline info
        with st.expander("📋 View Baseline Plan", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Created on", baseline_plan['created_date'].strftime('%Y-%m-%d'))
                st.metric("Corpus Needed", format_indian_currency(baseline_plan['corpus_needed']))
                st.metric("Current Age", baseline_plan['current_age'])
                st.metric("Retirement Age", baseline_plan['retirement_age'])
            with col2:
                st.metric("Equity Return", f"{baseline_plan['equity_return']*100:.1f}%")
                st.metric("Debt Return", f"{baseline_plan['debt_return']*100:.1f}%")
                st.metric("Annual Expense Growth", f"{baseline_plan['annual_expenses_growth']*100:.1f}%")
                st.metric("Post-Retirement Inflation", f"{baseline_plan['post_retirement_inflation']*100:.1f}%")
        
        # Show goal achievement speedometer
        current_equity, current_debt = get_retirement_investments()
        current_total = current_equity + current_debt
        target_total = baseline_plan['corpus_needed']
        progress_pct = min(100, (current_total / target_total) * 100)
        
        st.subheader("🎯 Goal Achievement")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.plotly_chart(create_speedometer(
                current=current_total,
                target=target_total,
                title=f"Progress: {progress_pct:.1f}%"
            ), use_container_width=True)
        # Dynamic message based on progress
        if progress_pct < 25:
            message = "Small steps lead to big dreams. Start now!"
        elif 25 <= progress_pct < 50:
             message = "You're building momentum — great job!"
        elif 50 <= progress_pct < 75:
             message = "Stay consistent, you're almost there!"
        elif 75 <= progress_pct < 90:
            message = "Incredible! Just a bit more!"
        elif 90 <= progress_pct < 100:
            message = "Almost there! Final push needed."
        else:
             message = "Target crushed! You're a wealth builder!"
        
        # Get actual investments from user preferences if available
        user_prefs = load_user_prefs()
        actual_equity = user_prefs.get('actual_equity', 0)
        actual_debt = user_prefs.get('actual_debt', 0)
        has_actuals = actual_equity > 0 or actual_debt > 0

        # Calculate monthly returns
        monthly_equity_return = (1 + baseline_plan['equity_return']) ** (1/12) - 1
        monthly_debt_return = (1 + baseline_plan['debt_return']) ** (1/12) - 1

        # Next milestone calculation
        next_milestone = None
        milestones = [1.25, 1.5, 2, 3]
        for m in milestones:
            if current_total < target_total * m:
                next_milestone = target_total * m
                break

        # Calculate years to next milestone if applicable
        milestone_text = ""
        if next_milestone and current_total > 0:
            required_growth = (next_milestone / current_total) ** (1/(baseline_plan['retirement_age'] - baseline_plan['current_age']))
            years_to_milestone = int(np.log(next_milestone / current_total) / np.log(1 + required_growth))
            milestone_text = f"**Next milestone**: ₹{format_indian_currency(next_milestone)} by age {baseline_plan['current_age'] + years_to_milestone}"

        # Display messages below speedometer
        with col1:
            st.markdown(f"<div style='text-align: center; margin-top: -20px;'>{message}</div>", unsafe_allow_html=True)
            if progress_pct >= 25 and current_total > 0:
                st.markdown("---")
                st.markdown("**Projected Milestones:**")
        
            # Define milestone percentages (25%, 50%, 75%, 100%, 125%, etc.)
            milestones = [
                (0.50, "50% of target"),
                (0.75, "75% of target"),
                (1.00, "100% of target"),
                (1.25, "25% above target"),
                (1.50, "50% above target")
            ]
            # Calculate years to reach each milestone
            current_age = baseline_plan['current_age']
            retirement_age = baseline_plan['retirement_age']
            years_remaining = retirement_age - current_age

            # Scenario 1: Original plan growth rate
            original_growth_rate = (target_total / current_total) ** (1/years_remaining) - 1

            # Scenario 2: Actual investments growth rate (if available)
            if has_actuals:
                # Simulate growth with actual monthly investments
                def calculate_years_to_milestone(target_amount):
                    equity_value = current_equity
                    debt_value = current_debt
                    months = 0

                    while (equity_value + debt_value) < target_amount:
                        # Apply monthly returns
                        equity_value *= (1 + monthly_equity_return)
                        debt_value *= (1 + monthly_debt_return)

                        # Add new investments (using actual values)
                        equity_value += actual_equity
                        debt_value += actual_debt

                        months += 1
                    return months / 12
        with st.container():
            st.markdown("<div class='milestone-container'>", unsafe_allow_html=True)

        # Display milestones with both original and actual projections
            for milestone, label in milestones:
                milestone_value = target_total * milestone

                # Original plan projection
                if current_total < milestone_value:
                    original_years_needed = int(np.ceil(np.log(milestone_value / current_total) / np.log(1 + original_growth_rate)))  # Fixed extra parenthesis
                    original_age = current_age + original_years_needed
                    milestone_text = f"- ₹{milestone_value/10000000:.2f} Cr ({label}) by age {original_age} (original plan)"
                else:
                    milestone_text = f"- ₹{milestone_value/10000000:.2f} Cr ({label}) ✅ Achieved (original plan)"
        
        # Actual investments projection (if available)
                if has_actuals and current_total < milestone_value:
                    actual_years_needed = calculate_years_to_milestone(milestone_value)
                    actual_age = int(round(current_age + actual_years_needed))  # Round to nearest year

                    if actual_years_needed < original_years_needed:
                        milestone_text += f" 🚀 EARLY by {original_years_needed-actual_years_needed:.1f} years (actual by age {actual_age})"
                    elif actual_years_needed > original_years_needed:
                        milestone_text += f" ⏳ LATE by {actual_years_needed-original_years_needed:.1f} years (actual by age {actual_age})"
                    else:
                        milestone_text += f" ⏱ ON TIME (actual by age {actual_age})"
                elif has_actuals:
                        milestone_text += " ✅ Achieved (actual)"
                        
                st.markdown(f"<div class='milestone-line'>{milestone_text}</div>", unsafe_allow_html=True)

                # Then modify the st.markdown call:
                st.markdown("</div>", unsafe_allow_html=True)
        
    # Add note if actual investments are being used
            if has_actuals:
                 st.markdown("")
                 st.markdown("*Projections based on your actual monthly investments of:*")
                 st.markdown(f"- Equity: {format_indian_currency(actual_equity * 12)}/year")
                 st.markdown(f"- Debt: {format_indian_currency(actual_debt * 12)}/year")
            else:
                 st.markdown("")
                 st.markdown("*To see projections based on your actual investments, set them in the Cashflow Planner tab*")
        
        # Show glide path
        st.subheader("📉 Asset Allocation Glide Path")
        _, annual_investments, projection_df, monthly_investments = calculate_retirement_corpus(
            current_age=baseline_plan['current_age'],
            retirement_age=baseline_plan['retirement_age'],
            current_monthly_expenses=baseline_plan['current_monthly_expenses'],
            annual_expenses_growth=baseline_plan['annual_expenses_growth'],
            recurring_annual_expenses=baseline_plan['recurring_annual_expenses'],
            post_retirement_inflation=baseline_plan['post_retirement_inflation'],
            equity_return=baseline_plan['equity_return'],
            debt_return=baseline_plan['debt_return'],
            current_equity=0,
            current_debt=0,
            life_expectancy=baseline_plan['life_expectancy']
        )
        st.pyplot(plot_glide_path(projection_df))
        
        # Check progress button
        if st.button("Check Progress Against Baseline", type="primary"):
            with st.spinner("Calculating progress against baseline..."):
                # Get current values
                current_equity, current_debt = get_retirement_investments()
                
                # Calculate where we should be based on baseline projections
                baseline_progress = get_plan_progress(baseline_plan['plan_id'])
                baseline_date = baseline_plan['created_date']
                days_passed = (datetime.now() - baseline_date).days
                years_passed = days_passed / 365
                
                projected_equity = baseline_progress.iloc[0]['current_equity'] * ((1 + baseline_plan['equity_return']) ** years_passed)
                projected_debt = baseline_progress.iloc[0]['current_debt'] * ((1 + baseline_plan['debt_return']) ** years_passed)
                
                # Calculate variance
                total_current = current_equity + current_debt
                total_projected = projected_equity + projected_debt
                variance = (total_current - total_projected) / total_projected if total_projected > 0 else 0
                on_track = variance >= -0.1  # Within 10% of target
                
                # Save progress update
                save_progress_update(baseline_plan['plan_id'], {
                    'current_equity': current_equity,
                    'current_debt': current_debt,
                    'projected_equity': projected_equity,
                    'projected_debt': projected_debt,
                    'on_track': on_track,
                    'variance': variance,
                    'notes': f'Progress check after {years_passed:.1f} years'
                })
                
                # Display results
                st.subheader("📊 Progress Against Baseline")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Current Corpus", format_indian_currency(total_current),
                            delta=f"{variance*100:.1f}% vs projected")
                with col2:
                    st.metric("Projected Corpus", format_indian_currency(total_projected))
                with col3:
                    status = "✅ On Track" if on_track else "⚠️ Needs Attention"
                    st.metric("Status", status)
                
                # Plot progress comparison
                progress_data = get_plan_progress(baseline_plan['plan_id'])
                if not progress_data.empty:
                    st.plotly_chart(plot_progress_comparison(progress_data))
                
                # Detailed comparison
                with st.expander("📝 Detailed Comparison", expanded=False):
                    comparison_df = pd.DataFrame({
                        'Metric': ['Equity', 'Debt', 'Total'],
                        'Current': [current_equity, current_debt, total_current],
                        'Projected': [projected_equity, projected_debt, total_projected],
                        'Variance (₹)': [
                            current_equity - projected_equity,
                            current_debt - projected_debt,
                            total_current - total_projected
                        ],
                        'Variance (%)': [
                            (current_equity - projected_equity) / projected_equity * 100 if projected_equity > 0 else 0,
                            (current_debt - projected_debt) / projected_debt * 100 if projected_debt > 0 else 0,
                            variance * 100
                        ]
                    })
                    comparison_df['Current'] = comparison_df['Current'].apply(format_indian_currency)
                    comparison_df['Projected'] = comparison_df['Projected'].apply(format_indian_currency)
                    comparison_df['Variance (₹)'] = comparison_df['Variance (₹)'].apply(format_indian_currency)
                    comparison_df['Variance (%)'] = comparison_df['Variance (%)'].apply(lambda x: f"{x:.1f}%")
                    st.dataframe(comparison_df, hide_index=True, use_container_width=True)
                
                # Recommendations
                st.subheader("💡 Recommendations")
                if on_track:
                    st.success("You're on track with your retirement goals! Keep up the good work.")
                    st.markdown("""
                    - Continue with your current investment strategy
                    - Consider reviewing your asset allocation annually
                    - Monitor expense growth to ensure it stays within projections
                    """)
                else:
                    st.warning("You're behind your retirement goals. Consider:")
                    if variance < -0.1:
                        st.markdown("""
                        - **Increase contributions**: Boost monthly investments by 10-20%
                        - **Review allocation**: Ensure you're following the recommended equity/debt split
                        - **Reduce expenses**: Find ways to cut unnecessary spending
                        - **Additional income**: Explore side income opportunities
                        - **Extend timeline**: Consider working 1-2 more years if possible
                        """)
                    
                    # Calculate required increase to get back on track
                    years_remaining = baseline_plan['retirement_age'] - baseline_plan['current_age'] - years_passed
                    if years_remaining > 0:
                        additional_needed = total_projected - total_current
                        monthly_additional = additional_needed / (years_remaining * 12)
                        st.info(f"To get back on track, consider increasing monthly investments by **{format_indian_currency(monthly_additional)}** for the next {years_remaining:.1f} years")

def portfolio_optimizer_tab():
    st.header("📊 Portfolio Optimizer")
    
    st.subheader("Portfolio Rebalancer")
    if st.button("Analyze Current Allocation"):
        with st.spinner("Analyzing your portfolio..."):
            try:
                # First get the current retirement investments
                retirement_equity, retirement_debt = get_retirement_investments()
                total_portfolio = retirement_equity + retirement_debt
                
                if total_portfolio == 0:
                    st.warning("No retirement investments found. Please add investments tagged for retirement.")
                    return
                
                equity_pct = (retirement_equity / total_portfolio) * 100
                debt_pct = (retirement_debt / total_portfolio) * 100
                
                # Display current allocation
                st.write("### Current Allocation")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Equity Allocation", f"{equity_pct:.1f}%", 
                             help="Recommended range: 60-80% based on your age and risk profile")
                with col2:
                    st.metric("Debt Allocation", f"{debt_pct:.1f}%", 
                             help="Recommended range: 20-40% based on your age and risk profile")
                
                # Function to calculate XIRR
                def calculate_xirr(investment_type):
                    with get_db_connection() as conn:
                        # Get all transactions for specific investment type
                        query = f"""
                            SELECT pd.date, pd.amount 
                            FROM portfolio_data pd
                            JOIN goals g ON pd.code = g.scheme_code
                            WHERE g.goal_name = 'Retirement'
                            AND g.investment_type = '{investment_type}'
                            AND pd.transaction_type IN ('invest', 'switch_in')
                            ORDER BY pd.date
                        """
                        cashflows = pd.read_sql(query, conn)
                    
                    if not cashflows.empty:
                        # Add current portfolio value as negative cashflow (outflow)
                        current_value = retirement_equity if investment_type == 'Equity' else retirement_debt
                        final_cashflow = pd.DataFrame({
                            'date': [datetime.now()],
                            'amount': [-current_value]
                        })
                        cashflows = pd.concat([cashflows, final_cashflow], ignore_index=True)
                        
                        # Convert dates to ordinal numbers for calculation
                        cashflows['date_ordinal'] = cashflows['date'].apply(lambda x: x.toordinal())
                        
                        try:
                            years = [(d - cashflows['date_ordinal'].iloc[0])/365.0 
                                    for d in cashflows['date_ordinal']]
                            residual = 1.0
                            step = 0.05
                            guess = 0.1
                            epsilon = 0.0001
                            limit = 10000
                            
                            for _ in range(limit):
                                t_r = guess
                                npv = 0.0
                                for i, cf in enumerate(cashflows['amount']):
                                    npv += cf / ((1.0 + t_r)**years[i])
                                
                                if abs(npv) < epsilon:
                                    return t_r
                                
                                # Newton-Raphson method
                                npv_prime = 0.0
                                for i, cf in enumerate(cashflows['amount']):
                                    npv_prime += -years[i] * cf / ((1.0 + t_r)**(years[i]+1))
                                
                                guess = guess - npv / npv_prime
                            
                            return guess
                        except:
                            return None
                    return None
                
                # Calculate and display XIRRs
                st.write("### Portfolio Performance (XIRR)")
                col1, col2 = st.columns(2)
                
                # Equity XIRR
                equity_xirr = calculate_xirr('Equity')
                with col1:
                    if equity_xirr is not None:
                        st.metric("Equity XIRR", f"{equity_xirr*100:.1f}%", 
                                 help="Annualized return for equity investments")
                    else:
                        st.metric("Equity XIRR", "N/A", 
                                 help="Could not calculate XIRR for equity")
                
                # Debt XIRR
                debt_xirr = calculate_xirr('Debt')
                with col2:
                    if debt_xirr is not None:
                        st.metric("Debt XIRR", f"{debt_xirr*100:.1f}%", 
                                 help="Annualized return for debt investments")
                    else:
                        st.metric("Debt XIRR", "N/A", 
                                 help="Could not calculate XIRR for debt")
                
                # Combined Portfolio XIRR
                combined_xirr = calculate_xirr(None)  # For all retirement investments
                if combined_xirr is not None:
                    st.metric("Overall Portfolio XIRR", f"{combined_xirr*100:.1f}%", 
                             help="Annualized return for entire retirement portfolio")
                else:
                    st.metric("Overall Portfolio XIRR", "N/A", 
                             help="Could not calculate overall portfolio XIRR")
                
                # Get all goals
                goals = get_goal_mappings()
                
                # Show retirement goal investments
                retirement_goals = goals[goals['goal_name'] == 'Retirement']
                if not retirement_goals.empty:
                    st.write("#### Current Retirement Investments")
                    st.dataframe(retirement_goals[['investment_type', 'scheme_name', 'current_value']]
                               .assign(current_value=lambda x: x['current_value'].apply(format_indian_currency)))
                
                # [Rest of the existing code remains the same...]
                # Allocation recommendations
                st.write("### Allocation Recommendations")
                
                # Get user's current age from preferences or default
                user_prefs = load_user_prefs()
                current_age = user_prefs.get('current_age', 30)
                
                # Calculate recommended allocation based on age
                recommended_equity = max(60, 100 - current_age)
                recommended_debt = 100 - recommended_equity
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Recommended Equity", f"{recommended_equity}%")
                with col2:
                    st.metric("Recommended Debt", f"{recommended_debt}%")
                
                # Provide specific recommendations
                if equity_pct < recommended_equity - 5:
                    st.warning("**Action Needed**: Your equity allocation is lower than recommended")
                    st.markdown(f"""
                    - Consider increasing equity allocation by **{recommended_equity - equity_pct:.1f}%**
                    - Options to rebalance:
                        - Redirect new investments to equity funds
                        - Switch from debt to equity funds (up to ₹{format_indian_currency((recommended_equity/100*total_portfolio) - retirement_equity)})
                    """)
                elif equity_pct > recommended_equity + 5:
                    st.warning("**Action Needed**: Your equity allocation is higher than recommended")
                    st.markdown(f"""
                    - Consider reducing equity allocation by **{equity_pct - recommended_equity:.1f}%**
                    - Options to rebalance:
                        - Redirect new investments to debt funds
                        - Switch from equity to debt funds (up to ₹{format_indian_currency(retirement_equity - (recommended_equity/100*total_portfolio))})
                    """)
                else:
                    st.success("**Good News**: Your current allocation matches recommendations!")
                    st.markdown("""
                    - Continue with your current investment strategy
                    - Review allocation annually or when your risk profile changes
                    """)
                
                # Additional fund-level recommendations
                if not retirement_goals.empty:
                    st.write("#### Fund-Level Recommendations")
                    
                    # Check for over-concentration in any single fund
                    retirement_goals['pct_of_portfolio'] = (retirement_goals['current_value'] / total_portfolio) * 100
                    concentrated_funds = retirement_goals[retirement_goals['pct_of_portfolio'] > 20]
                    
                    if not concentrated_funds.empty:
                        st.warning("**Concentration Risk**: The following funds represent >20% of your portfolio:")
                        st.dataframe(concentrated_funds[['scheme_name', 'investment_type', 'pct_of_portfolio']]
                                   .assign(pct_of_portfolio=lambda x: x['pct_of_portfolio'].apply(lambda y: f"{y:.1f}%")))
                        st.markdown("""
                        **Recommendations**:
                        - Consider diversifying across more funds
                        - Reduce contributions to these funds temporarily
                        - Rebalance by redirecting new investments to other funds
                        """)
                    else:
                        st.success("Your investments are well-diversified across funds")
                
            except Exception as e:
                st.error(f"Error analyzing portfolio: {str(e)}")
                
def cashflow_planner_tab():
    st.header("💰 Retirement Cashflow Planner")
    
    # Check for existing baseline plan
    existing_plans = get_retirement_plans()
    has_baseline = not existing_plans.empty and 'BASELINE' in existing_plans['plan_type'].values
    
    if not has_baseline:
        st.warning("Please create a baseline retirement plan first in the Retirement Planner tab.")
        return
    
    # Load user preferences
    user_prefs = load_user_prefs()
    
    baseline_plan = existing_plans[existing_plans['plan_type'] == 'BASELINE'].iloc[0]
    
    # Get current investments
    current_equity, current_debt = get_retirement_investments()
    
    # Calculate the monthly investment plan
    corpus_needed, annual_investments, projection_df, monthly_investments = calculate_retirement_corpus(
        current_age=baseline_plan['current_age'],
        retirement_age=baseline_plan['retirement_age'],
        current_monthly_expenses=baseline_plan['current_monthly_expenses'],
        annual_expenses_growth=baseline_plan['annual_expenses_growth'],
        recurring_annual_expenses=baseline_plan['recurring_annual_expenses'],
        post_retirement_inflation=baseline_plan['post_retirement_inflation'],
        equity_return=baseline_plan['equity_return'],
        debt_return=baseline_plan['debt_return'],
        current_equity=current_equity,
        current_debt=current_debt,
        life_expectancy=baseline_plan['life_expectancy']
    )

    # User inputs for actual investments
    st.subheader("Your Actual Monthly Investments")
    col1, col2 = st.columns(2)
    with col1:
        actual_equity = st.number_input(
            "Actual Equity Investment (₹ per month)", 
            min_value=0, 
            value=int(user_prefs.get('actual_equity', int(monthly_investments['Equity Investment'].mean()))), 
            step=1000,
            key='actual_equity_input'
        )
    with col2:
        actual_debt = st.number_input(
            "Actual Debt Investment (₹ per month)", 
            min_value=0, 
            value=int(user_prefs.get('actual_debt', int(monthly_investments['Debt Investment'].mean()))), 
            step=1000,
            key='actual_debt_input'
        )

    # Add Save button for cashflow settings
    if st.button("💾 Save Investment Settings", key='save_cashflow_settings'):
        user_prefs.update({
            'actual_equity': actual_equity,
            'actual_debt': actual_debt
        })
        save_user_prefs(user_prefs)
        st.success("Monthly investment settings saved!")

    # Add actual investment columns to the DataFrame
    monthly_investments['Equity Investment Actual'] = actual_equity
    monthly_investments['Debt Investment Actual'] = actual_debt
    
    # Calculate cumulative actual investments
    monthly_investments['Cumulative Equity Actual'] = monthly_investments['Equity Investment Actual'].cumsum()
    monthly_investments['Cumulative Debt Actual'] = monthly_investments['Debt Investment Actual'].cumsum()
    
    # Create date sequence properly
    start_date = pd.Timestamp(datetime.now().date()).replace(day=1)
    monthly_investments['Date'] = pd.date_range(
        start=start_date,
        periods=len(monthly_investments),
        freq='MS'  # Month Start frequency
    )
    
    # Calculate actual corpus growth with returns
    monthly_equity_return = (1 + baseline_plan['equity_return']) ** (1/12) - 1
    monthly_debt_return = (1 + baseline_plan['debt_return']) ** (1/12) - 1
    
    monthly_investments['Actual Equity Value'] = 0.0
    monthly_investments['Actual Debt Value'] = 0.0
    
    for i in range(len(monthly_investments)):
        if i == 0:
            monthly_investments.loc[i, 'Actual Equity Value'] = monthly_investments.loc[i, 'Equity Investment Actual']
            monthly_investments.loc[i, 'Actual Debt Value'] = monthly_investments.loc[i, 'Debt Investment Actual']
        else:
            monthly_investments.loc[i, 'Actual Equity Value'] = (
                monthly_investments.loc[i-1, 'Actual Equity Value'] * (1 + monthly_equity_return) + 
                monthly_investments.loc[i, 'Equity Investment Actual']
            )
            monthly_investments.loc[i, 'Actual Debt Value'] = (
                monthly_investments.loc[i-1, 'Actual Debt Value'] * (1 + monthly_debt_return) + 
                monthly_investments.loc[i, 'Debt Investment Actual']
            )
    
    # Show yearly investment summary table
    st.subheader("📅 Yearly Investment Plan (Suggested)")
    yearly_summary = monthly_investments.groupby('Year').agg({
        'Equity Investment': 'sum',
        'Debt Investment': 'sum',
        'Total Investment': 'sum'
    }).reset_index()
    
    # Format for display
    display_summary = yearly_summary.copy()
    display_summary['Equity Investment'] = display_summary['Equity Investment'].apply(format_indian_currency)
    display_summary['Debt Investment'] = display_summary['Debt Investment'].apply(format_indian_currency)
    display_summary['Total Investment'] = display_summary['Total Investment'].apply(format_indian_currency)
    display_summary['Year'] = display_summary['Year'].astype(int)
    
    st.dataframe(display_summary, hide_index=True, use_container_width=True)
    
    # Plot the cashflow plan with log scale
    st.subheader("📈 Investment Plan: Suggested vs Actual (Log Scale)")
    
    # Create the plot with log scale
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add suggested investments (using the actual values from monthly_investments)
    fig.add_trace(
        go.Scatter(
            x=monthly_investments['Date'],
            y=monthly_investments['Equity Investment'],
            name='Suggested Equity',
            line=dict(color='blue', width=2),
            hovertemplate='₹%{y:,.0f}<extra></extra>'
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(
            x=monthly_investments['Date'],
            y=monthly_investments['Debt Investment'],
            name='Suggested Debt',
            line=dict(color='orange', width=2),
            hovertemplate='₹%{y:,.0f}<extra></extra>'
        ),
        secondary_y=False
    )
    
    # Add actual investments
    fig.add_trace(
        go.Scatter(
            x=monthly_investments['Date'],
            y=monthly_investments['Equity Investment Actual'],
            name='Actual Equity',
            line=dict(color='blue', width=2, dash='dot'),
            hovertemplate='₹%{y:,.0f}<extra></extra>'
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(
            x=monthly_investments['Date'],
            y=monthly_investments['Debt Investment Actual'],
            name='Actual Debt',
            line=dict(color='orange', width=2, dash='dot'),
            hovertemplate='₹%{y:,.0f}<extra></extra>'
        ),
        secondary_y=False
    )
    
    # Add corpus projections on secondary axis
    fig.add_trace(
        go.Scatter(
            x=monthly_investments['Date'],
            y=monthly_investments['Actual Equity Value'] + monthly_investments['Actual Debt Value'],
            name='Projected Corpus',
            line=dict(color='green', width=3),
            hovertemplate='₹%{y:,.0f}<extra></extra>'
        ),
        secondary_y=True
    )
    
    # Add corpus needed line
    fig.add_hline(
        y=baseline_plan['corpus_needed'],
        line_dash="dash",
        line_color="purple",
        annotation_text=f"Corpus Needed: {format_indian_currency(baseline_plan['corpus_needed'])}",
        annotation_position="bottom right",
        secondary_y=True
    )
    
    # Update layout with log scale
    fig.update_layout(
        title='Monthly Investment Plan: Suggested vs Actual (Log Scale)',
        xaxis_title='Date',
        yaxis_title='Monthly Investment (₹) - Log Scale',
        yaxis2_title='Cumulative Corpus (₹) - Log Scale',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        yaxis_type="log",
        yaxis2_type="log"
    )
    
    fig.update_yaxes(
        title_text="Monthly Investment (₹)",
        secondary_y=False,
        tickprefix="₹",
        type="log"
    )
    
    fig.update_yaxes(
        title_text="Cumulative Corpus (₹)",
        secondary_y=True,
        tickprefix="₹",
        type="log"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Calculate retirement timeline
    st.subheader("⏳ Retirement Timeline Projection")
    actual_corpus = monthly_investments['Actual Equity Value'] + monthly_investments['Actual Debt Value']
    meets_corpus = monthly_investments[actual_corpus >= baseline_plan['corpus_needed']]
    
    original_years = baseline_plan['retirement_age'] - baseline_plan['current_age']
    
    if not meets_corpus.empty:
        first_meet = meets_corpus.iloc[0]
        years_to_meet = first_meet['Year'] - baseline_plan['current_age']
        months_to_meet = first_meet['Month']
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Original Retirement Age", baseline_plan['retirement_age'])
            st.metric("Original Timeline", f"{original_years} years")
        
        with col2:
            if years_to_meet < original_years:
                st.metric("Projected Retirement Age", first_meet['Year'], 
                         delta=f"{original_years - years_to_meet} years early")
            else:
                st.metric("Projected Retirement Age", first_meet['Year'], 
                         delta=f"{years_to_meet - original_years} years late")
            
            st.metric("Projected Corpus", format_indian_currency(first_meet['Actual Equity Value'] + first_meet['Actual Debt Value']))
    else:
        st.warning("With current investment amounts, you won't reach your retirement corpus by the planned age.")
        
        # Calculate how much longer it would take
        last_row = monthly_investments.iloc[-1]
        remaining_corpus = baseline_plan['corpus_needed'] - (last_row['Actual Equity Value'] + last_row['Actual Debt Value'])
        additional_years = np.ceil(remaining_corpus / (12 * (actual_equity + actual_debt)))
        
        st.error(f"You would need to work approximately {int(additional_years)} more years at current investment levels.")
    
    # Show detailed monthly investments
    with st.expander("📝 View Detailed Monthly Investment Plan", expanded=False):
        display_df = monthly_investments.copy()
        display_df['Year'] = display_df['Year'].astype(int)
        display_df['Month'] = display_df['Month'].astype(int)
        display_df = display_df[['Year', 'Month', 'Equity Investment', 'Debt Investment', 'Total Investment']]
        display_df['Equity Investment'] = display_df['Equity Investment'].apply(format_indian_currency)
        display_df['Debt Investment'] = display_df['Debt Investment'].apply(format_indian_currency)
        display_df['Total Investment'] = display_df['Total Investment'].apply(format_indian_currency)
        st.dataframe(display_df, hide_index=True, use_container_width=True)

import io
import pandas as pd
import plotly.express as px
from datetime import datetime

def bucket_strategy_tab():
    st.header("🪣 Retirement Bucket Strategy with Custom Projections")
    
    # 1. Load Baseline Data
    existing_plans = get_retirement_plans()
    has_baseline = not existing_plans.empty and 'BASELINE' in existing_plans['plan_type'].values
    
    if not has_baseline:
        st.warning("No baseline plan found. Create one in Retirement Planner tab first.")
        st.image("https://i.imgur.com/JtQ8YQg.png", width=300)
        return
    
    baseline_plan = existing_plans[existing_plans['plan_type'] == 'BASELINE'].iloc[0]
    
    # 2. Setup Key Variables
    current_age = baseline_plan['current_age']
    retirement_age = baseline_plan['retirement_age']
    life_expectancy = baseline_plan['life_expectancy']
    retirement_years = int(life_expectancy - retirement_age)
    corpus_needed = baseline_plan['corpus_needed']
    current_equity, current_debt = get_retirement_investments()
    current_corpus = current_equity + current_debt
    annual_expenses = (baseline_plan['current_monthly_expenses'] * 12 + 
                     baseline_plan['recurring_annual_expenses'])

    # 3. Add "Retire Today" option
    st.subheader("🔮 Retire Today Scenario")
    retire_now = st.checkbox("Show projection if I retire today", value=False)

    if retire_now:
        # Use current corpus instead of projected corpus_needed
        corpus_to_use = current_corpus
        starting_age = current_age
        years_to_project = int(life_expectancy - current_age)
        
        st.warning(f"""
        Projecting retirement starting today (age {current_age}) with:
        - Current corpus: {format_indian_currency(current_corpus)}
        - Annual expenses: {format_indian_currency(annual_expenses)}
        """)
    else:
        corpus_to_use = corpus_needed
        starting_age = retirement_age
        years_to_project = retirement_years

    # 3. Bucket Configuration UI
    st.subheader("⚙️ Projection Parameters")
    
    with st.expander("Adjust Assumptions", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            b1_return = st.slider("Bucket 1 Return%", 0.0, 10.0, 4.0, step=0.1)/100
            b2_return = st.slider("Bucket 2 Return%", 0.0, 12.0, 6.0, step=0.1)/100
        with col2:
            b3_return = st.slider("Bucket 3 Return%", 0.0, 15.0, 8.0, step=0.1)/100
            inflation = st.slider("Inflation%", 0.0, 10.0, 5.0, step=0.1)/100
        with col3:
            bucket1_pct = st.slider("Bucket 1%", 20, 40, 30)
            bucket2_pct = st.slider("Bucket 2%", 30, 50, 40)
            bucket3_pct = 100 - bucket1_pct - bucket2_pct
            st.metric("Bucket 3%", f"{bucket3_pct}%")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        bucket1_pct = st.slider("0-10 Years %", 20, 40, 30)
    with col2:
        bucket2_pct = st.slider("10-20 Years %", 30, 50, 40)
    with col3:
        bucket3_pct = 100 - bucket1_pct - bucket2_pct
        st.metric("20-30 Years %", f"{bucket3_pct}%")

    # 4. Calculate Projections with custom parameters
    longevity_df = calculate_corpus_longevity(
        total_corpus=corpus_to_use,
        annual_spending=annual_expenses,
        inflation=inflation,
        b1_pct=bucket1_pct,
        b2_pct=bucket2_pct,
        b3_pct=bucket3_pct,
        years=years_to_project,
        retire_age=starting_age,
        b1_return=b1_return,
        b2_return=b2_return,
        b3_return=b3_return
    )
    
    # 5. Display Results
    depletion_row = longevity_df[longevity_df['Total Corpus'] <= 0].iloc[0] if not longevity_df[longevity_df['Total Corpus'] <= 0].empty else longevity_df.iloc[-1]
    depletion_age = depletion_row['Age']
    years_funded = depletion_row['Year']

    st.subheader("📊 Custom Projection Results")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Depletion Age", f"Age {int(depletion_age)}")
        st.metric("Years Funded", f"{int(years_funded)} years")
    with col2:
        st.metric("Final Annual Spending", 
                 format_indian_currency(annual_expenses * (1 + inflation)**years_funded))
        st.metric("Average Return", 
                 f"{(b1_return*bucket1_pct + b2_return*bucket2_pct + b3_return*bucket3_pct):.1%}")
    
    # 6. Visualization
    fig = plot_corpus_longevity(longevity_df, depletion_age)
    st.plotly_chart(fig, use_container_width=True)

    # Extend projection to at least depletion age (minimum 10 years beyond retirement)
    extended_years = max(10, years_funded + 5)  # Show at least 5 years past depletion
    if len(longevity_df) < extended_years:
        longevity_df = calculate_corpus_longevity(
            corpus_needed,
            annual_expenses,
            inflation,
            bucket1_pct,
            bucket2_pct,
            bucket3_pct,
            extended_years,
            retirement_age
        )

    # 5. Display Results
    st.subheader("📈 Corpus Longevity Projection")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Projected Depletion Age", f"Age {depletion_age}")
    with col2:
        st.metric("Years Funded", f"{years_funded} years")

    # 6. Visualization
    fig = plot_corpus_longevity(longevity_df, depletion_age)
    st.plotly_chart(fig, use_container_width=True)
    
    # 7. Improvement Tips
    show_improvement_tips(current_equity, current_debt, bucket1_pct, bucket3_pct, current_age)
    
    # 8. Stress Testing
    with st.expander("🧪 Stress Test Scenarios", expanded=False):
        run_stress_tests(
            longevity_df.copy(), 
            annual_expenses,
            inflation,  # This is the user-adjusted inflation from the slider
            b1_return,  # Pass the bucket returns too
            b2_return,
            b3_return
        )

    # 9. Export Options
    st.download_button(
        label="📄 Download Projection Report",
        data=generate_report(longevity_df, bucket1_pct, bucket2_pct, bucket3_pct),
        file_name="bucket_strategy_report.csv",
        mime="text/csv"
    )

def calculate_corpus_longevity(total_corpus, annual_spending, inflation, 
                             b1_pct, b2_pct, b3_pct, years, retire_age,
                             b1_return=0.04, b2_return=0.06, b3_return=0.08):
    projection = []
    remaining_corpus = total_corpus
    
    # Convert bucket percentages to actual amounts
    bucket1_amount = total_corpus * (b1_pct/100)
    bucket2_amount = total_corpus * (b2_pct/100)
    bucket3_amount = total_corpus * (b3_pct/100)
    
    for year in range(1, int(years) + 1):
        current_age = retire_age + year
        yearly_spending = annual_spending * (1 + inflation) ** (year - 1)
        
        # Withdraw from buckets in order (1 → 2 → 3)
        withdrawn = 0
        remaining_withdrawal = yearly_spending
        
        # Bucket 1 (Cash/Liquid)
        if bucket1_amount > 0:
            bucket1_amount *= (1 + b1_return)  # Apply custom return
            withdraw_from_b1 = min(bucket1_amount, remaining_withdrawal)
            bucket1_amount -= withdraw_from_b1
            remaining_withdrawal -= withdraw_from_b1
            withdrawn += withdraw_from_b1
        
        # Bucket 2 (Income)
        if remaining_withdrawal > 0 and bucket2_amount > 0:
            bucket2_amount *= (1 + b2_return)
            withdraw_from_b2 = min(bucket2_amount, remaining_withdrawal)
            bucket2_amount -= withdraw_from_b2
            remaining_withdrawal -= withdraw_from_b2
            withdrawn += withdraw_from_b2
        
        # Bucket 3 (Growth)
        if remaining_withdrawal > 0 and bucket3_amount > 0:
            bucket3_amount *= (1 + b3_return)
            withdraw_from_b3 = min(bucket3_amount, remaining_withdrawal)
            bucket3_amount -= withdraw_from_b3
            remaining_withdrawal -= withdraw_from_b3
            withdrawn += withdraw_from_b3
        
        # Calculate total remaining corpus
        remaining_corpus = bucket1_amount + bucket2_amount + bucket3_amount
        
        projection.append({
            "Year": year,
            "Age": current_age,
            "Bucket1": bucket1_amount,
            "Bucket2": bucket2_amount,
            "Bucket3": bucket3_amount,
            "Total Corpus": remaining_corpus,
            "Withdrawal": yearly_spending,
            "Actual Withdrawn": withdrawn,
            "Shortfall": max(0, yearly_spending - withdrawn),
            "Return Rate": f"{(b1_return*b1_pct + b2_return*b2_pct + b3_return*b3_pct)/100:.1%}"
        })
        
        if remaining_corpus <= 0:
            break
    
    return pd.DataFrame(projection)

def plot_corpus_longevity(df, depletion_age=None):
    fig = go.Figure()
    
    # FIX: Using the correct bucket column names
    buckets = {
        'Bucket1': {'name': '0-10 Years (Cash)', 'color': '#636EFA'},
        'Bucket2': {'name': '10-20 Years (Income)', 'color': '#EF553B'}, 
        'Bucket3': {'name': '20+ Years (Growth)', 'color': '#00CC96'}
    }
    
    for bucket, style in buckets.items():
        fig.add_trace(go.Scatter(
            x=df['Age'],
            y=df[bucket],
            name=style['name'],
            stackgroup='one',
            line=dict(width=0.5, color=style['color']),
            fillcolor=style['color'],
            hovertemplate="Age: %{x}<br>Value: ₹%{y:,.0f}<br>Bucket: "+style['name'],
            fill='tonexty'
        ))
    
    # FIX: Using 'Total Corpus' instead of 'Corpus'
    fig.add_trace(go.Scatter(
        x=df['Age'],
        y=df['Total Corpus'],
        name='Total Portfolio',
        line=dict(color='#2A3F54', width=3),
        hovertemplate="Age: %{x}<br>Total: ₹%{y:,.0f}"
    ))
    
    # Add withdrawal line
    fig.add_trace(go.Scatter(
        x=df['Age'],
        y=df['Withdrawal'],
        name='Annual Spending',
        line=dict(color='red', width=2, dash='dot'),
        hovertemplate="Age: %{x}<br>Spending: ₹%{y:,.0f}"
    ))
    
    # Add depletion marker (using either provided depletion_age or calculated)
    final_depletion_age = depletion_age if depletion_age else df[df['Total Corpus'] <= 0]['Age'].min()
    if not pd.isna(final_depletion_age):
        fig.add_vline(
            x=final_depletion_age,
            line=dict(color='black', width=2, dash='dash'),
            annotation_text=f"Depletion at age {int(final_depletion_age)}",
            annotation_position="top right",
            annotation_font_size=12,
            annotation_font_color="red"
        )
    
    # Add shortfall markers if any
    if 'Shortfall' in df.columns and (df['Shortfall'] > 0).any():
        shortfall_years = df[df['Shortfall'] > 0]
        fig.add_trace(go.Scatter(
            x=shortfall_years['Age'],
            y=shortfall_years['Withdrawal'],
            mode='markers',
            marker=dict(color='red', size=8, symbol='x'),
            name='Withdrawal Shortfall',
            hovertemplate="Age: %{x}<br>Shortfall: ₹%{y:,.0f}"
        ))
    
    fig.update_layout(
        title="Retirement Bucket Strategy Projection",
        xaxis_title="Age",
        yaxis_title="Amount (₹)",
        hovermode="x unified",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(tickprefix="₹", gridcolor='lightgray'),
        xaxis=dict(gridcolor='lightgray'),
        plot_bgcolor='white'
    )
    
    return fig

def show_improvement_tips(current_equity, current_debt, b1_pct, b3_pct, current_age):
    with st.expander("💡 How to Improve Portfolio Longevity", expanded=True):
        st.markdown(f"""
        **1. Dynamic Withdrawals**  
        → Reduce spending by 10% when portfolio drops 15%  
        → Example: ₹12L → ₹10.8L in bad years (+3-5 years)  

        **2. Tax Optimization**  
        → Roth conversions when income < ₹7L taxable  
        → Harvest losses to offset capital gains  

        **3. Growth Boost**  
        → Current growth allocation: **{b3_pct}%**  
        → Ideal range: 30-50% for age {current_age}  
        → Consider increasing by {max(0, 40-b3_pct)}%  

        **4. Annuity Ladder**  
        → Use 10-20% of Bucket 1 to buy deferred annuities  
        → Guarantees income from age 75+  

        **5. Expense Flooring**  
        → Cover basics with SWP from debt funds (₹X/month)  
        → Keep discretionary spending flexible  
        """)

def run_stress_tests(base_df, annual_spending, base_inflation, b1_return, b2_return, b3_return):
    col1, col2 = st.columns(2)
    with col1:
        crash_test = st.checkbox("Simulate Market Crash (First 5 Years)", value=False)
        crash_severity = st.slider("Crash Severity (%)", 10, 50, 30, disabled=not crash_test)/100
    
    with col2:
        inflation_test = st.checkbox("Simulate High Inflation", value=False)
        stress_inflation = st.slider("Stress Inflation (%)", 
                                   max(1.0, base_inflation*100), 15.0, 7.0, 
                                   disabled=not inflation_test)/100
    
    if crash_test or inflation_test:
        test_df = base_df.copy()
        
        # Apply stress scenarios
        for i, row in test_df.iterrows():
            # Market crash impact
            if crash_test and row['Year'] <= 5:
                for bucket in ['Bucket1', 'Bucket2', 'Bucket3']:
                    test_df.at[i, bucket] *= (1 - crash_severity)  # Apply crash impact
            
            # Inflation impact - use either base or stress inflation
            current_inflation = stress_inflation if inflation_test else base_inflation
            test_df.at[i, 'Withdrawal'] = annual_spending * (1 + current_inflation) ** (row['Year'] - 1)
        
        # Recalculate with proper depletion logic
        for i in range(1, len(test_df)):
            # Apply returns to each bucket first
            test_df.at[i, 'Bucket1'] = test_df.at[i-1, 'Bucket1'] * (1 + b1_return)
            test_df.at[i, 'Bucket2'] = test_df.at[i-1, 'Bucket2'] * (1 + b2_return)
            test_df.at[i, 'Bucket3'] = test_df.at[i-1, 'Bucket3'] * (1 + b3_return)
            
            # Withdraw from buckets in order
            remaining_withdrawal = test_df.at[i, 'Withdrawal']
            for bucket in ['Bucket1', 'Bucket2', 'Bucket3']:
                if remaining_withdrawal > 0 and test_df.at[i, bucket] > 0:
                    withdraw_amount = min(test_df.at[i, bucket], remaining_withdrawal)
                    test_df.at[i, bucket] -= withdraw_amount
                    remaining_withdrawal -= withdraw_amount
            
            # Update total and track shortfalls
            test_df.at[i, 'Total Corpus'] = test_df.at[i, 'Bucket1'] + test_df.at[i, 'Bucket2'] + test_df.at[i, 'Bucket3']
            test_df.at[i, 'Shortfall'] = max(0, remaining_withdrawal)
            
            if test_df.at[i, 'Total Corpus'] <= 0:
                break
        
        # Visualize results
        depletion_age = test_df[test_df['Total Corpus'] <= 0]['Age'].min()
        fig = plot_corpus_longevity(test_df, depletion_age)
        
        st.warning(f"""
        Stress Test Results:
        - Projected depletion age: {int(depletion_age) if not pd.isna(depletion_age) else 'Never'}
        - Worst annual shortfall: {format_indian_currency(test_df['Shortfall'].max())}
        """)
        st.plotly_chart(fig, use_container_width=True)

def generate_report(df, b1, b2, b3):
    output = io.StringIO()
    df.to_csv(output, index=False)
    return output.getvalue()

def format_indian_currency(amount):
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

def get_retirement_plans():
    with get_db_connection() as conn:
        return pd.read_sql("SELECT * FROM retirement_plans ORDER BY created_date DESC", conn)

def get_retirement_investments():
    with get_db_connection() as conn:
        # Get equity investments for retirement
        equity_query = """
        SELECT COALESCE(SUM(pd.units * mf.value), 0) as equity_value
        FROM portfolio_data pd
        JOIN mutual_fund_nav mf ON pd.code = mf.code
        JOIN goals g ON pd.code = g.scheme_code
        WHERE g.goal_name = 'Retirement'
        AND g.investment_type = 'Equity'
        AND mf.nav = (SELECT MAX(nav) FROM mutual_fund_nav WHERE code = pd.code)
        AND pd.transaction_type IN ('invest', 'switch_in')
        """
        equity_value = pd.read_sql(equity_query, conn).iloc[0,0]
        
        # Get debt investments for retirement
        debt_query = """
        SELECT COALESCE(SUM(current_value), 0) as debt_value
        FROM goals
        WHERE goal_name = 'Retirement'
        AND investment_type = 'Debt'
        """
        debt_value = pd.read_sql(debt_query, conn).iloc[0,0]
        
        return equity_value, debt_value

def get_db_connection():
    return psycopg.connect(**DB_PARAMS)     
# -------------------- MAIN APP --------------------
def main():
    st.set_page_config(
        page_title="Integrated Financial Planner",
        page_icon="💰",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
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
    
    # Create tabs
    tab1, tab2, tab3,tab4 = st.tabs(["🏦 Retirement Planner", "💰 Cashflow Planner", "📊 Portfolio Optimizer","🪣 Bucket Strategy"])
    
    with tab1:
        retirement_planner_tab()
    
    with tab2:
        cashflow_planner_tab()
    
    with tab3:
        portfolio_optimizer_tab()
    with tab4:  # New tab
        bucket_strategy_tab()  # New function

if __name__ == "__main__":
    main()