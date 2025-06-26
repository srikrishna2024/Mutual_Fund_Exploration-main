import streamlit as st
import psycopg
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
import locale
import random

# Set locale for Indian number formatting
locale.setlocale(locale.LC_ALL, 'en_IN')

# Database configuration
DB_PARAMS = {
    'dbname': 'postgres',
    'user': 'postgres',
    'password': 'admin123',
    'host': 'localhost',
    'port': '5432'
}

# Base expected returns for equity and debt
EQUITY_BASE_RETURN = 0.12  # 12%
DEBT_BASE_RETURN = 0.07    # 7%

# Volatility parameters for random returns
EQUITY_VOLATILITY = 0.15    # 15% standard deviation
DEBT_VOLATILITY = 0.05      # 5% standard deviation

def get_db_connection():
    """Establish connection to the PostgreSQL database"""
    return psycopg.connect(**DB_PARAMS)

def format_indian_currency(amount):
    """Format amount in Indian currency style (lakhs, crores)"""
    def format_number(num):
        if num < 0:
            return f"-{format_number(abs(num))}"
        if num < 1000:
            return str(num)
        elif num < 100000:
            return f"{num/1000:.2f}K"
        elif num < 10000000:
            return f"{num/100000:.2f}L"
        else:
            return f"{num/10000000:.2f}Cr"
    
    return f"{format_number(float(amount))}"

def format_indian(number):
    """Format number with Indian comma separators"""
    try:
        return locale.format_string("%.2f", number, grouping=True)
    except:
        return str(number)

def get_random_return(base_return, volatility):
    """Generate a random return with normal distribution"""
    return random.gauss(base_return, volatility)

def simulate_portfolio_growth(
    current_value, 
    duration_years, 
    growth_rates, 
    goal_target=None,
    monthly_contribution=0,
    use_random_returns=False
):
    """
    Simulate portfolio growth under different growth rate scenarios.
    
    Args:
        current_value: Current portfolio value
        duration_years: Duration in years for the simulation
        growth_rates: Dictionary of scenario names and their annual growth rates
        goal_target: Optional target amount for the goal
        monthly_contribution: Monthly contribution amount
        use_random_returns: Boolean to toggle random returns
        
    Returns:
        DataFrame with simulation results
    """
    results = pd.DataFrame()
    
    # Generate timeline (monthly intervals)
    months = duration_years * 12
    timeline = [datetime.now() + timedelta(days=30*i) for i in range(months+1)]
    results['date'] = timeline
    
    # Simulate each growth scenario
    for scenario, annual_rate in growth_rates.items():
        values = [current_value]
        annual_rates = []
        
        # Generate random returns for each year if enabled
        if use_random_returns:
            for _ in range(duration_years):
                # Get equity and debt components of the return
                equity_part = annual_rate * (EQUITY_BASE_RETURN / (EQUITY_BASE_RETURN + DEBT_BASE_RETURN))
                debt_part = annual_rate * (DEBT_BASE_RETURN / (EQUITY_BASE_RETURN + DEBT_BASE_RETURN))
                
                # Apply random variation to each component
                equity_return = get_random_return(equity_part, EQUITY_VOLATILITY)
                debt_return = get_random_return(debt_part, DEBT_VOLATILITY)
                
                # Combine components
                annual_rates.append(equity_return + debt_return)
        
        for i in range(months):
            if use_random_returns:
                # Use the annual rate for all months in that year
                year_idx = i // 12
                monthly_rate = (1 + annual_rates[year_idx])**(1/12) - 1
            else:
                monthly_rate = (1 + annual_rate)**(1/12) - 1
                
            next_value = values[-1] * (1 + monthly_rate) + monthly_contribution
            values.append(next_value)
            
        results[scenario] = values
    
    # Add goal target line if provided
    if goal_target is not None and goal_target > 0:
        results['goal_target'] = goal_target
        
    return results

def check_if_columns_exist():
    """Check if equity_allocation and debt_allocation columns exist in goals table"""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            # Check if equity_allocation column exists
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.columns 
                    WHERE table_name = 'goals' AND column_name = 'equity_allocation'
                );
            """)
            has_equity_allocation = cur.fetchone()[0]
            
            # Check if debt_allocation column exists
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.columns 
                    WHERE table_name = 'goals' AND column_name = 'debt_allocation'
                );
            """)
            has_debt_allocation = cur.fetchone()[0]
            
            return has_equity_allocation and has_debt_allocation

def get_goal_data():
    """Retrieve goal data with current values and asset allocation"""
    with get_db_connection() as conn:
        # Check if the required columns exist
        has_allocation_columns = check_if_columns_exist()
        
        if has_allocation_columns:
            # Use query with allocation columns
            query = """
                SELECT 
                    goal_name,
                    SUM(current_value) as total_value,
                    AVG(equity_allocation) as equity_allocation,
                    AVG(debt_allocation) as debt_allocation
                FROM goals
                GROUP BY goal_name
                ORDER BY goal_name
            """
        else:
            # Use simpler query without allocation columns
            query = """
                SELECT 
                    goal_name,
                    SUM(current_value) as total_value
                FROM goals
                GROUP BY goal_name
                ORDER BY goal_name
            """
        
        df = pd.read_sql(query, conn)
        
        # Add default allocations if columns don't exist
        if not has_allocation_columns and not df.empty:
            df['equity_allocation'] = 0.60
            df['debt_allocation'] = 0.40
            
        return df

def calculate_weighted_xirr(equity_allocation, debt_allocation):
    """Calculate weighted XIRR based on asset allocation"""
    return (equity_allocation * EQUITY_BASE_RETURN) + (debt_allocation * DEBT_BASE_RETURN)

def get_portfolio_xirr_for_goal(goal_name=None):
    """
    Calculate the XIRR for a specific goal or the entire portfolio
    based on equity/debt allocation
    """
    # Check if allocation columns exist
    has_allocation_columns = check_if_columns_exist()
    
    if not has_allocation_columns:
        # Use default allocations
        return calculate_weighted_xirr(0.60, 0.40)
    
    try:
        with get_db_connection() as conn:
            # If goal_name provided, get the asset allocation for that goal
            if goal_name:
                query = """
                    SELECT 
                        AVG(equity_allocation) as equity_allocation,
                        AVG(debt_allocation) as debt_allocation
                    FROM goals
                    WHERE goal_name = %s
                """
                df = pd.read_sql(query, conn, params=(goal_name,))
                
                if not df.empty:
                    equity_allocation = df['equity_allocation'].iloc[0]
                    debt_allocation = df['debt_allocation'].iloc[0]
                else:
                    # Default allocations if no data found
                    equity_allocation = 0.60
                    debt_allocation = 0.40
            else:
                # Get overall portfolio allocation
                query = """
                    SELECT 
                        AVG(equity_allocation) as equity_allocation,
                        AVG(debt_allocation) as debt_allocation
                    FROM goals
                """
                df = pd.read_sql(query, conn)
                
                if not df.empty:
                    equity_allocation = df['equity_allocation'].iloc[0]
                    debt_allocation = df['debt_allocation'].iloc[0]
                else:
                    # Default allocations if no data found
                    equity_allocation = 0.60
                    debt_allocation = 0.40
            
            # Calculate weighted XIRR
            return calculate_weighted_xirr(equity_allocation, debt_allocation)
    except Exception as e:
        st.warning(f"Error calculating XIRR: {str(e)}. Using default weighted XIRR.")
        # Default to 60/40 allocation
        return calculate_weighted_xirr(0.60, 0.40)

def recommend_asset_allocation(current_xirr, target_xirr):
    """
    Recommend asset allocation to achieve target XIRR
    
    Args:
        current_xirr: Current XIRR
        target_xirr: Target XIRR to achieve
        
    Returns:
        Tuple of (equity_allocation, debt_allocation)
    """
    # If target XIRR is outside possible range, cap it
    if target_xirr > EQUITY_BASE_RETURN:
        return (1.0, 0.0)  # 100% equity (maximum possible return)
    elif target_xirr < DEBT_BASE_RETURN:
        return (0.0, 1.0)  # 100% debt (minimum possible return)
    
    # Calculate required equity allocation using weighted average formula
    # target_xirr = equity_alloc * EQUITY_RETURN + (1-equity_alloc) * DEBT_RETURN
    # Solve for equity_alloc
    equity_allocation = (target_xirr - DEBT_BASE_RETURN) / (EQUITY_BASE_RETURN - DEBT_BASE_RETURN)
    debt_allocation = 1.0 - equity_allocation
    
    return (equity_allocation, debt_allocation)

def calculate_recovery_allocation(crash_scenario, base_scenario, current_equity, current_debt, duration_years):
    """
    Calculate recommended allocation to recover from a crash scenario
    
    Args:
        crash_scenario: The expected return after crash (negative value)
        base_scenario: The base case expected return
        current_equity: Current equity allocation (0-1)
        current_debt: Current debt allocation (0-1)
        duration_years: Number of years for recovery
        
    Returns:
        Tuple of (new_equity_allocation, new_debt_allocation, recovery_xirr)
    """
    # Calculate the target XIRR needed to recover from the crash
    # Formula: (1 + recovery_xirr)^duration = (1 + base)^duration / (1 + crash)
    
    # First calculate what the final multiplier would be in each scenario
    base_multiplier = (1 + base_scenario) ** duration_years
    crash_multiplier = (1 + crash_scenario) ** duration_years
    
    # Calculate the multiplier needed to recover
    recovery_multiplier = base_multiplier / crash_multiplier
    
    # Convert to annual rate
    recovery_xirr = recovery_multiplier ** (1/duration_years) - 1
    
    # Cap at reasonable limits
    recovery_xirr = min(recovery_xirr, 0.20)  # Cap at 20% annual return
    
    # Calculate allocation needed for this XIRR
    new_equity, new_debt = recommend_asset_allocation(0, recovery_xirr)
    
    return (new_equity, new_debt, recovery_xirr)

def run_crash_simulator():
    st.set_page_config(page_title="Portfolio Crash Simulator", layout="wide")
    st.title("Portfolio Crash Simulator")
    
    # Get goal data
    goals_df = get_goal_data()
    
    st.write("Simulate how market fluctuations might impact your portfolio growth and goal achievement.")
    
    # Add return type selection
    return_type = st.radio(
        "Return Type",
        ["Fixed Returns", "Random Returns (More Realistic)"],
        index=0,
        help="Fixed uses constant returns each year. Random adds realistic variability."
    )
    use_random_returns = return_type == "Random Returns (More Realistic)"
    
    if use_random_returns:
        st.info("Using random returns - each simulation will show different paths. Results will vary each time you run it.")
    
    # Session state for selected goal and asset allocation
    if 'selected_goal' not in st.session_state:
        st.session_state.selected_goal = None
        st.session_state.equity_allocation = 0.60
        st.session_state.debt_allocation = 0.40
        st.session_state.current_xirr = calculate_weighted_xirr(0.60, 0.40)
    
    with st.form("crash_simulator_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            # Goal selection with current values
            if not goals_df.empty:
                goal_options = [f"{row['goal_name']} (₹{format_indian_currency(row['total_value'])})" 
                               for _, row in goals_df.iterrows()]
                selected_goal = st.selectbox("Select Goal", goal_options)
                
                # Extract goal name and value
                if selected_goal:
                    goal_name = selected_goal.split(" (₹")[0]
                    goal_row = goals_df[goals_df['goal_name'] == goal_name].iloc[0]
                    current_value = goal_row['total_value']
                    
                    # Get goal-specific asset allocation from the dataframe
                    equity_allocation = goal_row['equity_allocation']
                    debt_allocation = goal_row['debt_allocation']
                    
                    # Update session state
                    st.session_state.selected_goal = goal_name
                    st.session_state.equity_allocation = equity_allocation
                    st.session_state.debt_allocation = debt_allocation
                    
                    # Calculate XIRR based on goal's asset allocation
                    st.session_state.current_xirr = calculate_weighted_xirr(equity_allocation, debt_allocation)
                else:
                    current_value = 0
            else:
                st.warning("No goals found. Please map investments to goals first.")
                current_value = st.number_input("Current Portfolio Value", min_value=0.0, step=1000.0)
                
                # Manual asset allocation for users without goals
                equity_allocation = st.slider("Equity Allocation (%)", 0, 100, 60) / 100
                debt_allocation = 1.0 - equity_allocation
                
                st.session_state.equity_allocation = equity_allocation
                st.session_state.debt_allocation = debt_allocation
                st.session_state.current_xirr = calculate_weighted_xirr(equity_allocation, debt_allocation)
        
        with col2:
            duration_years = st.slider("Projection Duration (Years)", 1, 30, 10)
            target_amount = st.number_input(
                "Target Amount (optional)",
                min_value=0.0,
                value=float(current_value * 2) if current_value > 0 else 1000000.0,
                help="Leave at 0 if no specific target amount"
            )
            
            # Add monthly contribution option
            monthly_contribution = st.number_input(
                "Monthly Contribution (₹)",
                min_value=0.0,
                step=1000.0,
                value=0.0
            )
        
        # Asset allocation section
        st.subheader("Current Asset Allocation")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Show current asset allocation
            equity_pct = st.session_state.equity_allocation * 100
            debt_pct = st.session_state.debt_allocation * 100
            
            st.metric("Equity Allocation", f"{equity_pct:.1f}%")
        
        with col2:
            st.metric("Debt Allocation", f"{debt_pct:.1f}%")
        
        with col3:
            # Display current XIRR based on allocation
            st.metric("Current Estimated XIRR", f"{st.session_state.current_xirr:.2%}")
        
        # Define scenarios based on current XIRR
        current_xirr = st.session_state.current_xirr
        scenarios = {
            f"Base Case ({current_xirr:.1%})": current_xirr,
            "Crash (-30%)": max(current_xirr - 0.30, -0.25),  # Ensure it doesn't go below -25%
            "Downturn (-20%)": max(current_xirr - 0.20, -0.25),
            "Slow (-10%)": max(current_xirr - 0.10, -0.25),
            "Optimistic (+10%)": min(current_xirr + 0.10, 0.25),  # Ensure it doesn't go above 25%
            "Bull Run (+20%)": min(current_xirr + 0.20, 0.25),
            "Exceptional (+30%)": min(current_xirr + 0.30, 0.25)
        }
        
        recovery_period = st.slider("Recovery Period (Years)", 1, 10, 3, 
                                   help="Period to recover from a market crash")
        
        submitted = st.form_submit_button("Simulate Portfolio Growth")
        
        if submitted and current_value > 0:
            # Run simulation
            results = simulate_portfolio_growth(
                current_value,
                duration_years,
                scenarios,
                target_amount if target_amount > 0 else None,
                monthly_contribution,
                use_random_returns
            )
            
            # Create plot
            fig = go.Figure()
            
            # Add line for each scenario
            colors = px.colors.qualitative.Plotly
            for i, scenario in enumerate(scenarios.keys()):
                fig.add_trace(go.Scatter(
                    x=results['date'],
                    y=results[scenario],
                    mode='lines',
                    name=scenario,
                    line=dict(color=colors[i % len(colors)], width=2),
                ))
            
            # Add target line if provided
            if target_amount > 0:
                fig.add_trace(go.Scatter(
                    x=results['date'],
                    y=results['goal_target'],
                    mode='lines',
                    name='Target Amount',
                    line=dict(color='red', width=2, dash='dash'),
                ))
            
            # Format figure
            fig.update_layout(
                title=f"Portfolio Projection Over {duration_years} Years ({return_type})",
                xaxis_title="Date",
                yaxis_title="Portfolio Value (₹)",
                legend_title="Scenarios",
                height=600,
                hovermode="x unified"
            )
            
            # Format y-axis with Indian currency format
            fig.update_yaxes(
                tickprefix="₹",
                tickformat=",.0f"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Asset allocation recommendations
            st.subheader("Recommended Asset Allocations")
            st.write("Asset allocations needed to potentially meet your base case target in the same time period under different scenarios.")
            
            # Get the base case final value and time to goal (if target exists)
            base_case_final = results[list(scenarios.keys())[0]].iloc[-1]
            base_case_return = scenarios[list(scenarios.keys())[0]]
            
            recomm_df = pd.DataFrame({
                'Scenario': scenarios.keys(),
                'Scenario Return': scenarios.values()
            })
            
            # Add recommended allocations
            equity_allocs = []
            debt_allocs = []
            required_xirrs = []
            
            for scenario_name, scenario_return in scenarios.items():
                if target_amount > 0:
                    # Calculate required XIRR to reach target in same duration
                    # Using future value formula: FV = PV*(1+r)^n
                    # Solving for r: r = (FV/PV)^(1/n) - 1
                    try:
                        required_xirr = (target_amount / current_value) ** (1/duration_years) - 1
                    except:
                        required_xirr = base_case_return
                    
                    # Adjust for the scenario's starting point (if it's a crash scenario)
                    if scenario_return < base_case_return:
                        # Account for the initial loss in the scenario
                        # New required return needs to compensate for initial loss
                        # Final value after crash then recovery: 
                        # FV = PV*(1+crash)*(1+recovery)^(n-1) = Target
                        # Solving for recovery return:
                        # recovery = (Target/(PV*(1+crash)))^(1/(n-1)) - 1
                        # Then annualize it
                        try:
                            recovery_xirr = (target_amount / (current_value * (1 + scenario_return))) ** (1/duration_years) - 1
                            required_xirr = max(recovery_xirr, required_xirr)
                        except:
                            pass
                else:
                    # Without a target, just aim for base case return
                    required_xirr = base_case_return
                
                required_xirrs.append(required_xirr)
                
                # Recommend allocation to achieve this XIRR
                equity_alloc, debt_alloc = recommend_asset_allocation(0, required_xirr)
                equity_allocs.append(equity_alloc)
                debt_allocs.append(debt_alloc)
            
            recomm_df['Required XIRR'] = required_xirrs
            recomm_df['Recommended Equity'] = [f"{min(max(e*100, 0), 100):.1f}%" for e in equity_allocs]
            recomm_df['Recommended Debt'] = [f"{min(max(d*100, 0), 100):.1f}%" for d in debt_allocs]
            recomm_df['Achievable'] = [
                "Yes" if 0 <= e <= 1 and 0 <= d <= 1 and abs(e+d-1) < 0.01 else "No" 
                for e, d in zip(equity_allocs, debt_allocs)
            ]
            
            # Format the returns
            recomm_df['Scenario Return'] = recomm_df['Scenario Return'].apply(lambda x: f"{x:.2%}")
            recomm_df['Required XIRR'] = recomm_df['Required XIRR'].apply(lambda x: f"{x:.2%}")
            
            st.dataframe(recomm_df, use_container_width=True)
            
            # Show final values in table format
            st.subheader("Projected Final Values")
            final_values = {}
            for scenario in scenarios.keys():
                final_values[scenario] = results[scenario].iloc[-1]
                
            final_df = pd.DataFrame({
                'Scenario': final_values.keys(),
                'Final Value': final_values.values()
            })
            
            final_df['Final Value'] = final_df['Final Value'].apply(
                lambda x: f"₹{format_indian(x)}"
            )
            
            if target_amount > 0:
                final_df['Goal Achievement'] = [
                    f"{min(v / target_amount * 100, 100):.1f}%" 
                    for v in results[list(scenarios.keys())].iloc[-1].values
                ]
                
                # Calculate time to goal for each scenario
                time_to_goal = []
                for scenario in scenarios.keys():
                    # Find first month where value exceeds target
                    goal_reached = False
                    for i, value in enumerate(results[scenario]):
                        if value >= target_amount:
                            years = i / 12
                            time_to_goal.append(f"{years:.1f} years")
                            goal_reached = True
                            break
                    if not goal_reached:
                        time_to_goal.append("Not reached")
                
                final_df['Time to Goal'] = time_to_goal
                
            st.dataframe(final_df, use_container_width=True)
            
            # Calculate monthly contribution needed to reach goal
            if target_amount > 0:
                st.subheader("Additional Monthly Contributions to Reach Goal")
                
                # For scenarios where goal is not reached
                monthly_contrib = {}
                for scenario, rate in scenarios.items():
                    final_value = results[scenario].iloc[-1]
                    
                    if final_value < target_amount:
                        shortfall = target_amount - final_value
                        monthly_rate = (1 + rate)**(1/12) - 1
                        
                        # Calculate using future value of annuity formula
                        # FV = PMT × ((1 + r)^n - 1) / r
                        # Solving for PMT:
                        # PMT = FV × r / ((1 + r)^n - 1)
                        
                        try:
                            contrib = shortfall * monthly_rate / ((1 + monthly_rate)**(duration_years*12) - 1)
                            # Add current monthly contribution
                            contrib += monthly_contribution
                        except:
                            contrib = 0
                        
                        monthly_contrib[scenario] = contrib
                    else:
                        monthly_contrib[scenario] = 0
                
                # Only show if at least one scenario requires additional contributions
                if any(v > 0 for v in monthly_contrib.values()):
                    contrib_df = pd.DataFrame({
                        'Scenario': monthly_contrib.keys(),
                        'Monthly Contribution Needed': monthly_contrib.values()
                    })
                    
                    contrib_df['Monthly Contribution Needed'] = contrib_df['Monthly Contribution Needed'].apply(
                        lambda x: f"₹{format_indian(x)}" if x > 0 else "Goal reached without additional contribution"
                    )
                    
                    st.dataframe(contrib_df, use_container_width=True)
            
            # Post-Crash Recovery Recommendations
            st.subheader("Post-Crash Recovery Strategies")
            st.write(f"If a market crash occurs, here are recommended allocation adjustments to potentially recover within {recovery_period} years:")
            
            # Calculate recovery allocations for negative scenarios
            recovery_data = []
            
            # Get base case return
            base_return = scenarios[list(scenarios.keys())[0]]  # First scenario is base case
            
            for scenario_name, scenario_return in scenarios.items():
                # Only show recovery strategies for negative scenarios (crashes)
                if scenario_return < 0 or "Crash" in scenario_name or "Downturn" in scenario_name:
                    # Calculate the severity of the crash (difference from base return)
                    crash_severity = abs(base_return - scenario_return)
                    
                    # Calculate recommended allocation to recover
                    new_equity, new_debt, recovery_xirr = calculate_recovery_allocation(
                        scenario_return, 
                        base_return,
                        st.session_state.equity_allocation,
                        st.session_state.debt_allocation,
                        recovery_period
                    )
                    
                    # Adjust the recovery XIRR based on crash severity
                    recovery_xirr = recovery_xirr * (1 + crash_severity)
                    
                    # Recalculate allocation with adjusted XIRR
                    new_equity, new_debt = recommend_asset_allocation(0, recovery_xirr)
                    
                    equity_change = new_equity - st.session_state.equity_allocation
                    
                    recovery_data.append({
                        'Scenario': scenario_name,
                        'Crash Impact': f"{scenario_return:.1%}",
                        'Current Equity': f"{st.session_state.equity_allocation*100:.1f}%",
                        'Recommended Equity': f"{new_equity*100:.1f}%",
                        'Equity Change': f"{equity_change*100:+.1f}%",
                        'Required XIRR': f"{recovery_xirr:.2%}",
                        'Achievable': "Yes" if 0 <= new_equity <= 1 and 0 <= new_debt <= 1 else "No"
                    })
            
            if recovery_data:
                recovery_df = pd.DataFrame(recovery_data)
                st.dataframe(recovery_df, use_container_width=True)
                
                # Add explanatory notes
                st.info("""
                **Recovery Strategy Notes:**
                - More severe crashes (larger negative impacts) require more aggressive equity allocations to recover
                - The recommended equity increase is proportional to the crash severity
                - These are mathematical projections - actual market recovery may vary
                - Consider your risk tolerance before making allocation changes
                """)
            else:
                st.info("No significant negative scenarios to show recovery strategies for.")

if __name__ == "__main__":
    run_crash_simulator()