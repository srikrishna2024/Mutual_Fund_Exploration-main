import streamlit as st
import pandas as pd
import psycopg
from plotly import graph_objects as go

def format_indian_number(number):
    """Format a number in Indian style (lakhs, crores)"""
    if number >= 10000000:  # crores
        return f"₹{number/10000000:.2f} Cr"
    elif number >= 100000:  # lakhs
        return f"₹{number/100000:.2f} L"
    elif number >= 1000:  # thousands
        return f"₹{number/1000:.2f} K"
    else:
        return f"₹{number:.2f}"

def connect_to_db():
    """Establish a database connection."""
    DB_PARAMS = {
        'dbname': 'postgres',
        'user': 'postgres',
        'password': 'admin123',
        'host': 'localhost',
        'port': '5432'
    }
    return psycopg.connect(**DB_PARAMS)

def get_goals():
    """Retrieve distinct goals from the goals table."""
    with connect_to_db() as conn:
        query = "SELECT DISTINCT goal_name FROM goals ORDER BY goal_name"
        return pd.read_sql(query, conn)['goal_name'].tolist()

def get_goal_data(goal_name):
    """Retrieve current equity and debt investment data for a selected goal."""
    with connect_to_db() as conn:
        # Get current investments
        query = """
        SELECT investment_type, SUM(current_value) AS total_value
        FROM goals
        WHERE goal_name = %s
        GROUP BY investment_type
        """
        df = pd.read_sql(query, conn, params=[goal_name])
        investments = {row['investment_type']: row['total_value'] for _, row in df.iterrows()}
        
        # Get current NAV values for all funds in this goal
        query = """
        SELECT g.code, g.scheme_name, g.units, mf.nav_value
        FROM goals g
        JOIN mutual_fund_nav mf ON g.code = mf.code 
        WHERE g.goal_name = %s
        AND mf.nav = (SELECT MAX(nav) FROM mutual_fund_nav WHERE code = g.code)
        """
        nav_data = pd.read_sql(query, conn, params=[goal_name])
        
        # Calculate current values
        if not nav_data.empty:
            nav_data['current_value'] = nav_data['units'] * nav_data['nav_value']
            equity_value = nav_data[nav_data['investment_type'] == 'Equity']['current_value'].sum()
            debt_value = nav_data[nav_data['investment_type'] == 'Debt']['current_value'].sum()
            return equity_value, debt_value
        else:
            return investments.get('Equity', 0), investments.get('Debt', 0)

def calculate_growth(initial, rate, years, annual_contribution=0):
    """Calculate yearly growth based on compound interest and contributions."""
    values = [initial]
    for year in range(1, years + 1):
        new_value = values[-1] * (1 + rate) + annual_contribution
        values.append(new_value)
    return values

def calculate_total_growth_glidepath(
    initial_equity, initial_debt, equity_rate, debt_rate, years, annual_investment,
    starting_equity_allocation, ending_equity_allocation, allocation_change_start_year, investment_increase=0
):
    """
    Calculate yearly growth with a glide path for changing equity-debt allocation
    and sequence of returns risk.

    Returns:
    - A DataFrame with separate equity and debt growth lines.
    - Final total values.
    """
    # Initialize values
    equity_values = [initial_equity]
    debt_values = [initial_debt]
    total_values = [initial_equity + initial_debt]
    current_annual_investment = annual_investment

    # Calculate yearly allocation adjustments
    allocation_change_years = max(1, allocation_change_start_year)
    years_to_adjust = years - allocation_change_years + 1
    equity_allocations = [
        max(
            starting_equity_allocation - ((starting_equity_allocation - ending_equity_allocation) / years_to_adjust) * max(0, year - allocation_change_years),
            ending_equity_allocation
        )
        for year in range(years)
    ]
    debt_allocations = [100 - equity_allocation for equity_allocation in equity_allocations]

    # Yearly calculations
    for year in range(1, years + 1):
        previous_equity = equity_values[-1]
        previous_debt = debt_values[-1]

        # Current allocation percentages
        equity_allocation = equity_allocations[year - 1]
        debt_allocation = debt_allocations[year - 1]

        # Contributions based on allocation
        yearly_equity_contribution = current_annual_investment * (equity_allocation / 100)
        yearly_debt_contribution = current_annual_investment * (debt_allocation / 100)

        # Growth calculations
        equity_growth = (previous_equity + yearly_equity_contribution) * (1 + equity_rate)
        debt_growth = (previous_debt + yearly_debt_contribution) * (1 + debt_rate)

        # Append new values
        equity_values.append(equity_growth)
        debt_values.append(debt_growth)
        total_values.append(equity_growth + debt_growth)

        # Increase annual investment
        current_annual_investment *= (1 + investment_increase)

    # Create DataFrame for plotting
    growth_df = pd.DataFrame({
        'Year': list(range(1, years + 2)),
        'Equity': equity_values,
        'Debt': debt_values,
        'Total': total_values,
        'Equity Allocation': [starting_equity_allocation] + equity_allocations,
        'Debt Allocation': [100 - starting_equity_allocation] + debt_allocations
    })

    return growth_df, total_values[-1]

def create_comparison_plot(years, expected_growth, conservative_growth, benchmark_growth):
    """Generate an interactive plot comparing different growth paths."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=expected_growth,
        name="Expected Growth",
        mode="lines+markers",
        hovertemplate="₹%{y:,.2f}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        y=conservative_growth,
        name="Conservative Growth",
        mode="lines+markers",
        hovertemplate="₹%{y:,.2f}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        y=benchmark_growth,
        name="Benchmark Growth",
        mode="lines+markers",
        hovertemplate="₹%{y:,.2f}<extra></extra>"
    ))
    fig.update_layout(
        title="Investment Growth Comparison",
        xaxis_title="Years",
        yaxis_title="Value (₹)",
        legend_title="Growth Paths",
        yaxis=dict(
            tickformat=",",
            separatethousands=True
        )
    )
    return fig

def create_simulation_plot_with_glidepath(growth_df):
    """
    Create a simulation plot showing equity, debt, and total growth over the years.
    """
    fig = go.Figure()

    # Add equity growth line
    fig.add_trace(go.Scatter(
        x=growth_df['Year'],
        y=growth_df['Equity'],
        mode='lines+markers',
        name='Equity Growth',
        hovertemplate="Year %{x}<br>₹%{y:,.2f}<extra></extra>"
    ))

    # Add debt growth line
    fig.add_trace(go.Scatter(
        x=growth_df['Year'],
        y=growth_df['Debt'],
        mode='lines+markers',
        name='Debt Growth',
        hovertemplate="Year %{x}<br>₹%{y:,.2f}<extra></extra>"
    ))

    # Add total growth line
    fig.add_trace(go.Scatter(
        x=growth_df['Year'],
        y=growth_df['Total'],
        mode='lines+markers',
        name='Total Growth',
        hovertemplate="Year %{x}<br>₹%{y:,.2f}<extra></extra>"
    ))

    # Configure plot layout
    fig.update_layout(
        title="Investment Growth with Glide Path",
        xaxis_title="Year",
        yaxis_title="Value (₹)",
        legend_title="Growth Paths",
        yaxis=dict(tickformat=",", separatethousands=True)
    )

    return fig

def calculate_retirement_needs(current_expenses, inflation_rate, years_to_retire, life_expectancy):
    """
    Calculate detailed retirement corpus needs with year-by-year breakdown.
    Returns both total corpus needed and yearly expenses dataframe.
    """
    retirement_age = 60  # Retirement starts at age 60
    retirement_years = life_expectancy - retirement_age  # Years in retirement
    future_annual_expense = current_expenses * (1 + inflation_rate) ** years_to_retire

    # Create lists for year-by-year breakdown
    years_list = []
    age_list = []
    expenses_list = []
    cumulative_corpus_list = []

    total_corpus_needed = 0  # Total retirement corpus
    for year in range(retirement_years):
        # Expense for each year during retirement
        expense_in_year = future_annual_expense * (1 + inflation_rate) ** year
        years_list.append(year + 1)
        age_list.append(retirement_age + year)
        expenses_list.append(expense_in_year)

        # Calculate remaining corpus needed for future years
        remaining_years = retirement_years - year
        year_corpus = sum(
            expense_in_year / ((1 + inflation_rate) ** future_year)
            for future_year in range(remaining_years)
        )
        cumulative_corpus_list.append(year_corpus)
        if year == 0:  # The total corpus needed at retirement age
            total_corpus_needed = year_corpus

    # Create DataFrame with the breakdown
    breakdown_df = pd.DataFrame({
        'Year': years_list,
        'Age': age_list,
        'Annual Expenses': expenses_list,
        'Corpus Required': cumulative_corpus_list
    })

    return total_corpus_needed, breakdown_df

def main():
    st.set_page_config(page_title="Goal GlidePath Analysis", layout="wide")
    st.title("Goal GlidePath Analysis")

    # Retrieve goals from the database
    goals = get_goals()
    if not goals:
        st.warning("No goals found in the database.")
        return

    # Goal selection
    selected_goal = st.selectbox("Select Goal", goals)
    if not selected_goal:
        return

    # Get current investments with NAV-based current values
    equity, debt = get_goal_data(selected_goal)
    total_current_value = equity + debt
    
    st.subheader("Current Investment Status")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Equity Value", format_indian_number(equity))
    with col2:
        st.metric("Debt Value", format_indian_number(debt))
    with col3:
        st.metric("Total Current Value", format_indian_number(total_current_value))

    # Investment details
    st.subheader("Investment Plan")
    col1, col2 = st.columns(2)
    with col1:
        annual_investment = st.number_input("Initial Yearly Investment (₹)", min_value=0, value=50000)
    with col2:
        investment_increase = st.number_input("Yearly Investment Increase (%)", min_value=0.0, max_value=50.0, value=5.0) / 100

    years = st.slider("Years to Goal", min_value=1, max_value=30, value=10)

    # Target Corpus for Non-Retirement Goals
    target_corpus = None
    if selected_goal.lower() != "retirement":
        target_corpus = st.number_input("Target Corpus (₹)", min_value=0, value=1000000)

    # Glide path allocation details
    st.subheader("Asset Allocation Strategy")
    if selected_goal.lower() == "retirement":
        col1, col2 = st.columns(2)
        with col1:
            starting_equity_allocation = st.slider(
                "Starting Equity Allocation (%)", min_value=50, max_value=100, value=60
            )
            ending_equity_allocation = st.slider(
                "Ending Equity Allocation (%)", min_value=0, max_value=50, value=40
            )
        with col2:
            years_before_retirement_to_start = st.slider(
                "Years Before Retirement to Start Adjusting Allocation", min_value=1, max_value=years - 1, value=5
            )
    else:
        col1, col2 = st.columns(2)
        with col1:
            starting_equity_allocation = st.slider(
                "Equity Allocation (%)", min_value=0, max_value=100, value=60
            )
        with col2:
            ending_equity_allocation = 100 - starting_equity_allocation
            st.write(f"Debt Allocation: {ending_equity_allocation}%")

    # Return expectations
    st.subheader("Return Expectations")
    col1, col2, col3 = st.columns(3)
    with col1:
        equity_rate = st.number_input("Expected Equity Return (%)", min_value=0.0, value=12.0) / 100
    with col2:
        debt_rate = st.number_input("Expected Debt Return (%)", min_value=0.0, value=7.0) / 100
    with col3:
        benchmark_rate = st.number_input("Expected Benchmark Return (%)", min_value=0.0, value=12.0) / 100

    # Calculate growth with glide path
    allocation_change_start_year = years - years_before_retirement_to_start if selected_goal.lower() == "retirement" else years
    growth_df, final_value = calculate_total_growth_glidepath(
        equity, debt, equity_rate, debt_rate, years, annual_investment,
        starting_equity_allocation, ending_equity_allocation,
        allocation_change_start_year, investment_increase
    )

    # Calculate benchmark growth
    benchmark_values = [total_current_value]
    current_investment = annual_investment
    for year in range(1, years + 1):
        new_value = benchmark_values[-1] * (1 + benchmark_rate) + current_investment
        benchmark_values.append(new_value)
        current_investment *= (1 + investment_increase)

    # Display the simulation plot
    st.subheader("Projected Growth")
    simulation_plot = create_simulation_plot_with_glidepath(growth_df)
    simulation_plot.add_trace(go.Scatter(
        x=list(range(1, years + 2)),
        y=benchmark_values,
        mode='lines+markers',
        name='Benchmark Growth',
        hovertemplate="Year %{x}<br>₹%{y:,.2f}<extra></extra>"
    ))
    st.plotly_chart(simulation_plot, use_container_width=True)

    # Allocation chart
    st.subheader("Allocation Over Time")
    allocation_plot = go.Figure()
    allocation_plot.add_trace(go.Scatter(
        x=growth_df['Year'],
        y=growth_df['Equity Allocation'],
        mode='lines+markers',
        name='Equity Allocation',
        hovertemplate="Year %{x}<br>%{y:.1f}%<extra></extra>"
    ))
    allocation_plot.add_trace(go.Scatter(
        x=growth_df['Year'],
        y=growth_df['Debt Allocation'],
        mode='lines+markers',
        name='Debt Allocation',
        hovertemplate="Year %{x}<br>%{y:.1f}%<extra></extra>"
    ))
    allocation_plot.update_layout(
        title="Asset Allocation Glide Path",
        xaxis_title="Year",
        yaxis_title="Allocation (%)",
        legend_title="Asset Class"
    )
    st.plotly_chart(allocation_plot, use_container_width=True)

    # Insights
    st.subheader("Projection Summary")
    final_equity = growth_df['Equity'].iloc[-1]
    final_debt = growth_df['Debt'].iloc[-1]
    benchmark_final = benchmark_values[-1]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Projected Equity Value", format_indian_number(final_equity))
    with col2:
        st.metric("Projected Debt Value", format_indian_number(final_debt))
    with col3:
        st.metric("Projected Total Value", format_indian_number(final_value))

    # On Track or Off Track Analysis
    st.subheader("Goal Status")
    if selected_goal.lower() == "retirement":
        if final_value >= benchmark_final:
            st.success("✅ You are on track to meet your retirement goal!")
        else:
            st.error("⚠️ You are off track. Consider adjusting your investments.")
    else:
        if total_current_value >= target_corpus:
            st.success("✅ You have already achieved your target corpus!")
        elif final_value >= target_corpus:
            st.success("✅ You are on track to meet your target corpus!")
        else:
            st.error("⚠️ You are off track. Consider increasing your investments or adjusting your allocation.")

    if selected_goal.lower() == "retirement":
        st.subheader("Retirement Planning Details")
        current_age = st.number_input("Current Age", min_value=20, max_value=70, value=30)
        retirement_age = st.number_input("Retirement Age", min_value=current_age + 1, max_value=80, value=60)
        life_expectancy = st.number_input("Life Expectancy", min_value=retirement_age + 1, max_value=100, value=80)
        current_expenses = st.number_input("Current Annual Expenses (₹)", min_value=0, value=500000)
        inflation_rate = st.number_input("Expected Inflation Rate (%)", min_value=0.0, value=5.0) / 100

        years_to_retire = retirement_age - current_age
        retirement_corpus_needed, retirement_breakdown = calculate_retirement_needs(
            current_expenses, inflation_rate, years_to_retire, life_expectancy
        )

        st.write(f"Total Retirement Corpus Required at Age {retirement_age}: {format_indian_number(retirement_corpus_needed)}")

        with st.expander("View Year-by-Year Retirement Breakdown"):
            retirement_breakdown['Annual Expenses'] = retirement_breakdown['Annual Expenses'].apply(format_indian_number)
            retirement_breakdown['Corpus Required'] = retirement_breakdown['Corpus Required'].apply(format_indian_number)
            st.dataframe(retirement_breakdown)

if __name__ == "__main__":
    main()