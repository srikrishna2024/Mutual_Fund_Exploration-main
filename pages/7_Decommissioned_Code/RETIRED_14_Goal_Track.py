import streamlit as st
import pandas as pd
import numpy as np
import psycopg
from datetime import datetime, date
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    """Create database connection"""
    DB_PARAMS = {
        'dbname': 'postgres',
        'user': 'postgres',
        'password': 'admin123',
        'host': 'localhost',
        'port': '5432'
    }
    return psycopg.connect(**DB_PARAMS)

def get_goals():
    """Retrieve distinct goals from the goals table"""
    with connect_to_db() as conn:
        query = "SELECT DISTINCT goal_name FROM goals ORDER BY goal_name"
        return pd.read_sql(query, conn)['goal_name'].tolist()

def get_goal_data(goal_name):
    """Retrieve current equity and debt investment data for a selected goal"""
    with connect_to_db() as conn:
        query = """
        SELECT 
            g.goal_name,
            g.investment_type,
            g.scheme_name,
            g.scheme_code,
            CASE 
                WHEN g.is_manual_entry THEN g.current_value
                ELSE COALESCE(p.units * n.nav_value, 0)
            END as current_value,
            COALESCE(g.is_manual_entry, FALSE) as is_manual_entry
        FROM goals g
        LEFT JOIN (
            SELECT 
                scheme_name, 
                code,
                SUM(CASE 
                    WHEN transaction_type IN ('switch_out', 'redeem') THEN -units
                    WHEN transaction_type IN ('invest', 'switch_in') THEN units
                    ELSE 0 
                END) as units
            FROM portfolio_data
            GROUP BY scheme_name, code
        ) p ON g.scheme_code = p.code
        LEFT JOIN (
            SELECT code, value as nav_value
            FROM mutual_fund_nav
            WHERE (code, nav) IN (
                SELECT code, MAX(nav)
                FROM mutual_fund_nav
                GROUP BY code
            )
        ) n ON g.scheme_code = n.code
        WHERE g.goal_name = %s
        """
        df = pd.read_sql(query, conn, params=[goal_name])
        
        equity_value = df[df['investment_type'] == 'Equity']['current_value'].sum()
        debt_value = df[df['investment_type'] == 'Debt']['current_value'].sum()
        
        return equity_value, debt_value, df

def calculate_required_investment(target_amount, current_value, years, return_rate):
    """Calculate the required annual investment to reach target"""
    if years <= 0:
        return 0
    
    if return_rate <= 0:
        return max(0, (target_amount - current_value) / years)
    
    # Future value of current investments
    fv_current = current_value * (1 + return_rate) ** years
    
    if fv_current >= target_amount:
        return 0  # No additional investment needed
    
    # PMT calculation: amount needed annually to reach target
    required_pmt = (target_amount - fv_current) * return_rate / ((1 + return_rate) ** years - 1)
    return max(0, required_pmt)

def calculate_growth_path(current_value, annual_investment, return_rate, years):
    """Calculate year-by-year growth path"""
    values = [current_value]
    for year in range(1, years + 1):
        values.append(values[-1] * (1 + return_rate) + annual_investment)
    return values

def calculate_projected_growth(
    current_equity, 
    current_debt, 
    equity_return, 
    debt_return, 
    years, 
    annual_investment, 
    equity_allocation, 
    debt_allocation,
    investment_increase=0
):
    """Calculate projected growth with annual contributions and allocation"""
    equity_values = [current_equity]
    debt_values = [current_debt]
    total_values = [current_equity + current_debt]
    current_investment = annual_investment
    
    for year in range(1, years + 1):
        # Current investments grow at expected rates
        equity_growth = equity_values[-1] * (1 + equity_return)
        debt_growth = debt_values[-1] * (1 + debt_return)
        
        # New contributions based on allocation
        equity_contribution = current_investment * (equity_allocation / 100)
        debt_contribution = current_investment * (debt_allocation / 100)
        
        # Total values after growth and new contributions
        new_equity = equity_growth + equity_contribution
        new_debt = debt_growth + debt_contribution
        
        equity_values.append(new_equity)
        debt_values.append(new_debt)
        total_values.append(new_equity + new_debt)
        
        # Increase investment if specified
        current_investment *= (1 + investment_increase)
    
    return pd.DataFrame({
        'Year': list(range(years + 1)),
        'Equity': equity_values,
        'Debt': debt_values,
        'Total': total_values
    })

def calculate_actual_growth(goal_name, years, goal_df):
    """Calculate actual growth trajectory for a goal"""
    with connect_to_db() as conn:
        query = """
        WITH goal_funds AS (
            SELECT DISTINCT scheme_code 
            FROM goals 
            WHERE goal_name = %s AND (is_manual_entry IS NULL OR is_manual_entry = FALSE)
        ),
        transactions AS (
            SELECT 
                p.date,
                p.scheme_name,
                p.code,
                CASE 
                    WHEN p.transaction_type IN ('switch_out', 'redeem') THEN -p.units
                    WHEN p.transaction_type IN ('invest', 'switch_in') THEN p.units
                    ELSE 0 
                END as units_change,
                CASE 
                    WHEN p.transaction_type IN ('switch_out', 'redeem') THEN p.amount
                    WHEN p.transaction_type IN ('invest', 'switch_in') THEN -p.amount
                    ELSE 0 
                END as cashflow
            FROM portfolio_data p
            JOIN goal_funds g ON p.code = g.scheme_code
        ),
        nav_data AS (
            SELECT 
                code, 
                nav as date, 
                value as nav_value
            FROM mutual_fund_nav
            WHERE code IN (SELECT scheme_code FROM goal_funds)
            ORDER BY code, nav
        )
        SELECT 
            t.date,
            t.scheme_name,
            t.code,
            t.units_change,
            t.cashflow,
            n.nav_value
        FROM transactions t
        LEFT JOIN nav_data n ON t.code = n.code AND t.date = n.date
        ORDER BY t.date
        """
        df = pd.read_sql(query, conn, params=[goal_name])
    
    manual_investments = goal_df[goal_df['is_manual_entry'] == True]
    
    if df.empty and manual_investments.empty:
        return None
    
    # Process mutual fund transactions
    mf_values = []
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        daily_totals = df.groupby('date').agg({
            'cashflow': 'sum',
            'units_change': 'sum'
        }).reset_index()
        
        latest_nav = df.groupby('code')['nav_value'].last().reset_index()
        
        unique_dates = daily_totals['date'].unique()
        
        for date in unique_dates:
            transactions_up_to_date = df[df['date'] <= date]
            units_held = transactions_up_to_date.groupby('code')['units_change'].sum().reset_index()
            units_held = units_held[units_held['units_change'] > 0]
            
            if not units_held.empty:
                units_held = units_held.merge(latest_nav, on='code', how='left')
                current_value = (units_held['units_change'] * units_held['nav_value']).sum()
            else:
                current_value = 0
            
            mf_values.append({
                'date': date,
                'value': current_value,
                'type': 'Mutual Fund'
            })
    
    # Process manual investments
    manual_values = []
    if not manual_investments.empty:
        for _, row in manual_investments.iterrows():
            manual_values.append({
                'date': datetime.now(),
                'value': row['current_value'],
                'type': 'Manual'
            })
    
    # Combine both types of investments
    if mf_values and manual_values:
        combined_values = mf_values + manual_values
        combined_df = pd.DataFrame(combined_values)
        combined_df = combined_df.groupby('date')['value'].sum().reset_index()
    elif mf_values:
        combined_df = pd.DataFrame(mf_values)
    elif manual_values:
        combined_df = pd.DataFrame(manual_values)
    else:
        return None
    
    if not df.empty:
        combined_df = combined_df.merge(daily_totals[['date', 'cashflow']], on='date', how='left')
        combined_df['cashflow'] = combined_df['cashflow'].fillna(0)
        combined_df['cumulative_investment'] = combined_df['cashflow'].cumsum()
    else:
        combined_df['cumulative_investment'] = combined_df['value']
    
    return combined_df

def create_tracking_plot(projected_df, actual_data, target_amount, years_to_goal):
    """Create comparison plot between projected and actual growth"""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Projected growth line - solid for past, dotted for future
    if actual_data is not None:
        current_year = (datetime.now() - actual_data['date'].min()).days / 365.25
        past_mask = projected_df['Year'] <= current_year
        future_mask = projected_df['Year'] > current_year
        
        # Past projection
        fig.add_trace(
            go.Scatter(
                x=projected_df[past_mask]['Year'],
                y=projected_df[past_mask]['Total'],
                name="Projected Growth (Past)",
                line=dict(color='green', width=3),
                hovertemplate="Year %{x}<br>Value: ₹%{y:,.2f}<extra></extra>"
            ),
            secondary_y=False
        )
        
        # Future projection
        fig.add_trace(
            go.Scatter(
                x=projected_df[future_mask]['Year'],
                y=projected_df[future_mask]['Total'],
                name="Projected Growth (Future)",
                line=dict(color='green', width=3, dash='dot'),
                hovertemplate="Year %{x}<br>Value: ₹%{y:,.2f}<extra></extra>"
            ),
            secondary_y=False
        )
    else:
        # If no actual data, show full projection as dotted line
        fig.add_trace(
            go.Scatter(
                x=projected_df['Year'],
                y=projected_df['Total'],
                name="Projected Growth",
                line=dict(color='green', width=3, dash='dot'),
                hovertemplate="Year %{x}<br>Value: ₹%{y:,.2f}<extra></extra>"
            ),
            secondary_y=False
        )
    
    # Actual growth line
    if actual_data is not None:
        start_date = actual_data['date'].min()
        actual_data['Years'] = (actual_data['date'] - start_date).dt.days / 365.25
        
        fig.add_trace(
            go.Scatter(
                x=actual_data['Years'],
                y=actual_data['value'],
                name="Actual Growth",
                line=dict(color='blue', width=3),
                hovertemplate="Year %{x:.1f}<br>Value: ₹%{y:,.2f}<extra></extra>"
            ),
            secondary_y=False
        )
    
    # Target line
    fig.add_trace(
        go.Scatter(
            x=[0, years_to_goal],
            y=[target_amount, target_amount],
            name="Target Amount",
            line=dict(color='red', dash='dash'),
            hovertemplate="Target: ₹%{y:,.2f}<extra></extra>"
        ),
        secondary_y=False
    )
    
    # Mark the point where projected path crosses target
    if projected_df['Total'].iloc[-1] >= target_amount:
        cross_year = projected_df[projected_df['Total'] >= target_amount]['Year'].min()
        cross_value = projected_df[projected_df['Year'] == cross_year]['Total'].values[0]
        
        fig.add_trace(
            go.Scatter(
                x=[cross_year],
                y=[cross_value],
                name="Target Achieved",
                mode='markers',
                marker=dict(color='gold', size=12),
                hovertemplate="Achieved in Year %{x}<br>Value: ₹%{y:,.2f}<extra></extra>"
            ),
            secondary_y=False
        )
    
    # Deviation shading
    if actual_data is not None and len(actual_data) > 1:
        projected_at_actual = np.interp(
            actual_data['Years'],
            projected_df['Year'],
            projected_df['Total']
        )
        
        fig.add_trace(
            go.Scatter(
                x=actual_data['Years'].tolist() + actual_data['Years'].tolist()[::-1],
                y=projected_at_actual.tolist() + actual_data['value'].tolist()[::-1],
                fill='toself',
                fillcolor='rgba(255,165,0,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name="Deviation",
                hoverinfo='skip'
            ),
            secondary_y=False
        )
    
    fig.update_layout(
        title="Goal Progress Tracking",
        xaxis_title="Years",
        yaxis_title="Portfolio Value (₹)",
        legend_title="Growth Paths",
        hovermode="x unified",
        yaxis=dict(
            tickformat=",",
            separatethousands=True
        )
    )
    
    return fig

def main():
    st.set_page_config(page_title="Are We On Track Tool", layout="wide")
    st.title("Goal Progress Tracking Tool")
    
    if 'analyze_clicked' not in st.session_state:
        st.session_state.analyze_clicked = False
    
    goals = get_goals()
    if not goals:
        st.warning("No goals found in the database. Please create goals first.")
        return
    
    selected_goal = st.selectbox("Select Goal", goals)
    
    st.header("1. Goal Definition")
    col1, col2 = st.columns(2)
    with col1:
        target_amount = st.number_input("Target Amount (₹)", min_value=0, value=1000000)
    with col2:
        max_date = date(2100, 12, 31)
        target_date = st.date_input("Target Date", 
                                  min_value=datetime.today(),
                                  max_value=max_date,
                                  value=date(datetime.today().year + 10, 1, 1))
    
    years_to_goal = max(0.1, (target_date - datetime.today().date()).days / 365.25)  # Ensure at least 0.1 years
    st.write(f"Years to Goal: {years_to_goal:.1f} years")
    
    current_equity, current_debt, goal_df = get_goal_data(selected_goal)
    total_current = current_equity + current_debt
    
    st.header("2. Growth Assumptions")
    col1, col2 = st.columns(2)
    with col1:
        equity_return = st.number_input("Expected Equity Return (%)", min_value=0.0, max_value=30.0, value=12.0) / 100
        debt_return = st.number_input("Expected Debt Return (%)", min_value=0.0, max_value=15.0, value=7.0) / 100
    with col2:
        equity_allocation = st.slider("Equity Allocation (%)", min_value=0, max_value=100, value=60)
        debt_allocation = 100 - equity_allocation
        st.write(f"Debt Allocation: {debt_allocation}%")
    
    # Calculate blended return rate
    blended_return = (equity_return * equity_allocation/100) + (debt_return * debt_allocation/100)
    
    st.header("3. Investment Plan")
    col1, col2 = st.columns(2)
    with col1:
        annual_investment = st.number_input("Annual Investment (₹)", min_value=0, value=120000)
    with col2:
        investment_increase = st.number_input("Yearly Investment Increase (%)", min_value=0.0, max_value=50.0, value=5.0) / 100
    
    # Calculate required investment
    required_annual = calculate_required_investment(
        target_amount, 
        total_current, 
        years_to_goal, 
        blended_return
    )
    
    # Calculate required equity and debt investments
    required_equity = required_annual * (equity_allocation / 100)
    required_debt = required_annual * (debt_allocation / 100)
    
    # Always show investment information for all goals
    st.write(f"**Required Annual Investment:** {format_indian_number(required_annual)}")
    st.write(f"**- Equity Component:** {format_indian_number(required_equity)} ({equity_allocation}%)")
    st.write(f"**- Debt Component:** {format_indian_number(required_debt)} ({debt_allocation}%)")
    st.write(f"**Your Annual Investment:** {format_indian_number(annual_investment)}")
    
    if st.button("Analyze Progress", type="primary"):
        st.session_state.analyze_clicked = True
    
    if st.session_state.analyze_clicked:
        projected_df = calculate_projected_growth(
            current_equity,
            current_debt,
            equity_return,
            debt_return,
            int(np.ceil(years_to_goal)),
            annual_investment,
            equity_allocation,
            debt_allocation,
            investment_increase
        )
        
        actual_data = calculate_actual_growth(selected_goal, int(np.ceil(years_to_goal)), goal_df)
        
        st.header("Current Status")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Equity Value", format_indian_number(current_equity))
        with col2:
            st.metric("Current Debt Value", format_indian_number(current_debt))
        with col3:
            st.metric("Total Current Value", format_indian_number(total_current))
        
        if projected_df is not None:
            time_elapsed_fraction = min(1.0, years_to_goal / projected_df['Year'].iloc[-1])
            projected_current = np.interp(
                time_elapsed_fraction * projected_df['Year'].iloc[-1],
                projected_df['Year'],
                projected_df['Total']
            )
            
            progress_percent = (total_current / target_amount) * 100
            deviation = total_current - projected_current
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Progress Toward Target", f"{progress_percent:.1f}%")
            with col2:
                st.metric("Deviation from Projected", 
                         f"{format_indian_number(deviation)}",
                         "Ahead" if deviation >= 0 else "Behind")
        
        st.header("Progress Tracking")
        fig = create_tracking_plot(projected_df, actual_data, target_amount, years_to_goal)
        st.plotly_chart(fig, use_container_width=True)
        
        st.header("Detailed Projection")
        if projected_df is not None:
            # Show when target will be achieved
            if projected_df['Total'].iloc[-1] >= target_amount:
                cross_year = projected_df[projected_df['Total'] >= target_amount]['Year'].min()
                st.success(f"Projected to achieve target in year {cross_year:.1f} ({(cross_year - years_to_goal):.1f} years {'early' if cross_year < years_to_goal else 'late'})")
            else:
                st.error("Projected to miss target with current plan")
            
            # Show final projected value
            final_projected = projected_df['Total'].iloc[-1]
            st.write(f"Final Projected Value: {format_indian_number(final_projected)}")
            st.write(f"Target Amount: {format_indian_number(target_amount)}")
            st.write(f"Difference: {format_indian_number(final_projected - target_amount)}")
            
            # Show detailed year-by-year investment requirements
            st.subheader("Year-by-Year Investment Plan")
            st.write("Here's how much you should invest each year in equity and debt to reach your target:")
            
            # Create a dataframe for yearly investment plan
            investment_plan = []
            current_investment = annual_investment
            for year in range(1, int(np.ceil(years_to_goal)) + 1):
                equity_investment = current_investment * (equity_allocation / 100)
                debt_investment = current_investment * (debt_allocation / 100)
                investment_plan.append({
                    'Year': year,
                    'Total Investment': current_investment,
                    'Equity Investment': equity_investment,
                    'Debt Investment': debt_investment
                })
                current_investment *= (1 + investment_increase)
            
            investment_plan_df = pd.DataFrame(investment_plan)
            st.dataframe(investment_plan_df.style.format({
                'Total Investment': lambda x: format_indian_number(x),
                'Equity Investment': lambda x: format_indian_number(x),
                'Debt Investment': lambda x: format_indian_number(x)
            }))
        
        st.header("Recommendations")
        if projected_df is not None:
            if total_current >= target_amount:
                st.success("🎉 Congratulations! You've already achieved your target amount!")
            else:
                if required_annual > 0 and annual_investment > required_annual * 1.05:  # 5% buffer
                    if projected_df['Total'].iloc[-1] < target_amount:
                        st.warning("Despite investing more than required, you're projected to miss target.")
                        st.write("This suggests your actual returns may be lower than expected.")
                        st.write(f"Expected return: {blended_return*100:.1f}%")
                        st.write("Consider reviewing your investment performance.")
                    else:
                        st.success(f"Your investments are sufficient to reach target by year {cross_year:.1f}")
                elif required_annual > 0 and annual_investment < required_annual * 0.95:  # 5% buffer
                    additional_needed = required_annual - annual_investment
                    additional_equity = additional_needed * (equity_allocation / 100)
                    additional_debt = additional_needed * (debt_allocation / 100)
                    
                    st.warning(f"Increase annual investment by {format_indian_number(additional_needed)} to reach target")
                    st.write(f"- Additional Equity needed: {format_indian_number(additional_equity)}")
                    st.write(f"- Additional Debt needed: {format_indian_number(additional_debt)}")
                else:
                    st.success("Your investments are approximately on track to reach target")

if __name__ == "__main__":
    main()