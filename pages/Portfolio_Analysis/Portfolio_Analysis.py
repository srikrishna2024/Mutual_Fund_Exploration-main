import streamlit as st
import pandas as pd
from datetime import datetime
import psycopg
import plotly.express as px
import plotly.graph_objects as go
from pyxirr import xirr  # You'll need to install the xirr package: pip install xirr

def format_indian_number(number):
    """
    Format a number in Indian style (with commas for lakhs, crores)
    Example: 10234567 becomes 1,02,34,567
    """
    if pd.isna(number):
        return "0"
    
    number = float(number)
    is_negative = number < 0
    number = abs(number)
    
    # Convert to string with 2 decimal places
    str_number = f"{number:,.2f}"
    
    # Split the decimal part
    parts = str_number.split('.')
    whole_part = parts[0].replace(',', '')
    decimal_part = parts[1] if len(parts) > 1 else '00'
    
    # Format the whole part in Indian style
    result = ""
    length = len(whole_part)
    
    # Handle numbers less than 1000
    if length <= 3:
        result = whole_part
    else:
        # Add the last 3 digits
        result = whole_part[-3:]
        # Add other digits in groups of 2
        remaining = whole_part[:-3]
        while remaining:
            result = (remaining[-2:] if len(remaining) >= 2 else remaining) + ',' + result
            remaining = remaining[:-2]
    
    # Add decimal part and negative sign if needed
    formatted = f"₹{'-' if is_negative else ''}{result}.{decimal_part}"
    return formatted

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

def get_portfolio_data():
    """Retrieve all records from portfolio_data table"""
    with connect_to_db() as conn:
        query = """
            SELECT date, scheme_name, code, transaction_type, value, units, amount 
            FROM portfolio_data 
            ORDER BY date, scheme_name
        """
        return pd.read_sql(query, conn)

def get_latest_nav():
    """Retrieve the latest NAVs from mutual_fund_nav table"""
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

def get_benchmark_data():
    """Retrieve benchmark data"""
    with connect_to_db() as conn:
        query = """
            SELECT date, price
            FROM benchmark
            ORDER BY date
        """
        return pd.read_sql(query, conn)

def get_goal_mappings():
    """Retrieve goal mappings from the goals table including both MF and debt investments"""
    with connect_to_db() as conn:
        query = """
            WITH mf_latest_values AS (
                SELECT g.goal_name, g.investment_type, g.scheme_name, g.scheme_code,
                       CASE 
                           WHEN g.is_manual_entry THEN g.current_value
                           ELSE COALESCE(p.units * n.value, 0)
                       END as current_value
                FROM goals g
                LEFT JOIN (
                    SELECT scheme_name, code,
                           SUM(CASE 
                               WHEN transaction_type IN ('switch_out', 'redeem') THEN -units
                               WHEN transaction_type IN ('invest', 'switch_in') THEN units
                               ELSE 0 
                           END) as units
                    FROM portfolio_data
                    GROUP BY scheme_name, code
                ) p ON g.scheme_code = p.code
                LEFT JOIN (
                    SELECT code, value
                    FROM mutual_fund_nav
                    WHERE (code, nav) IN (
                        SELECT code, MAX(nav)
                        FROM mutual_fund_nav
                        GROUP BY code
                    )
                ) n ON g.scheme_code = n.code
            )
            SELECT goal_name, investment_type, scheme_name, scheme_code, current_value
            FROM mf_latest_values
            ORDER BY goal_name, investment_type, scheme_name
        """
        return pd.read_sql(query, conn)

def prepare_cashflows(df):
    """Prepare cashflow data from portfolio transactions"""
    df['cashflow'] = df.apply(lambda x: 
        -x['amount'] if x['transaction_type'] in ('invest', 'switch_in')  # Negative because it's money going out
        else x['amount'] if x['transaction_type'] in ('redeem', 'switch_out')  # Positive because it's money coming in
        else 0, 
        axis=1
    )
    return df

def calculate_units(df):
    """Calculate net units for each scheme based on transactions"""
    df['units_change'] = df.apply(lambda x: 
        x['units'] if x['transaction_type'] in ('invest', 'switch_in')
        else -x['units'] if x['transaction_type'] in ('redeem', 'switch_out')
        else 0,
        axis=1
    )
    
    # Calculate cumulative units for each scheme
    df = df.sort_values(['scheme_name', 'date'])
    df['cumulative_units'] = df.groupby(['scheme_name', 'code'])['units_change'].cumsum()
    return df

def calculate_portfolio_weights(df, latest_nav, goal_mappings):
    """Calculate current portfolio weights for each scheme including debt investments"""
    # First calculate MF values by getting the latest units for each scheme
    mf_df = df.groupby(['scheme_name', 'code']).agg({
        'cumulative_units': 'last'
    }).reset_index()

    mf_df = mf_df.merge(latest_nav, on='code', how='left')
    mf_df['current_value'] = mf_df['cumulative_units'] * mf_df['nav_value']

    # Get debt investments from goal mappings, excluding any that are already in MF investments
    debt_investments = goal_mappings[
        (goal_mappings['investment_type'] == 'Debt') & 
        (~goal_mappings['scheme_name'].isin(mf_df['scheme_name']))
    ]
    
    # Combine MF and debt investments
    combined_df = pd.concat([
        mf_df[['scheme_name', 'current_value']],
        debt_investments[['scheme_name', 'current_value']]
    ])

    # Group by scheme_name to handle any remaining duplicates
    combined_df = combined_df.groupby('scheme_name')['current_value'].sum().reset_index()

    total_value = combined_df['current_value'].sum()
    combined_df['weight'] = (combined_df['current_value'] / total_value) * 100 if total_value > 0 else 0

    return combined_df

def calculate_portfolio_vs_benchmark(df, latest_nav, goal_mappings, benchmark_df):
    """Calculate portfolio value over time vs benchmark performance"""
    if df.empty or benchmark_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    # Convert dates to datetime
    df['date'] = pd.to_datetime(df['date'])
    benchmark_df['date'] = pd.to_datetime(benchmark_df['date'])
    
    # Get the oldest date from portfolio
    oldest_portfolio_date = df['date'].min()
    
    # Filter benchmark data to start from the oldest portfolio date
    benchmark_df = benchmark_df[benchmark_df['date'] >= oldest_portfolio_date].copy()
    
    if benchmark_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    # Get all unique dates from portfolio transactions
    portfolio_dates = df['date'].sort_values().unique()
    
    portfolio_growth = []
    benchmark_growth = []
    
    # Initialize benchmark tracking
    benchmark_units = 0
    benchmark_invested = 0
    
    for date in portfolio_dates:
        # Calculate portfolio value as of this date
        transactions_up_to_date = df[df['date'] <= date].copy()
        
        # Portfolio value calculation
        current_units = transactions_up_to_date.groupby(['scheme_name', 'code'])['cumulative_units'].last().reset_index()
        current_units = current_units.merge(latest_nav, on='code', how='left')
        current_units['current_value'] = current_units['cumulative_units'] * current_units['nav_value']
        
        # Include debt investments in the portfolio value (assuming they remain constant)
        debt_value = goal_mappings[goal_mappings['investment_type'] == 'Debt']['current_value'].sum()
        portfolio_value = current_units['current_value'].sum() + debt_value
        
        portfolio_growth.append({
            'date': date,
            'portfolio_value': portfolio_value
        })
        
        # Calculate benchmark value as of this date
        # Get new investments on this date
        new_investments = df[df['date'] == date]
        new_investment_amount = new_investments[
            new_investments['transaction_type'].isin(['invest', 'switch_in'])
        ]['amount'].sum()
        
        if new_investment_amount > 0:
            # Find benchmark price on or closest to this date
            benchmark_on_date = benchmark_df[benchmark_df['date'] <= date]
            if not benchmark_on_date.empty:
                benchmark_price_on_date = benchmark_on_date['price'].iloc[-1]
                
                if benchmark_price_on_date > 0:
                    # Buy benchmark units with the same investment amount
                    units_bought = new_investment_amount / benchmark_price_on_date
                    benchmark_units += units_bought
                    benchmark_invested += new_investment_amount
        
        # Calculate current benchmark value
        benchmark_on_date = benchmark_df[benchmark_df['date'] <= date]
        if not benchmark_on_date.empty and benchmark_units > 0:
            current_benchmark_price = benchmark_on_date['price'].iloc[-1]
            benchmark_value = benchmark_units * current_benchmark_price
        else:
            benchmark_value = benchmark_invested
        
        benchmark_growth.append({
            'date': date,
            'benchmark_value': benchmark_value,
            'benchmark_invested': benchmark_invested
        })
    
    portfolio_df = pd.DataFrame(portfolio_growth)
    benchmark_df_result = pd.DataFrame(benchmark_growth)
    
    return portfolio_df, benchmark_df_result

def calculate_annual_performance(df, latest_nav, goal_mappings, benchmark_df):
    """Calculate annual portfolio value and benchmark performance"""
    if df.empty or benchmark_df.empty:
        return pd.DataFrame()
    
    # Convert dates to datetime
    df['date'] = pd.to_datetime(df['date'])
    benchmark_df['date'] = pd.to_datetime(benchmark_df['date'])
    
    # Get all unique years in the data
    min_year = df['date'].dt.year.min()
    max_year = datetime.now().year
    years = range(min_year, max_year + 1)
    
    annual_data = []
    annual_data = []
    
    for year in years:
        year_end = datetime(year, 12, 31)
        # Calculate portfolio value at year end
        portfolio_transactions = df[df['date'] <= year_end].copy()
        
        if not portfolio_transactions.empty:
            # Get portfolio value
            final_units = portfolio_transactions.groupby(['scheme_name', 'code'])['cumulative_units'].last().reset_index()
            final_units = final_units.merge(latest_nav, on='code', how='left')
            final_units['current_value'] = final_units['cumulative_units'] * final_units['nav_value']
            debt_final_value = goal_mappings[goal_mappings['investment_type'] == 'Debt']['current_value'].sum()
            portfolio_value = final_units['current_value'].sum() + debt_final_value
            
            # Calculate total invested amount up to this year
            invested_amount = portfolio_transactions[
                portfolio_transactions['transaction_type'].isin(['invest', 'switch_in'])
            ]['amount'].sum()
            
            # Calculate benchmark performance
            investments_up_to_year = df[df['date'] <= year_end]
            total_benchmark_investment = investments_up_to_year[
                investments_up_to_year['transaction_type'].isin(['invest', 'switch_in'])
            ]['amount'].sum()
            
            # Calculate benchmark units accumulated up to this year
            benchmark_units_year = 0
            for _, row in investments_up_to_year[
                investments_up_to_year['transaction_type'].isin(['invest', 'switch_in'])
            ].iterrows():
                investment_date = row['date']
                investment_amount = row['amount']
                
                # Find benchmark price on or before investment date
                benchmark_on_date = benchmark_df[benchmark_df['date'] <= investment_date]
                if not benchmark_on_date.empty:
                    benchmark_price_on_date = benchmark_on_date['price'].iloc[-1]
                    if benchmark_price_on_date > 0:
                        units_bought = investment_amount / benchmark_price_on_date
                        benchmark_units_year += units_bought
            
            # Get benchmark price at year end
            benchmark_at_year_end = benchmark_df[benchmark_df['date'] <= year_end]
            if not benchmark_at_year_end.empty and benchmark_units_year > 0:
                benchmark_price_at_year_end = benchmark_at_year_end['price'].iloc[-1]
                benchmark_value = benchmark_units_year * benchmark_price_at_year_end
            else:
                benchmark_value = total_benchmark_investment
            
            # Calculate returns
            portfolio_return = ((portfolio_value - invested_amount) / invested_amount * 100) if invested_amount > 0 else 0
            benchmark_return = ((benchmark_value - total_benchmark_investment) / total_benchmark_investment * 100) if total_benchmark_investment > 0 else 0
            
            annual_data.append({
                'year': year,
                'portfolio_value': portfolio_value,
                'benchmark_value': benchmark_value,
                'invested_amount': invested_amount,
                'portfolio_return': portfolio_return,
                'benchmark_return': benchmark_return
            })
    
    return pd.DataFrame(annual_data)

from pyxirr import xirr  # You'll need to install the xirr package: pip install xirr

def calculate_equity_xirr(df, latest_nav):
    """Calculate XIRR for equity portion of the portfolio"""
    try:
        # 1. Select equity transactions
        ts = df[df['transaction_type'].isin(['invest','redeem','switch_in','switch_out'])]
        if ts.empty:
            return 0.0

        # 2. Build cashflows: negative for invest/switch_in, positive for redeem/switch_out
        dates, amounts = [], []
        for _, row in ts.iterrows():
            dates.append(row['date'])
            amt = -row['amount'] if row['transaction_type'] in ('invest','switch_in') else row['amount']
            amounts.append(float(amt))

        # 3. Add current value as final inflow
        total_current = 0.0
        for code in df['code'].unique():
            units = df.loc[df['code'] == code, 'cumulative_units'].iloc[-1]
            navs = latest_nav.loc[latest_nav['code'] == code, 'nav_value']
            if not navs.empty and not pd.isna(navs.values[0]):
                total_current += units * float(navs.values[0])

        if total_current > 0:
            dates.append(datetime.now())
            amounts.append(float(total_current))

        # 4. Calculate XIRR
        if len(dates) >= 2 and any(a < 0 for a in amounts) and any(a > 0 for a in amounts):
            rate = xirr(dates, amounts)
            return rate * 100.0  # convert to percentage

        return 0.0
    except Exception as e:
        st.error(f"Error in XIRR calculation: {e}")
        return 0.0



def main():
    st.set_page_config(page_title="Portfolio Analysis", layout="wide")
    st.title("Portfolio Value Analysis Dashboard")

    try:
        df = get_portfolio_data()
        latest_nav = get_latest_nav()
        goal_mappings = get_goal_mappings()
        benchmark_df = get_benchmark_data()

        if df.empty or latest_nav.empty:
            st.warning("No data found. Please ensure portfolio data and NAV data are available.")
            return

        df['date'] = pd.to_datetime(df['date'])
        df = prepare_cashflows(df)
        df = calculate_units(df)

        # Calculate portfolio vs benchmark performance
        portfolio_growth_df, benchmark_growth_df = calculate_portfolio_vs_benchmark(df, latest_nav, goal_mappings, benchmark_df)
        
        # Calculate annual performance
        annual_performance_df = calculate_annual_performance(df, latest_nav, goal_mappings, benchmark_df)

        # Calculate portfolio weights including debt investments
        weights_df = calculate_portfolio_weights(df, latest_nav, goal_mappings)

        # Calculate equity and debt values
        equity_value = weights_df[weights_df['scheme_name'].isin(df['scheme_name'].unique())]['current_value'].sum()
        debt_value = goal_mappings[goal_mappings['investment_type'] == 'Debt']['current_value'].sum()
        total_portfolio_value = equity_value + debt_value

        equity_percent = (equity_value / total_portfolio_value) * 100 if total_portfolio_value > 0 else 0
        debt_percent = (debt_value / total_portfolio_value) * 100 if total_portfolio_value > 0 else 0

        # Calculate total invested amount for equity only
        equity_invested = df[df['transaction_type'].isin(['invest', 'switch_in'])]['amount'].sum()
        
        # Calculate equity XIRR
        equity_xirr = calculate_equity_xirr(df, latest_nav)

        # Display Overall Portfolio Metrics
        st.subheader("Overall Portfolio Metrics")
        
        # Row 1: Current Portfolio Value
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Current Portfolio Value (including Equity and Debt)", format_indian_number(total_portfolio_value))
        with col2:
            st.metric("Total Invested Equity", format_indian_number(equity_invested))
        
        # Row 2: Equity and Debt Values with percentages
        col3, col4 = st.columns(2)
        with col3:
            st.metric("Current Equity Value", f"{format_indian_number(equity_value)} ({equity_percent:.1f}%)")
        with col4:
            st.metric("Current Debt Value", f"{format_indian_number(debt_value)} ({debt_percent:.1f}%)")
        
        # Row 3: Equity XIRR
        # Row 3: Equity XIRR
        col5, _ = st.columns(2)
        with col5:
            st.metric("Equity XIRR", f"{equity_xirr:.1f}%")
        weights_df['Current Value'] = weights_df['current_value'].apply(format_indian_number)
        weights_df['Weight (%)'] = weights_df['weight'].round(2)
        
        # Display formatted columns
        display_metrics = weights_df[['scheme_name', 'Current Value', 'Weight (%)']]
        display_metrics.columns = ['Scheme Name', 'Current Value', 'Weight (%)']
        st.dataframe(display_metrics)

        # Display Portfolio Value vs Benchmark Over Time
        st.subheader("Portfolio Value vs Benchmark Over Time")
        if not portfolio_growth_df.empty and not benchmark_growth_df.empty:
            # Merge the dataframes
            merged_df = pd.merge(portfolio_growth_df, benchmark_growth_df, on='date', how='inner')
            
            # Create the plot
            fig = go.Figure()
            
            # Add portfolio line
            fig.add_trace(go.Scatter(
                x=merged_df['date'],
                y=merged_df['portfolio_value'],
                mode='lines',
                name='Portfolio Value',
                line=dict(color='blue', width=2),
                hovertemplate='Date: %{x}<br>Portfolio Value: ₹%{y:,.0f}<extra></extra>'
            ))
            
            # Add benchmark line
            fig.add_trace(go.Scatter(
                x=merged_df['date'],
                y=merged_df['benchmark_value'],
                mode='lines',
                name='Benchmark Value',
                line=dict(color='red', width=2),
                hovertemplate='Date: %{x}<br>Benchmark Value: ₹%{y:,.0f}<extra></extra>'
            ))
            
            fig.update_layout(
                title='Portfolio Value vs Benchmark Performance Over Time',
                xaxis_title='Date',
                yaxis_title='Value (₹)',
                hovermode='x unified',
                legend=dict(x=0.02, y=0.98),
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show current comparison
            if not merged_df.empty:
                latest_portfolio = merged_df['portfolio_value'].iloc[-1]
                latest_benchmark = merged_df['benchmark_value'].iloc[-1]
                latest_invested = merged_df['benchmark_invested'].iloc[-1]
                
                portfolio_return = ((latest_portfolio - latest_invested) / latest_invested * 100) if latest_invested > 0 else 0
                benchmark_return = ((latest_benchmark - latest_invested) / latest_invested * 100) if latest_invested > 0 else 0
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Portfolio Return", f"{portfolio_return:.1f}%")
                with col2:
                    st.metric("Benchmark Return", f"{benchmark_return:.1f}%")
                with col3:
                    outperformance = portfolio_return - benchmark_return
                    st.metric("Outperformance", f"{outperformance:.1f}%")
        else:
            st.warning("Insufficient data to calculate portfolio vs benchmark comparison.")

        # Display Annual Performance Comparison
        st.subheader("Annual Performance: Portfolio vs Benchmark")
        if not annual_performance_df.empty:
            # Create the plot
            fig = go.Figure()
            
            # Add portfolio return line
            fig.add_trace(go.Scatter(
                x=annual_performance_df['year'],
                y=annual_performance_df['portfolio_return'],
                mode='lines+markers',
                name='Portfolio Return',
                line=dict(color='blue', width=2),
                hovertemplate='Year: %{x}<br>Portfolio Return: %{y:.1f}%<extra></extra>'
            ))
            
            # Add benchmark return line
            fig.add_trace(go.Scatter(
                x=annual_performance_df['year'],
                y=annual_performance_df['benchmark_return'],
                mode='lines+markers',
                name='Benchmark Return',
                line=dict(color='red', width=2),
                hovertemplate='Year: %{x}<br>Benchmark Return: %{y:.1f}%<extra></extra>'
            ))
            
            fig.update_layout(
                title='Annual Returns: Portfolio vs Benchmark',
                xaxis_title='Year',
                yaxis_title='Return (%)',
                hovermode='x unified',
                legend=dict(x=0.02, y=0.98),
                height=400
            )
            
            # Show years as integers on x-axis
            fig.update_xaxes(type='category')
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show annual performance table
            display_annual = annual_performance_df.copy()
            display_annual['Portfolio Value'] = display_annual['portfolio_value'].apply(format_indian_number)
            display_annual['Benchmark Value'] = display_annual['benchmark_value'].apply(format_indian_number)
            display_annual['Invested Amount'] = display_annual['invested_amount'].apply(format_indian_number)
            display_annual['Portfolio Return (%)'] = display_annual['portfolio_return'].round(1)
            display_annual['Benchmark Return (%)'] = display_annual['benchmark_return'].round(1)
            display_annual['Outperformance (%)'] = (display_annual['portfolio_return'] - display_annual['benchmark_return']).round(1)
            
            st.dataframe(display_annual[['year', 'Portfolio Value', 'Benchmark Value', 'Invested Amount', 
                                       'Portfolio Return (%)', 'Benchmark Return (%)', 'Outperformance (%)']])
        else:
            st.warning("Insufficient data to calculate annual performance comparison.")

        # Goal-wise Equity and Debt Split
        st.subheader("Goal-wise Equity and Debt Split")
        
        goals = goal_mappings['goal_name'].unique()
        num_rows = (len(goals) + 1) // 2
        
        for row_idx in range(num_rows):
            col1, col2 = st.columns(2)
            
            with col1:
                if row_idx * 2 < len(goals):
                    goal = goals[row_idx * 2]
                    goal_data = goal_mappings[goal_mappings['goal_name'] == goal]
                    equity_value = goal_data[goal_data['investment_type'] == 'Equity']['current_value'].sum()
                    debt_value = goal_data[goal_data['investment_type'] == 'Debt']['current_value'].sum()
                    total_value = equity_value + debt_value

                    if total_value > 0:
                        equity_percent = (equity_value / total_value) * 100
                        debt_percent = (debt_value / total_value) * 100

                        fig = px.pie(
                            values=[equity_value, debt_value],
                            names=['Equity', 'Debt'],
                            title=f"{goal} - Equity vs Debt Split",
                            height=300
                        )
                        fig.update_traces(textinfo='percent+label', pull=[0.1, 0])
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.write(f"**Total Value:** {format_indian_number(total_value)}")
                        st.write(f"**Equity:** {equity_percent:.1f}% ({format_indian_number(equity_value)})")
                        st.write(f"**Debt:** {debt_percent:.1f}% ({format_indian_number(debt_value)})")

            with col2:
                if row_idx * 2 + 1 < len(goals):
                    goal = goals[row_idx * 2 + 1]
                    goal_data = goal_mappings[goal_mappings['goal_name'] == goal]
                    equity_value = goal_data[goal_data['investment_type'] == 'Equity']['current_value'].sum()
                    debt_value = goal_data[goal_data['investment_type'] == 'Debt']['current_value'].sum()
                    total_value = equity_value + debt_value

                    if total_value > 0:
                        equity_percent = (equity_value / total_value) * 100
                        debt_percent = (debt_value / total_value) * 100

                        fig = px.pie(
                            values=[equity_value, debt_value],
                            names=['Equity', 'Debt'],
                            title=f"{goal} - Equity vs Debt Split",
                            height=300
                        )
                        fig.update_traces(textinfo='percent+label', pull=[0.1, 0])
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.write(f"**Total Value:** {format_indian_number(total_value)}")
                        st.write(f"**Equity:** {equity_percent:.1f}% ({format_indian_number(equity_value)})")
                        st.write(f"**Debt:** {debt_percent:.1f}% ({format_indian_number(debt_value)})")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please check your database connection and data integrity.")

if __name__ == "__main__":
    main()