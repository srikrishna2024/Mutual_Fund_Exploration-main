import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import psycopg
import plotly.express as px
import plotly.graph_objects as go

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

def get_fund_list():
    """Get list of all mutual funds from the database"""
    with connect_to_db() as conn:
        query = """
            SELECT DISTINCT code, scheme_name
            FROM mutual_fund_nav
            ORDER BY scheme_name
        """
        return pd.read_sql(query, conn)

def get_fund_nav_history(fund_code):
    """Get NAV history for a specific fund"""
    with connect_to_db() as conn:
        query = """
            SELECT code, scheme_name, nav as date, value
            FROM mutual_fund_nav
            WHERE code = %s
            ORDER BY nav
        """
        df = pd.read_sql(query, conn, params=(fund_code,))
        # Convert date column to datetime
        df['date'] = pd.to_datetime(df['date'])
        return df

def identify_high_impact_periods(timeseries, window_size=30, threshold_percentile=90):
    """
    Identify periods of significant market movement that had high impact on returns
    
    Parameters:
    - timeseries: DataFrame containing NAV and investment values
    - window_size: Size of the rolling window (in days) to look for impact periods
    - threshold_percentile: Percentile threshold to consider a period as high impact
    
    Returns:
    - List of dictionaries containing high impact periods info
    """
    if len(timeseries) < window_size + 1:
        return []
    
    # Calculate daily returns
    timeseries['daily_return'] = timeseries['value'].pct_change() * 100
    
    # Calculate rolling returns for different windows
    windows = [window_size, 60, 90]  # 30, 60, 90 days windows
    
    high_impact_periods = []
    
    for w in windows:
        if len(timeseries) < w + 1:
            continue
            
        # Calculate rolling return for each window
        timeseries[f'rolling_{w}d_return'] = (
            timeseries['value'].pct_change(periods=w) * 100
        )
        
        # Calculate rolling volatility 
        timeseries[f'rolling_{w}d_volatility'] = (
            timeseries['daily_return'].rolling(window=w).std()
        )
        
        # Calculate impact score (magnitude of return / time period)
        timeseries[f'impact_score_{w}d'] = (
            timeseries[f'rolling_{w}d_return'].abs() / w
        )
        
        # Identify high impact periods
        if not timeseries[f'impact_score_{w}d'].dropna().empty:
            threshold = np.percentile(
                timeseries[f'impact_score_{w}d'].dropna(), 
                threshold_percentile
            )
            
            # Find periods above threshold
            high_impact_mask = timeseries[f'impact_score_{w}d'] > threshold
            
            if high_impact_mask.sum() > 0:
                # Convert index to list to avoid NumPy integer issues
                high_impact_indices = timeseries[high_impact_mask].index.tolist()
                
                # Get the start of each period
                for i in range(len(high_impact_indices)):
                    idx = high_impact_indices[i]
                    # Use the date column directly as it's already datetime
                    current_day = timeseries.loc[idx, 'date']
                    prev_day = timeseries.loc[high_impact_indices[i-1], 'date'] if i > 0 else None
                    
                    # Check if this is a new period or continue existing one
                    is_new_period = (i == 0) or (prev_day is None) or ((current_day - prev_day).days > w/2)
                    
                    if is_new_period:  # New period if gap > half window
                        start_idx = max(0, timeseries.index.get_loc(idx) - w + 1)
                        end_idx = timeseries.index.get_loc(idx)
                        
                        start_date = timeseries.iloc[start_idx]['date']
                        end_date = timeseries.iloc[end_idx]['date']
                        
                        period_return = timeseries.iloc[end_idx]['value'] / timeseries.iloc[start_idx]['value'] - 1
                        
                        # Ensure date objects for correct date arithmetic
                        date_diff = max(1, (end_date - start_date).days)
                        
                        # Only add if the return is significant
                        if abs(period_return) > 0.05:  # 5% minimum threshold
                            high_impact_periods.append({
                                'start_date': start_date,
                                'end_date': end_date,
                                'period_days': date_diff,
                                'return': period_return * 100,
                                'annualized_return': ((1 + period_return) ** (365 / date_diff) - 1) * 100,
                                'volatility': timeseries.iloc[end_idx][f'rolling_{w}d_volatility']
                            })
    
    # Remove overlapping periods, keeping the ones with highest impact
    if high_impact_periods:
        high_impact_periods.sort(key=lambda x: abs(x['return']), reverse=True)
        
        # Filter out overlapping periods
        filtered_periods = []
        for period in high_impact_periods:
            overlap = False
            for filtered in filtered_periods:
                # Check if periods overlap significantly
                p_start = period['start_date']
                p_end = period['end_date']
                f_start = filtered['start_date']
                f_end = filtered['end_date']
                
                if (max(p_start, f_start) <= min(p_end, f_end)):
                    days_overlap = (min(p_end, f_end) - max(p_start, f_start)).days
                    if days_overlap > min(period['period_days'], filtered['period_days']) * 0.5:
                        overlap = True
                        break
            
            if not overlap:
                filtered_periods.append(period)
        
        # Sort by date
        filtered_periods.sort(key=lambda x: x['start_date'])
        return filtered_periods
    
    return []

def calculate_investment_growth(nav_history, investment_amount, start_date):
    """
    Calculate the growth of an investment from start_date to the present
    """
    # Ensure nav_history is sorted by date
    nav_history = nav_history.sort_values('date')
    
    # Convert start_date to pd.Timestamp for comparison
    start_date = pd.Timestamp(start_date)
    
    # Find NAV on or after the start date
    start_nav_data = nav_history[nav_history['date'] >= start_date].iloc[0]
    start_nav = start_nav_data['value']
    actual_start_date = start_nav_data['date']
    
    # Find the latest NAV
    end_nav_data = nav_history.iloc[-1]
    end_nav = end_nav_data['value']
    end_date = end_nav_data['date']
    
    # Calculate units purchased
    units = investment_amount / start_nav
    
    # Calculate current value
    current_value = units * end_nav
    
    # Calculate growth percentage
    growth_percentage = ((current_value - investment_amount) / investment_amount) * 100
    
    # Create a daily NAV timeseries for plotting
    investment_timeseries = nav_history[nav_history['date'] >= actual_start_date].copy()
    investment_timeseries['units'] = units
    investment_timeseries['investment_value'] = investment_timeseries['value'] * units
    
    # Calculate daily returns
    investment_timeseries['daily_return'] = investment_timeseries['value'].pct_change() * 100
    
    # Identify high-impact periods
    high_impact_periods = identify_high_impact_periods(investment_timeseries)
    
    return {
        'start_date': actual_start_date,
        'end_date': end_date,
        'investment_amount': investment_amount,
        'start_nav': start_nav,
        'end_nav': end_nav,
        'units': units,
        'current_value': current_value,
        'growth_percentage': growth_percentage,
        'investment_timeseries': investment_timeseries,
        'high_impact_periods': high_impact_periods
    }

def main():
    st.set_page_config(page_title="Mutual Fund Comparison", layout="wide")
    st.title("Mutual Fund Comparison")
    
    # Load fund list from database
    fund_list = get_fund_list()
    
    if fund_list.empty:
        st.warning("No mutual fund data found in the database.")
        return
    
    # Create sidebar for input controls
    st.sidebar.header("Fund Selection")
    
    # Allow user to select up to 3 funds
    selected_funds = st.sidebar.multiselect(
        "Select Mutual Funds (up to 3)",
        options=fund_list['scheme_name'].tolist(),
        default=[],
        max_selections=3
    )
    
    if not selected_funds:
        st.warning("Please select at least one fund.")
        return
    
    # Get the codes for the selected funds
    selected_fund_codes = fund_list[fund_list['scheme_name'].isin(selected_funds)]['code'].tolist()
    
    # Load NAV history for the selected funds
    nav_histories = {}
    for fund_code in selected_fund_codes:
        nav_history = get_fund_nav_history(fund_code)
        nav_histories[fund_code] = nav_history
    
    # Find the oldest available date among all selected funds
    oldest_date = min([nav_history['date'].min() for nav_history in nav_histories.values()])
    
    # Allow user to select a start date (restricted to oldest available date)
    st.sidebar.header("Investment Parameters")
    start_date = st.sidebar.date_input(
        "Start Date",
        value=oldest_date.date(),  # Convert to date for date_input
        min_value=oldest_date.date(),
        max_value=datetime.today().date()
    )
    
    # Allow user to input a single investment amount for all funds
    investment_amount = st.sidebar.number_input(
        "Investment Amount (₹)",
        min_value=1000,
        max_value=10000000,
        value=10000,
        step=1000
    )
    
    # Add Analyze button
    analyze_button = st.sidebar.button("Analyze", type="primary")
    
    if analyze_button:
        # Calculate investment growth for each fund
        results = {}
        for fund_name, fund_code in zip(selected_funds, selected_fund_codes):
            result = calculate_investment_growth(nav_histories[fund_code], investment_amount, start_date)
            results[fund_name] = result
        
        # Plot the investment growth over time
        st.subheader("Investment Growth Over Time")
        
        # Create figure
        fig = go.Figure()
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Different colors for each fund
        for i, (fund_name, result) in enumerate(results.items()):
            timeseries = result['investment_timeseries']
            fig.add_trace(go.Scatter(
                x=timeseries['date'],
                y=timeseries['investment_value'],
                mode='lines',
                name=fund_name,
                line=dict(color=colors[i], width=2)
            ))
        
        # Highlight high-impact periods
        all_impact_periods = []
        for fund_name, result in results.items():
            impact_periods = result['high_impact_periods']
            for period in impact_periods:
                all_impact_periods.append({
                    'start_date': period['start_date'],
                    'end_date': period['end_date'], 
                    'return': period['return'],
                    'color': colors[selected_funds.index(fund_name)],
                    'fund_name': fund_name
                })
        
        # Sort and deduplicate impact periods across all funds
        if all_impact_periods:
            all_impact_periods.sort(key=lambda x: (x['start_date'], abs(x['return'])), reverse=True)
            
            # Keep only the most significant periods
            unique_periods = []
            for period in all_impact_periods:
                # Check if this period overlaps with any existing period
                overlaps = False
                for up in unique_periods:
                    # If periods overlap significantly
                    p_start = period['start_date']
                    p_end = period['end_date']
                    up_start = up['start_date']
                    up_end = up['end_date']
                    
                    if (max(p_start, up_start) <= min(p_end, up_end)):
                        days_overlap = (min(p_end, up_end) - max(p_start, up_start)).days
                        total_days = (p_end - p_start).days
                        if days_overlap > total_days * 0.5:
                            overlaps = True
                            break
                
                if not overlaps:
                    unique_periods.append(period)
            
            # Get the top 5 most impactful periods for highlighting
            unique_periods.sort(key=lambda x: abs(x['return']), reverse=True)
            top_periods = unique_periods[:5]
            
            # Add highlighting for high impact periods
            for period in top_periods:
                # Add a semi-transparent rectangle to highlight the period
                fig.add_shape(
                    type="rect",
                    x0=period['start_date'],
                    y0=0,
                    x1=period['end_date'],
                    y1=max([r['investment_timeseries']['investment_value'].max() for r in results.values()]) * 1.05,
                    fillcolor="rgba(255, 0, 0, 0.1)" if period['return'] < 0 else "rgba(0, 255, 0, 0.1)",
                    line=dict(width=0),
                    layer="below"
                )
                
                # Add annotation for the period
                fig.add_annotation(
                    x=period['start_date'] + (period['end_date'] - period['start_date'])/2,
                    y=max([r['investment_timeseries']['investment_value'].max() for r in results.values()]),
                    text=f"{period['return']:.1f}% in {(period['end_date'] - period['start_date']).days} days",
                    showarrow=True,
                    arrowhead=1,
                    bgcolor="rgba(255,255,255,0.8)"
                )
        
        # Update layout
        fig.update_layout(
            title=f"Growth of Investments in Selected Funds",
            xaxis_title="Date",
            yaxis_title="Value (₹)",
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display high-impact periods table
        if all_impact_periods:
            st.subheader("High-Impact Market Periods")
            st.write("These periods significantly affected the investment performance:")
            
            impact_df = pd.DataFrame([
                {
                    'Start Date': p['start_date'].date(),  # Convert to date for display
                    'End Date': p['end_date'].date(),  # Convert to date for display
                    'Duration (Days)': (p['end_date'] - p['start_date']).days,
                    'Return (%)': round(p['return'], 2),
                    'Annualized (%)': round(((1 + p['return']/100) ** (365 / max(1, (p['end_date'] - p['start_date']).days)) - 1) * 100, 2),
                    'Fund': p['fund_name'],
                    'Impact': 'High Negative' if p['return'] < -15 else 
                             'Negative' if p['return'] < 0 else
                             'High Positive' if p['return'] > 15 else 'Positive'
                } 
                for p in unique_periods[:10]  # Show top 10 impact periods
            ])
            
            # Apply styling
            st.dataframe(
                impact_df.style.apply(
                    lambda x: ['background-color: rgba(255,0,0,0.1)' if x['Return (%)'] < 0 else 'background-color: rgba(0,255,0,0.1)' for _ in x],
                    axis=1
                ),
                use_container_width=True,
                hide_index=True
            )
            
            st.info("These periods represent market movements with outsized impact on returns. " +
                    "Negative periods are highlighted in red, positive in green.")

if __name__ == "__main__":
    main()