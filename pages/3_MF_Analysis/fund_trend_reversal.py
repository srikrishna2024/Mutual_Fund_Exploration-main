import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import psycopg
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import newton

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
        return pd.read_sql(query, conn, params=(fund_code,))

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
                    # Convert to datetime objects to ensure proper date arithmetic
                    current_day = pd.to_datetime(timeseries.loc[idx, 'date'])
                    prev_day = pd.to_datetime(timeseries.loc[high_impact_indices[i-1], 'date']) if i > 0 else None
                    
                    # Check if this is a new period or continue existing one
                    is_new_period = (i == 0) or (prev_day is None) or ((current_day - prev_day).days > w/2)
                    
                    if is_new_period:  # New period if gap > half window
                        start_idx = max(0, timeseries.index.get_loc(idx) - w + 1)
                        end_idx = timeseries.index.get_loc(idx)
                        
                        start_date = pd.to_datetime(timeseries.iloc[start_idx]['date'])
                        end_date = pd.to_datetime(timeseries.iloc[end_idx]['date'])
                        
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
                # Ensure proper datetime objects
                p_start = pd.to_datetime(period['start_date'])
                p_end = pd.to_datetime(period['end_date'])
                f_start = pd.to_datetime(filtered['start_date'])
                f_end = pd.to_datetime(filtered['end_date'])
                
                if (max(p_start, f_start) <= min(p_end, f_end)):
                    days_overlap = (min(p_end, f_end) - max(p_start, f_start)).days
                    if days_overlap > min(period['period_days'], filtered['period_days']) * 0.5:
                        overlap = True
                        break
            
            if not overlap:
                filtered_periods.append(period)
        
        # Sort by date
        filtered_periods.sort(key=lambda x: pd.to_datetime(x['start_date']))
        return filtered_periods
    
    return []

def detect_potential_high_impact(timeseries, window_size=30, threshold_percentile=90):
    """
    Detect if a potential high-impact period is building up based on recent trends
    
    Parameters:
    - timeseries: DataFrame containing NAV and investment values
    - window_size: Size of the rolling window (in days) to look for impact periods
    - threshold_percentile: Percentile threshold to consider a period as high impact
    
    Returns:
    - Dictionary containing potential high impact period info
    """
    if len(timeseries) < window_size + 1:
        return None
    
    # Calculate daily returns
    timeseries['daily_return'] = timeseries['value'].pct_change() * 100
    
    # Calculate rolling return for the window
    timeseries[f'rolling_{window_size}d_return'] = (
        timeseries['value'].pct_change(periods=window_size) * 100
    )
    
    # Calculate rolling volatility 
    timeseries[f'rolling_{window_size}d_volatility'] = (
        timeseries['daily_return'].rolling(window=window_size).std()
    )
    
    # Calculate impact score (magnitude of return / time period)
    timeseries[f'impact_score_{window_size}d'] = (
        timeseries[f'rolling_{window_size}d_return'].abs() / window_size
    )
    
    # Identify if the latest impact score is above the threshold
    if not timeseries[f'impact_score_{window_size}d'].dropna().empty:
        threshold = np.percentile(
            timeseries[f'impact_score_{window_size}d'].dropna(), 
            threshold_percentile
        )
        
        latest_impact_score = timeseries[f'impact_score_{window_size}d'].iloc[-1]
        
        if latest_impact_score > threshold:
            return {
                'start_date': timeseries['date'].iloc[-window_size],
                'end_date': timeseries['date'].iloc[-1],
                'period_days': window_size,
                'return': timeseries[f'rolling_{window_size}d_return'].iloc[-1],
                'volatility': timeseries[f'rolling_{window_size}d_volatility'].iloc[-1],
                'impact_score': latest_impact_score
            }
    
    return None

def detect_trend_reversal(timeseries, window_size=30):
    """
    Detect trend reversals (e.g., from negative to positive or sideways movements)
    
    Parameters:
    - timeseries: DataFrame containing NAV and investment values
    - window_size: Size of the rolling window (in days) to look for trend reversals
    
    Returns:
    - Dictionary containing trend reversal info
    """
    if len(timeseries) < window_size + 1:
        return None
    
    # Calculate rolling return for the window
    timeseries[f'rolling_{window_size}d_return'] = (
        timeseries['value'].pct_change(periods=window_size) * 100
    )
    
    # Check if the trend has reversed
    latest_return = timeseries[f'rolling_{window_size}d_return'].iloc[-1]
    prev_return = timeseries[f'rolling_{window_size}d_return'].iloc[-2]
    
    if latest_return > 0 and prev_return < 0:
        return {
            'start_date': timeseries['date'].iloc[-window_size],
            'end_date': timeseries['date'].iloc[-1],
            'period_days': window_size,
            'return': latest_return,
            'trend_change': 'Negative to Positive'
        }
    elif abs(latest_return) < 5 and abs(prev_return) >= 5:  # Sideways movement
        return {
            'start_date': timeseries['date'].iloc[-window_size],
            'end_date': timeseries['date'].iloc[-1],
            'period_days': window_size,
            'return': latest_return,
            'trend_change': 'Negative to Sideways'
        }
    
    return None

def main():
    st.set_page_config(page_title="Mutual Fund NAV Analysis", layout="wide")
    st.title("Mutual Fund NAV Analysis")
    
    # Load fund list from database
    fund_list = get_fund_list()
    
    if fund_list.empty:
        st.warning("No mutual fund data found in the database.")
        return
    
    # Create sidebar for input controls
    st.sidebar.header("Fund Selection")
    
    # Fund selection
    selected_fund = st.sidebar.selectbox(
        "Select Mutual Fund",
        options=fund_list['scheme_name'].tolist(),
        index=0
    )
    
    # Get the code for the selected fund
    selected_fund_code = fund_list[fund_list['scheme_name'] == selected_fund]['code'].iloc[0]
    
    # Load NAV history for the selected fund
    nav_history = get_fund_nav_history(selected_fund_code)
    
    if nav_history.empty:
        st.warning(f"No NAV data found for {selected_fund}")
        return
    
    # Format dates
    nav_history['date'] = pd.to_datetime(nav_history['date'])
    
    # Add Analyze button
    analyze_button = st.sidebar.button("Analyze", type="primary")
    
    if analyze_button:
        # Identify historical high-impact periods
        high_impact_periods = identify_high_impact_periods(nav_history)
        
        # Detect potential high-impact period
        potential_high_impact = detect_potential_high_impact(nav_history)
        
        # Detect trend reversals
        trend_reversal = detect_trend_reversal(nav_history)
        
        # Display the selected fund details
        st.header(f"Analysis for {selected_fund}")
        st.write(f"Data available from {nav_history['date'].min().date()} to {nav_history['date'].max().date()}")
        
        # Plot the NAV history with high-impact periods
        st.subheader("NAV History with High-Impact Periods")
        
        # Create NAV chart
        nav_fig = px.line(
            nav_history, 
            x='date', 
            y='value',
            title=f"NAV History for {selected_fund}"
        )
        
        # Highlight historical high-impact periods
        if high_impact_periods:
            for period in high_impact_periods:
                # Add a semi-transparent rectangle to highlight the period
                nav_fig.add_shape(
                    type="rect",
                    x0=pd.to_datetime(period['start_date']),
                    y0=0,
                    x1=pd.to_datetime(period['end_date']),
                    y1=nav_history['value'].max() * 1.05,
                    fillcolor="rgba(255, 0, 0, 0.1)" if period['return'] < 0 else "rgba(0, 255, 0, 0.1)",
                    line=dict(width=0),
                    layer="below"
                )
        
        # Highlight potential high-impact period
        if potential_high_impact:
            nav_fig.add_shape(
                type="rect",
                x0=pd.to_datetime(potential_high_impact['start_date']),
                y0=0,
                x1=pd.to_datetime(potential_high_impact['end_date']),
                y1=nav_history['value'].max() * 1.05,
                fillcolor="rgba(255, 165, 0, 0.2)",
                line=dict(width=0),
                layer="below"
            )
            
            nav_fig.add_annotation(
                x=pd.to_datetime(potential_high_impact['start_date']) + (pd.to_datetime(potential_high_impact['end_date']) - pd.to_datetime(potential_high_impact['start_date']))/2,
                y=nav_history['value'].max(),
                text=f"Potential High Impact: {potential_high_impact['return']:.1f}% in {potential_high_impact['period_days']} days",
                showarrow=True,
                arrowhead=1,
                bgcolor="rgba(255,255,255,0.8)"
            )
        
        # Highlight trend reversals
        if trend_reversal:
            nav_fig.add_shape(
                type="rect",
                x0=pd.to_datetime(trend_reversal['start_date']),
                y0=0,
                x1=pd.to_datetime(trend_reversal['end_date']),
                y1=nav_history['value'].max() * 1.05,
                fillcolor="rgba(0, 0, 255, 0.1)",
                line=dict(width=0),
                layer="below"
            )
            
            nav_fig.add_annotation(
                x=pd.to_datetime(trend_reversal['start_date']) + (pd.to_datetime(trend_reversal['end_date']) - pd.to_datetime(trend_reversal['start_date']))/2,
                y=nav_history['value'].max(),
                text=f"Trend Reversal: {trend_reversal['trend_change']}",
                showarrow=True,
                arrowhead=1,
                bgcolor="rgba(255,255,255,0.8)"
            )
        
        nav_fig.update_layout(
            xaxis_title="Date",
            yaxis_title="NAV (₹)",
            hovermode="x unified"
        )
        
        st.plotly_chart(nav_fig, use_container_width=True)
        
        # Display high-impact periods table
        if high_impact_periods:
            st.subheader("Historical High-Impact Periods")
            st.write("These periods significantly affected the NAV performance:")
            
            impact_df = pd.DataFrame([
                {
                    'Start Date': pd.to_datetime(p['start_date']).date(),
                    'End Date': pd.to_datetime(p['end_date']).date(),
                    'Duration (Days)': (pd.to_datetime(p['end_date']) - pd.to_datetime(p['start_date'])).days,
                    'Return (%)': round(p['return'], 2),
                    'Annualized (%)': round(((1 + p['return']/100) ** (365 / max(1, (pd.to_datetime(p['end_date']) - pd.to_datetime(p['start_date'])).days)) - 1) * 100, 2),
                    'Impact': 'High Negative' if p['return'] < -15 else 
                             'Negative' if p['return'] < 0 else
                             'High Positive' if p['return'] > 15 else 'Positive'
                } 
                for p in high_impact_periods
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
            
            st.info("These periods represent market movements with outsized impact on NAV. " +
                    "Negative periods are highlighted in red, positive in green.")
        
        # Display potential high-impact period
        if potential_high_impact:
            st.subheader("Potential High-Impact Period")
            st.write("A potential high-impact period may be building up based on recent trends:")
            
            st.write(f"**Start Date:** {potential_high_impact['start_date'].date()}")
            st.write(f"**End Date:** {potential_high_impact['end_date'].date()}")
            st.write(f"**Duration:** {potential_high_impact['period_days']} days")
            st.write(f"**Return:** {potential_high_impact['return']:.2f}%")
            st.write(f"**Volatility:** {potential_high_impact['volatility']:.2f}")
            st.write(f"**Impact Score:** {potential_high_impact['impact_score']:.2f}")
            
            st.warning("This is a potential high-impact period based on recent trends. " +
                       "It may not necessarily result in a significant impact, but it warrants attention.")
        
        # Display trend reversal
        if trend_reversal:
            st.subheader("Trend Reversal Detected")
            st.write(f"A trend reversal has been detected: **{trend_reversal['trend_change']}**")
            st.write(f"**Start Date:** {trend_reversal['start_date'].date()}")
            st.write(f"**End Date:** {trend_reversal['end_date'].date()}")
            st.write(f"**Duration:** {trend_reversal['period_days']} days")
            st.write(f"**Return:** {trend_reversal['return']:.2f}%")

if __name__ == "__main__":
    main()