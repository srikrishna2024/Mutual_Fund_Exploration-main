import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import psycopg
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

def get_fund_categories():
    """Retrieve all mutual fund categories"""
    with connect_to_db() as conn:
        query = """
            SELECT DISTINCT scheme_category 
            FROM mutual_fund_master_data 
            ORDER BY scheme_category
        """
        return [row[0] for row in conn.execute(query).fetchall()]

def get_funds_in_category(category):
    """Retrieve all funds in a specific category"""
    with connect_to_db() as conn:
        query = """
            SELECT code, scheme_name 
            FROM mutual_fund_master_data 
            WHERE scheme_category = %s
            ORDER BY scheme_name
        """
        return pd.read_sql(query, conn, params=(category,))

def get_fund_nav_history(code):
    """Retrieve historical NAV data for a specific fund"""
    with connect_to_db() as conn:
        query = """
            SELECT nav as date, value::float as nav_value
            FROM mutual_fund_nav
            WHERE code = %s AND value::float > 0
            ORDER BY nav
        """
        df = pd.read_sql(query, conn, params=(code,))
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            return df.set_index('date')['nav_value']
        return pd.Series(dtype=float)

def get_benchmark_data():
    """Retrieve benchmark data"""
    with connect_to_db() as conn:
        query = """
            SELECT date, price as nav_value
            FROM benchmark
            ORDER BY date
        """
        df = pd.read_sql(query, conn)
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            return df.set_index('date')['nav_value']
        return pd.Series(dtype=float)

def calculate_rolling_returns(series, window_days, step_days=30):
    """Calculate rolling returns for a given window with monthly steps"""
    if len(series) < window_days:
        return None
    
    returns = []
    max_index = len(series) - window_days
    
    for i in range(0, max_index + 1, step_days):
        start_value = series.iloc[i]
        end_value = series.iloc[i + window_days - 1]
        period_return = (end_value / start_value - 1) * 100
        returns.append({
            'start_date': series.index[i].strftime('%Y-%m-%d'),
            'end_date': series.index[i + window_days - 1].strftime('%Y-%m-%d'),
            'return': period_return
        })
    
    return pd.DataFrame(returns)

def align_series_data(fund_nav, benchmark_nav, start_date, end_date):
    """Align fund and benchmark data to common date range"""
    # Filter both series to the same date range
    fund_filtered = fund_nav[(fund_nav.index >= start_date) & (fund_nav.index <= end_date)]
    benchmark_filtered = benchmark_nav[(benchmark_nav.index >= start_date) & (benchmark_nav.index <= end_date)]
    
    if fund_filtered.empty or benchmark_filtered.empty:
        return None, None
    
    # Find common date range
    common_start = max(fund_filtered.index[0], benchmark_filtered.index[0])
    common_end = min(fund_filtered.index[-1], benchmark_filtered.index[-1])
    
    # Filter to common range
    fund_common = fund_filtered[(fund_filtered.index >= common_start) & (fund_filtered.index <= common_end)]
    benchmark_common = benchmark_filtered[(benchmark_filtered.index >= common_start) & (benchmark_filtered.index <= common_end)]
    
    return fund_common, benchmark_common

def calculate_risk_metrics(series):
    """Calculate risk metrics including max drawdown and recovery time"""
    if len(series) < 30:  # Need at least 30 days of data
        return None, None, None
    
    try:
        # Calculate daily returns and handle any infinite or NaN values
        daily_returns = series.pct_change().fillna(0)
        daily_returns = daily_returns.replace([np.inf, -np.inf], 0)
        
        # Calculate cumulative returns
        cumulative_returns = (1 + daily_returns).cumprod()
        
        # Calculate rolling maximum (peak)
        peak = cumulative_returns.expanding(min_periods=1).max()
        
        # Calculate drawdown
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = drawdown.min() * 100
        
        # Find the date of maximum drawdown
        trough_date = drawdown.idxmin()
        recovery_days = None
        recovery_status = "Not Recovered"
        
        if pd.notnull(trough_date):
            # Get the peak value at the time of maximum drawdown
            recovery_level = peak.loc[trough_date]
            
            # Look for recovery after the trough date
            recovery_data = cumulative_returns[trough_date:]
            recovery_candidates = recovery_data[recovery_data >= recovery_level]
            
            if not recovery_candidates.empty:
                recovery_date = recovery_candidates.index[0]
                recovery_days = (recovery_date - trough_date).days
                recovery_status = f"Recovered ({recovery_days} days)"
            else:
                # Check partial recovery status
                current_level = cumulative_returns.iloc[-1]
                trough_level = cumulative_returns.loc[trough_date]
                
                if current_level > trough_level:
                    recovery_percentage = ((current_level - trough_level) / 
                                         (recovery_level - trough_level)) * 100
                    recovery_percentage = min(recovery_percentage, 99.9)  # Cap at 99.9%
                    recovery_status = f"Partial ({recovery_percentage:.1f}%)"
                else:
                    recovery_status = "Still Declining"
        
        return max_drawdown, recovery_days, recovery_status
    
    except Exception as e:
        print(f"Error in risk metrics calculation: {str(e)}")
        return None, None, "Error"

def analyze_fund_period(fund_code, fund_name, benchmark_nav, period_years):
    """Analyze a single fund against benchmark for selected period"""
    try:
        # Get fund NAV history
        fund_nav = get_fund_nav_history(fund_code)
        
        if fund_nav.empty:
            print(f"No NAV data for {fund_name}")
            return None
        
        # Calculate end date (most recent date)
        end_date = fund_nav.index[-1]
        
        # Calculate start date based on selected period
        if period_years == 'YTD':
            start_date = datetime(end_date.year, 1, 1)
        else:
            years = int(period_years.replace('Y', ''))
            start_date = end_date - timedelta(days=years*365)
        
        # Align fund and benchmark data to common date range
        fund_nav_aligned, benchmark_nav_aligned = align_series_data(fund_nav, benchmark_nav, start_date, end_date)
        
        if fund_nav_aligned is None or benchmark_nav_aligned is None:
            print(f"No aligned data for {fund_name} in period {period_years}")
            return None
        
        # Calculate appropriate window size
        available_days = len(fund_nav_aligned)
        
        if period_years == 'YTD':
            # For YTD, use 30-day rolling windows
            window_days = min(30, available_days // 3)
        else:
            years = int(period_years.replace('Y', ''))
            # Use smaller windows for shorter periods, ensure we have at least 3 windows
            if years == 1:
                window_days = min(90, available_days // 3)  # 3-month windows for 1Y
            elif years <= 3:
                window_days = min(180, available_days // 3)  # 6-month windows for 3Y
            else:
                window_days = min(365, available_days // 3)  # 1-year windows for 5Y+
        
        # Ensure minimum window size
        window_days = max(window_days, 30)
        
        if window_days > available_days:
            print(f"Insufficient data for {fund_name}: need {window_days} days, have {available_days}")
            return None
        
        # Use smaller step size for more data points
        step_days = max(7, window_days // 6)  # Weekly steps or window/6
        
        # Calculate rolling returns for fund and benchmark using aligned data
        fund_rolling = calculate_rolling_returns(fund_nav_aligned, window_days, step_days)
        benchmark_rolling = calculate_rolling_returns(benchmark_nav_aligned, window_days, step_days)
        
        if fund_rolling is None or benchmark_rolling is None:
            print(f"Could not calculate rolling returns for {fund_name}")
            return None
        
        # Since we're using aligned data and same parameters, the periods should match exactly
        if len(fund_rolling) != len(benchmark_rolling):
            # Take the minimum to ensure alignment
            min_periods = min(len(fund_rolling), len(benchmark_rolling))
            fund_rolling = fund_rolling.head(min_periods)
            benchmark_rolling = benchmark_rolling.head(min_periods)
        
        # Calculate consistency score using direct comparison (since data is aligned)
        fund_returns = fund_rolling['return'].values
        benchmark_returns = benchmark_rolling['return'].values
        
        beat_count = (fund_returns > benchmark_returns).sum()
        consistency_score = (beat_count / len(fund_returns)) * 100
        
        # Calculate risk metrics using aligned fund data
        max_drawdown, recovery_days, recovery_status = calculate_risk_metrics(fund_nav_aligned)
        
        # Calculate additional metrics
        total_return = ((fund_nav_aligned.iloc[-1] / fund_nav_aligned.iloc[0]) - 1) * 100
        volatility = fund_nav_aligned.pct_change().std() * np.sqrt(252) * 100  # Annualized volatility
        
        return {
            'fund_name': fund_name,
            'fund_code': fund_code,
            'period': period_years,
            'start_date': fund_nav_aligned.index[0],
            'end_date': fund_nav_aligned.index[-1],
            'consistency_score': consistency_score,
            'total_return': total_return,
            'annualized_volatility': volatility,
            'max_drawdown': max_drawdown,
            'recovery_days': recovery_days,
            'recovery_status': recovery_status,
            'data_points': len(fund_returns),
            'available_days': available_days
        }
    
    except Exception as e:
        print(f"Error analyzing {fund_name} ({fund_code}): {str(e)}")
        return None

def main():
    st.set_page_config(page_title="Mutual Fund Consistency Analysis", layout="wide")
    st.title("Mutual Fund Consistency Analysis")
    
    # Get fund categories
    categories = get_fund_categories()
    if not categories:
        st.error("No fund categories found in database")
        return
    
    # User inputs
    col1, col2 = st.columns(2)
    with col1:
        selected_category = st.selectbox("Select Fund Category", categories)
    with col2:
        period_options = ['YTD', '1Y', '3Y', '5Y', '10Y']
        selected_period = st.selectbox("Select Analysis Period", period_options)
    
    # Additional options
    with st.expander("Advanced Options"):
        show_debug = st.checkbox("Show Debug Information", False)
        min_data_points = st.slider("Minimum Data Points Required", 5, 50, 10)
    
    # Get benchmark data
    benchmark_nav = get_benchmark_data()
    if benchmark_nav.empty:
        st.error("No benchmark data available")
        return
    
    if st.button("Analyze Funds"):
        with st.spinner(f"Analyzing funds in {selected_category} category..."):
            # Get all funds in selected category
            funds = get_funds_in_category(selected_category)
            if funds.empty:
                st.warning(f"No funds found in {selected_category} category")
                return
            
            # Analyze each fund against benchmark
            results = []
            failed_funds = []
            progress_bar = st.progress(0)
            total_funds = len(funds)
            
            for i, (code, name) in enumerate(zip(funds['code'], funds['scheme_name'])):
                analysis = analyze_fund_period(code, name, benchmark_nav, selected_period)
                if analysis and analysis['data_points'] >= min_data_points:
                    results.append(analysis)
                else:
                    failed_funds.append({'code': code, 'name': name, 'reason': 'Insufficient data or analysis failed'})
                progress_bar.progress((i + 1) / total_funds)
            
            if not results:
                st.warning("No valid analysis results for any funds in this category")
                if show_debug and failed_funds:
                    st.subheader("Failed Fund Analysis")
                    failed_df = pd.DataFrame(failed_funds)
                    st.dataframe(failed_df)
                return
            
            # Create results dataframe
            results_df = pd.DataFrame([{
                'Fund Name': r['fund_name'][:50] + '...' if len(r['fund_name']) > 50 else r['fund_name'],
                'Fund Code': r['fund_code'],
                'Period': r['period'],
                'Total Return (%)': f"{r['total_return']:.2f}%",
                'Consistency Score (%)': f"{r['consistency_score']:.1f}%",
                'Volatility (%)': f"{r['annualized_volatility']:.2f}%",
                'Max Drawdown (%)': f"{r['max_drawdown']:.2f}%" if pd.notnull(r['max_drawdown']) else "N/A",
                'Recovery Days': f"{r['recovery_days']:.0f}" if pd.notnull(r['recovery_days']) else "N/A",
                'Recovery Status': r['recovery_status'],
                'Data Points': r['data_points'],
                'Available Days': r['available_days']
            } for r in results])
            
            # Sort by consistency score (need to extract numeric value for sorting)
            results_df['_sort_score'] = [r['consistency_score'] for r in results]
            results_df = results_df.sort_values('_sort_score', ascending=False).drop('_sort_score', axis=1)
            
            # Display summary statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Funds Analyzed", len(results))
            with col2:
                avg_consistency = sum(r['consistency_score'] for r in results) / len(results)
                st.metric("Average Consistency Score", f"{avg_consistency:.1f}%")
            with col3:
                successful_rate = (len(results) / total_funds) * 100
                st.metric("Analysis Success Rate", f"{successful_rate:.1f}%")
            with col4:
                recovered_funds = sum(1 for r in results if 'Recovered' in r['recovery_status'])
                recovery_rate = (recovered_funds / len(results)) * 100
                st.metric("Recovery Rate", f"{recovery_rate:.1f}%")
            
            # Display results
            st.subheader(f"Analysis Results for {selected_period} Period")
            
            # Show simplified view by default
            display_columns = ['Fund Name', 'Consistency Score (%)', 'Total Return (%)', 
                             'Max Drawdown (%)', 'Recovery Status', 'Data Points']
            st.dataframe(results_df[display_columns], use_container_width=True)
            
            # Show detailed view as expandable
            with st.expander("View Detailed Results"):
                st.dataframe(results_df, use_container_width=True)
            
            # Add download button
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name=f"fund_analysis_{selected_category}_{selected_period}.csv",
                mime="text/csv"
            )
            
            # Visualizations
            if len(results) > 0:
                # Define consistent color mapping for all charts
                color_map = {
                    'Recovered': '#2ca02c',  # Green
                    'Partial': '#ff7f0e',    # Orange
                    'Not Recovered': '#d62728',  # Red
                    'Still Declining': '#8c564b',  # Brown
                    'Error': '#7f7f7f'      # Gray
                }
                
                # Create two columns for charts
                chart_col1, chart_col2 = st.columns(2)
                
                with chart_col1:
                    st.subheader("Consistency Score Distribution")
                    consistency_scores = [r['consistency_score'] for r in results]
                    
                    # Plotly histogram
                    fig1 = px.histogram(
                        x=consistency_scores,
                        nbins=min(20, len(results)//2 + 1),
                        labels={'x': 'Consistency Score (%)', 'y': 'Number of Funds'},
                        title=f'Consistency Scores - {selected_category}',
                        opacity=0.7,
                        color_discrete_sequence=['#1f77b4'],
                        histnorm='percent'
                    )
                    fig1.add_vline(x=50, line_dash="dash", line_color="red", 
                                 annotation_text="50% Benchmark", annotation_position="top")
                    fig1.update_layout(
                        yaxis_title="Percentage of Funds (%)",
                        bargap=0.1
                    )
                    st.plotly_chart(fig1, use_container_width=True)
                
                with chart_col2:
                    st.subheader("Fund Recovery Status (Lollipop Chart)")
                    
                    # Prepare recovery data
                    recovery_data = []
                    for r in results:
                        status = r['recovery_status']
                        recovery_pct = 0
                        
                        if 'Recovered' in status:
                            recovery_pct = 100
                        elif 'Partial' in status:
                            # Extract the percentage from the status string
                            try:
                                recovery_pct = float(status.split('(')[1].split('%')[0])
                            except:
                                recovery_pct = 50  # default if parsing fails
                        elif 'Still Declining' in status:
                            recovery_pct = -10  # special value for declining funds
                        
                        recovery_data.append({
                            'Fund Name': r['fund_name'],
                            'Recovery %': recovery_pct,
                            'Status': status,
                            'Max Drawdown': abs(r['max_drawdown']) if r['max_drawdown'] else 0,
                            'Consistency Score': r['consistency_score'],
                            'Total Return': r['total_return']
                        })
                    
                    recovery_df = pd.DataFrame(recovery_data)
                    
                    # Add radio button for selecting top/bottom performers
                    recovery_view = st.radio(
                        "Show:",
                        ["Top 10 Recovered", "Worst 10 Not Recovered"],
                        horizontal=True
                    )
                    
                    if recovery_view == "Top 10 Recovered":
                        # Sort by recovery percentage descending and get top 10
                        display_df = recovery_df.sort_values('Recovery %', ascending=False).head(10)
                        title_suffix = "Top 10 Recovered Funds"
                    else:
                        # Sort by recovery percentage ascending and get worst 10
                        display_df = recovery_df.sort_values('Recovery %', ascending=True).head(10)
                        title_suffix = "Worst 10 Not Recovered Funds"
                    
                    # Create color mapping
                    display_df['Color'] = display_df['Recovery %'].apply(
                        lambda x: '#2ca02c' if x >= 100 else  # Green for fully recovered
                                 '#ff7f0e' if x >= 50 else     # Orange for partially recovered
                                 '#d62728'                     # Red for not recovered
                    )
                    
                    # Create lollipop chart
                    fig2 = go.Figure()
                    
                    # Add segments (lines)
                    fig2.add_trace(go.Scatter(
                        x=display_df['Recovery %'],
                        y=display_df['Fund Name'],
                        mode='lines',
                        line=dict(color='#7f7f7f', width=1),
                        showlegend=False,
                        hoverinfo='none'
                    ))
                    
                    # Add dots (markers)
                    fig2.add_trace(go.Scatter(
                        x=display_df['Recovery %'],
                        y=display_df['Fund Name'],
                        mode='markers',
                        marker=dict(
                            color=display_df['Color'],
                            size=10,
                            line=dict(width=1, color='DarkSlateGrey')
                        ),
                        name='Recovery Status',
                        customdata=display_df[['Status', 'Max Drawdown', 'Consistency Score', 'Total Return']],
                        hovertemplate=(
                            "<b>%{y}</b><br>"
                            "Recovery: %{x:.1f}%<br>"
                            "Status: %{customdata[0]}<br>"
                            "Max Drawdown: %{customdata[1]:.1f}%<br>"
                            "Consistency: %{customdata[2]:.1f}%<br>"
                            "Return: %{customdata[3]:.1f}%<extra></extra>"
                        )
                    ))
                    
                    # Add vertical line at 100% recovery
                    fig2.add_vline(x=100, line_dash="dot", line_color="green", opacity=0.3)
                    
                    # Customize layout
                    fig2.update_layout(
                        title=f'{title_suffix} - {selected_category}',
                        xaxis_title="Recovery Percentage from Max Drawdown",
                        yaxis_title="Fund Name",
                        height=500,
                        hovermode='closest',
                        showlegend=False,
                        xaxis=dict(
                            range=[-20, max(120, display_df['Recovery %'].max() + 10)],
                            tickvals=[0, 50, 100, 150, 200],
                            ticktext=['0% (Not Recovered)', '50%', '100% (Fully Recovered)', '150%', '200%']
                        ),
                        margin=dict(l=150)  # Increase left margin for fund names
                    )
                    
                    st.plotly_chart(fig2, use_container_width=True)
                
                # Bubble chart: Consistency vs Returns
                st.subheader("Consistency Score vs Total Returns")
                
                # Prepare data for bubble chart
                bubble_data = pd.DataFrame({
                    'Consistency Score': [r['consistency_score'] for r in results],
                    'Total Return': [r['total_return'] for r in results],
                    'Max Drawdown': [abs(r['max_drawdown']) if r['max_drawdown'] else 0 for r in results],
                    'Recovery Status': [r['recovery_status'] for r in results],
                    'Fund Name': [r['fund_name'] for r in results],
                    'Volatility': [r['annualized_volatility'] for r in results]
                })
                
                # Simplify recovery status for coloring
                bubble_data['Status Group'] = bubble_data['Recovery Status'].apply(
                    lambda x: 'Recovered' if 'Recovered' in x else
                             'Partial' if 'Partial' in x else
                             'Not Recovered' if 'Not Recovered' in x else
                             'Still Declining' if 'Still Declining' in x else
                             'Error'
                )
                
                # Create bubble chart with size based on max drawdown and color based on recovery status
                fig3 = px.scatter(
                    bubble_data,
                    x='Consistency Score',
                    y='Total Return',
                    size='Max Drawdown',
                    color='Status Group',
                    color_discrete_map=color_map,
                    hover_name='Fund Name',
                    size_max=30,
                    title=f'Consistency vs Returns - {selected_category} ({selected_period})',
                    labels={
                        'Consistency Score': 'Consistency Score (%)',
                        'Total Return': 'Total Return (%)',
                        'Max Drawdown': 'Max Drawdown (%)',
                        'Status Group': 'Recovery Status'
                    },
                    custom_data=['Recovery Status', 'Volatility']
                )
                
                # Add reference lines
                fig3.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                fig3.add_vline(x=50, line_dash="dash", line_color="gray", opacity=0.5)
                
                # Customize hover template
                fig3.update_traces(
                    hovertemplate=(
                        "<b>%{hovertext}</b><br><br>"
                        "Consistency: %{x:.1f}%<br>"
                        "Return: %{y:.1f}%<br>"
                        "Max Drawdown: %{marker.size:.1f}%<br>"
                        "Volatility: %{customdata[1]:.1f}%<br>"
                        "Status: %{customdata[0]}<extra></extra>"
                    )
                )
                
                # Customize layout
                fig3.update_layout(
                    height=600,
                    hovermode='closest',
                    legend_title_text='Recovery Status',
                    xaxis_range=[0, 100]
                )
                
                st.plotly_chart(fig3, use_container_width=True)
                
                # Top performers
                st.subheader("Top 10 Performers by Consistency Score")
                top_performers = results_df.head(10)[display_columns]
                st.dataframe(top_performers, use_container_width=True)
            
            # Show failed funds if debug is enabled
            if show_debug and failed_funds:
                st.subheader("Failed Fund Analysis")
                st.write(f"Total failed: {len(failed_funds)}")
                failed_df = pd.DataFrame(failed_funds)
                st.dataframe(failed_df)
            
            # Metric explanations
            with st.expander("Metrics Explanation"):
                st.markdown("""
                **Performance Metrics:**
                - **Consistency Score**: Percentage of rolling periods where fund outperformed the benchmark
                - **Total Return**: Overall return for the selected period
                - **Volatility**: Annualized standard deviation of daily returns (risk measure)
                
                **Risk Metrics:**
                - **Max Drawdown**: Worst peak-to-trough decline during the period (lower is better)
                - **Recovery Days**: Time taken to recover from maximum drawdown
                - **Recovery Status**: Current recovery state from maximum drawdown
                
                **Recovery Status Types:**
                - **Recovered (X days)**: Fund has fully recovered from its maximum drawdown in X days
                - **Partial (X%)**: Fund has recovered X% from its maximum drawdown
                - **Not Recovered**: Fund hasn't recovered from its maximum drawdown
                - **Still Declining**: Fund continues to decline from its drawdown
                
                **Interpretation:**
                - Consistency Score > 50%: Fund generally outperforms benchmark
                - Lower Max Drawdown + Higher Consistency = Better risk-adjusted performance
                - "N/A" Recovery Days = Fund hasn't recovered from its worst loss yet
                """)

if __name__ == "__main__":
    main()