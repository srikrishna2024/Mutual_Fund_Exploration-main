import streamlit as st
import pandas as pd
import numpy as np
from arch import arch_model
import plotly.graph_objects as go
from datetime import datetime, timedelta
import psycopg

# Database connection
def connect_to_db():
    DB_PARAMS = {
        'dbname': 'postgres',
        'user': 'postgres',
        'password': 'admin123',
        'host': 'localhost',
        'port': '5432'
    }
    return psycopg.connect(**DB_PARAMS)

def get_fund_list():
    with connect_to_db() as conn:
        query = """
            SELECT DISTINCT code, scheme_name
            FROM mutual_fund_nav
            ORDER BY scheme_name
        """
        return pd.read_sql(query, conn)

def get_nav_history(code):
    with connect_to_db() as conn:
        query = """
            SELECT nav as date, value
            FROM mutual_fund_nav
            WHERE code = %s
            ORDER BY nav
        """
        return pd.read_sql(query, conn, params=(code,))

def calculate_trend_metrics(nav_df):
    """Calculate trend metrics for different time periods"""
    current_date = nav_df['date'].max()
    
    # Define time periods
    periods = {
        '1M': current_date - timedelta(days=30),
        '3M': current_date - timedelta(days=90),
        '6M': current_date - timedelta(days=180),
        'YTD': datetime(current_date.year, 1, 1),
        '1Y': current_date - timedelta(days=365),
        '3Y': current_date - timedelta(days=3*365)
    }
    
    metrics = {}
    for period_name, start_date in periods.items():
        period_data = nav_df[nav_df['date'] >= start_date]
        if len(period_data) > 1:
            start_value = period_data['value'].iloc[0]
            end_value = period_data['value'].iloc[-1]
            return_pct = (end_value / start_value - 1) * 100
            metrics[period_name] = {
                'return': return_pct,
                'trend': "↑" if return_pct > 0 else "↓",
                'volatility': period_data['return'].std() * 100,
                'delta_color': "normal" if return_pct > 0 else "inverse"
            }
        else:
            metrics[period_name] = {
                'return': 0,
                'trend': "↔",
                'volatility': 0,
                'delta_color': "off"
            }
    
    return metrics

def calculate_kpis(garch_results, nav_df, trend_metrics):
    """Calculate key GARCH performance indicators"""
    alpha1 = garch_results.params.get('alpha[1]', 0)
    beta1 = garch_results.params.get('beta[1]', 0)
    omega = garch_results.params.get('omega', 0)
    
    # Clustering KPIs
    cluster_stats = nav_df.groupby('cluster')['garch_vol'].agg(['mean', 'count'])
    current_cluster = nav_df['cluster'].iloc[-1]
    cluster_duration = (nav_df['date'].iloc[-1] - nav_df[nav_df['cluster'] == current_cluster]['date'].min()).days
    
    # Calculate mean return adjusted for volatility (simple Sharpe-like ratio)
    mean_return = nav_df['return'].mean() * 100  # Convert to percentage
    risk_adjusted_return = mean_return / nav_df['garch_vol'].mean() if nav_df['garch_vol'].mean() != 0 else 0
    
    kpis = {
        # Volatility KPIs
        'Current Volatility': nav_df['garch_vol'].iloc[-1],
        'Volatility Trend': "Increasing" if nav_df['vol_diff'].iloc[-1] > 0 else "Decreasing",
        'Persistence (α+β)': alpha1 + beta1,
        'Long-Term Avg Volatility': nav_df['garch_vol'].mean(),
        'Volatility Ratio (Current/LT)': nav_df['garch_vol'].iloc[-1]/nav_df['garch_vol'].mean(),
        'Model Fit (ω)': omega,
        'Mean Return (%)': mean_return,
        'Risk Adjusted Return': risk_adjusted_return,
        
        # Clustering KPIs
        'Current Cluster': current_cluster,
        'Cluster Duration (days)': cluster_duration,
        'Cluster Volatility': cluster_stats.loc[current_cluster, 'mean'],
        'Cluster Trend': nav_df['trend'].iloc[-1],
        
        # Trend Metrics
        'Trend Metrics': trend_metrics
    }
    return kpis

def generate_recommendation(kpis):
    """Generate investment recommendation with time period context"""
    current_vol = kpis['Current Volatility']
    lt_vol = kpis['Long-Term Avg Volatility']
    persistence = kpis['Persistence (α+β)']
    trend_metrics = kpis['Trend Metrics']
    cluster_duration = kpis['Cluster Duration (days)']
    risk_adjusted_return = kpis['Risk Adjusted Return']
    
    # Get most relevant periods
    ytd_return = trend_metrics['YTD']['return']
    m3_return = trend_metrics['3M']['return']
    cluster_trend = kpis['Cluster Trend']
    
    recommendations = []
    time_context = []
    
    # Volatility analysis
    if current_vol > lt_vol * 2:
        recommendations.append("Very high current volatility (2x above average)")
        time_context.append("current")
    elif current_vol > lt_vol * 1.5:
        recommendations.append("High current volatility (1.5x above average)")
        time_context.append("current")
    elif current_vol < lt_vol * 0.8:
        recommendations.append("Low current volatility (20% below average)")
        time_context.append("current")
    
    # Cluster analysis
    if cluster_trend == "Increasing" and cluster_duration > 30:
        recommendations.append("prolonged increasing volatility cluster")
        time_context.append(f"{cluster_duration}-day cluster")
    
    # Recent performance
    if m3_return > 10:
        recommendations.append("strong 3-month returns")
        time_context.append("3-month")
    elif m3_return < -5:
        recommendations.append("negative 3-month returns")
        time_context.append("3-month")
    
    if ytd_return > 15:
        recommendations.append("strong YTD performance")
        time_context.append("YTD")
    
    # Risk-adjusted returns
    if risk_adjusted_return > 1:
        recommendations.append("good risk-adjusted returns")
    elif risk_adjusted_return < 0:
        recommendations.append("negative risk-adjusted returns")
    
    # Generate time context description
    if not time_context:
        time_context_str = "based on long-term trends"
    else:
        time_context_str = f"based on {' + '.join(set(time_context))} data"
    
    # Generate final recommendation
    if not recommendations:
        return f"Neutral - Market conditions are normal {time_context_str}"
    
    if ("High current volatility" in recommendations or "Very high current volatility" in recommendations) and any(x in ['negative 3-month returns', 'prolonged increasing volatility cluster', 'negative risk-adjusted returns'] for x in recommendations):
        return f"⚠️ Caution - High volatility with weak returns {time_context_str}. Consider reducing exposure."
    elif "Low current volatility" in recommendations and any(x in ['strong 3-month returns', 'strong YTD performance', 'good risk-adjusted returns'] for x in recommendations):
        return f"✅ Opportunity - Favorable conditions {time_context_str}. Consider adding to position."
    elif "negative 3-month returns" in recommendations and ("High current volatility" in recommendations or "Very high current volatility" in recommendations):
        return f"🔴 Risk Off - Poor performance with high volatility {time_context_str}. Consider defensive positions."
    else:
        return f"➖ Mixed Signals - {', '.join(recommendations)} {time_context_str}"

def display_garch_interpretation(kpis):
    """Display GARCH KPI interpretation guide"""
    with st.expander("📊 GARCH KPIs Interpretation Guide"):
        st.markdown("""
        ### 1. GARCH Volatility Level (σ)
        **What it measures:** How much the fund's returns are swinging daily  
        **Current Value:** {:.2f}%  
        **Long-Term Average:** {:.2f}%  
        **Interpretation:**  
        - {}  
        - **Thresholds:**  
          • < {:.2f}% (Below average) → Potential buying opportunity  
          • > {:.2f}% (2x average) → Caution advised  

        ### 2. Volatility Trend
        **What it measures:** Whether volatility is rising or falling  
        **Current Trend:** {}  
        **Interpretation:**  
        - {}  

        ### 3. Persistence (α + β)
        **What it measures:** How long volatility shocks last  
        **Current Value:** {:.2f}  
        **Interpretation:**  
        - {}  
        - **Thresholds:**  
          • < 0.9 → Volatility shocks fade quickly  
          • ≥ 0.9 → Volatility clusters persist  

        ### 4. Risk-Adjusted Return
        **What it measures:** Returns relative to volatility  
        **Current Value:** {:.2f}  
        **Interpretation:**  
        - {}  
        - **Thresholds:**  
          • > 1 → Good risk-reward ratio  
          • < 0 → Negative returns after adjusting for risk  

        ### 5. Volatility Regime Shifts
        **What it measures:** Changes in volatility patterns  
        **Current Cluster Duration:** {} days  
        **Interpretation:**  
        - {}  
        - **Thresholds:**  
          • < 30 days → New volatility regime  
          • > 30 days → Established trend  
        """.format(
            kpis['Current Volatility'],
            kpis['Long-Term Avg Volatility'],
            "Low volatility (smooth growth)" if kpis['Current Volatility'] < kpis['Long-Term Avg Volatility'] else "High volatility (wild swings)",
            kpis['Long-Term Avg Volatility'] * 0.8,
            kpis['Long-Term Avg Volatility'] * 2,
            kpis['Volatility Trend'],
            "Market confidence is stable" if kpis['Volatility Trend'] == "Decreasing" else "Uncertainty may be increasing",
            kpis['Persistence (α+β)'],
            "Volatility shocks will fade quickly" if kpis['Persistence (α+β)'] < 0.9 else "Volatility clusters may persist",
            kpis['Risk Adjusted Return'],
            "Good returns for risk taken" if kpis['Risk Adjusted Return'] > 1 else "Poor returns for risk taken",
            kpis['Cluster Duration (days)'],
            "New volatility pattern emerging" if kpis['Cluster Duration (days)'] < 30 else "Established volatility trend"
        ))

def plot_volatility_clustering(nav_df, fund_name, selected_period):
    st.subheader("Volatility Clustering Analysis")

    nav_df = nav_df.copy()
    nav_df['date'] = pd.to_datetime(nav_df['date'])
    nav_df = nav_df.sort_values('date')

    period_days = {
        'YTD': 'Year-to-Date',
        '1M': '1 Month',
        '3M': '3 Months',
        '6M': '6 Months',
        '1Y': '1 Year',
        '3Y': '3 Years',
        '5Y': '5 Years',
        '10Y': '10 Years',
        'All': 'All History'
    }

    if selected_period == 'YTD':
        cutoff = datetime(nav_df['date'].max().year, 1, 1)
        nav_df = nav_df[nav_df['date'] >= cutoff]
    elif selected_period != 'All':
        days = int(''.join(filter(str.isdigit, selected_period))) * 30 if 'M' in selected_period else \
               int(''.join(filter(str.isdigit, selected_period))) * 365
        cutoff = nav_df['date'].max() - timedelta(days=days)
        nav_df = nav_df[nav_df['date'] >= cutoff]

    nav_df['return'] = nav_df['value'].pct_change().dropna()
    nav_df.dropna(inplace=True)

    # GARCH model
    try:
        am = arch_model(nav_df['return'] * 100, vol='Garch', p=1, q=1, rescale=False)
        res = am.fit(disp="off")
        nav_df['garch_vol'] = np.sqrt(res.conditional_volatility)
    except Exception as e:
        st.error(f"GARCH model failed: {str(e)}")
        return

    nav_df['rolling_vol'] = nav_df['return'].rolling(window=30).std()
    nav_df['vol_diff'] = nav_df['garch_vol'].diff()
    nav_df['trend'] = np.where(nav_df['vol_diff'] > 0, 'Increasing', 'Decreasing')
    nav_df['cluster'] = (nav_df['trend'] != nav_df['trend'].shift()).cumsum()
    
    # Calculate metrics
    trend_metrics = calculate_trend_metrics(nav_df)
    kpis = calculate_kpis(res, nav_df, trend_metrics)
    recommendation = generate_recommendation(kpis)
    
    # Display KPIs
    st.subheader("📊 Key Performance Indicators")
    
    # Volatility metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current Volatility", f"{kpis['Current Volatility']:.2f}%",
                delta=f"{kpis['Volatility Trend']}",
                delta_color="inverse" if kpis['Volatility Trend'] == "Increasing" else "normal",
                help="Daily volatility level from GARCH model")
    with col2:
        st.metric("Persistence (α+β)", f"{kpis['Persistence (α+β)']:.2f}",
                help="How long volatility shocks persist (0-1 scale)")
    with col3:
        st.metric("Risk-Adjusted Return", f"{kpis['Risk Adjusted Return']:.2f}",
                help="Mean return divided by volatility")
    with col4:
        st.metric("Cluster Duration", f"{kpis['Cluster Duration (days)']} days",
                help="Days in current volatility regime")

    # Performance metrics - fixed with proper arrows and colors
    st.subheader("📈 Performance Across Time Horizons")
    periods = ['1M', '3M', '6M', 'YTD', '1Y', '3Y']
    cols = st.columns(len(periods))
    for i, period in enumerate(periods):
        with cols[i]:
            st.metric(
                label=f"{period} Return",
                value=f"{trend_metrics[period]['return']:.1f}%",
                delta=trend_metrics[period]['trend'],
                delta_color=trend_metrics[period]['delta_color']
            )

    # GARCH Interpretation Guide
    display_garch_interpretation(kpis)

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=nav_df['date'],
        y=nav_df['rolling_vol'],
        mode='lines',
        name='30-Day Rolling Volatility',
        line=dict(color='gray', width=1)
    ))
    fig.add_trace(go.Scatter(
        x=nav_df['date'],
        y=nav_df['garch_vol'],
        mode='lines',
        name='GARCH Volatility',
        line=dict(color='black', width=2)
    ))

    # Add clustering periods
    colors = {'Increasing': 'rgba(0,255,0,0.2)', 'Decreasing': 'rgba(255,0,0,0.2)'}
    added_trends = set()
    cluster_groups = nav_df.groupby(['cluster', 'trend'])
    
    for (cluster_id, trend), group in cluster_groups:
        if len(group) > 5:
            show_legend = trend not in added_trends
            added_trends.add(trend)
            fig.add_shape(
                type="rect",
                x0=group['date'].min(),
                x1=group['date'].max(),
                y0=0,
                y1=nav_df['garch_vol'].max() * 1.1,
                fillcolor=colors.get(trend, 'rgba(0,0,0,0.1)'),
                line=dict(width=0),
                layer="below"
            )
            fig.add_trace(go.Scatter(
                x=[group['date'].min(), group['date'].max()],
                y=[0, 0],
                mode='lines',
                line=dict(color='rgba(0,0,0,0)'),
                showlegend=show_legend,
                name=f"{trend} Volatility Cluster",
                hoverinfo='text',
                text=[f"{trend} Volatility Cluster\nFrom {group['date'].min().date()} to {group['date'].max().date()}"] * 2
            ))

    fig.update_layout(
        title=f"Volatility Clustering in {fund_name} ({selected_period})",
        xaxis_title="Date",
        yaxis_title="Volatility",
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Recommendation and Interpretation
    st.subheader("💡 Investment Recommendation")
    st.info(recommendation)
    
    with st.expander("📚 Cluster Duration Interpretation"):
        st.markdown("""
        ### Cluster Duration Meaning:
        - **< 30 days**: New volatility pattern still forming
        - **30-90 days**: Established volatility regime
        - **> 90 days**: Strong, persistent market condition
        
        ### How to Use This:
        1. **Short clusters (<30 days)**: 
           - May represent temporary market noise
           - Wait for confirmation before acting
        2. **Medium clusters (30-90 days)**:
           - Likely represents a real market phase
           - Consider adjusting positions accordingly
        3. **Long clusters (>90 days)**:
           - Strong market regime in place
           - High confidence in continued trend
        """)

# Streamlit App
def main():
    st.set_page_config(page_title="Mutual Fund Volatility Dashboard", layout="wide")
    st.title("📈 Mutual Fund Volatility Analysis")
    
    st.write("""
    Advanced volatility analysis with comprehensive GARCH interpretation and cluster duration analysis.
    Recommendations now incorporate multiple time horizons and volatility regimes.
    """)
    
    fund_list = get_fund_list()
    if fund_list.empty:
        st.warning("No mutual fund data found in database.")
        return

    col1, col2 = st.columns(2)
    with col1:
        selected_fund = st.selectbox("Select Mutual Fund", fund_list['scheme_name'])
    with col2:
        selected_period = st.selectbox("Analysis Period", 
                                     ['YTD', '1M', '3M', '6M', '1Y', '3Y', '5Y', '10Y', 'All'],
                                     index=0)

    selected_code = fund_list[fund_list['scheme_name'] == selected_fund]['code'].iloc[0]
    nav_history = get_nav_history(selected_code)
    
    if nav_history.empty:
        st.warning("No NAV data found for selected fund.")
        return

    plot_volatility_clustering(nav_history, selected_fund, selected_period)

if __name__ == "__main__":
    main()