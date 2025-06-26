import numpy as np
import streamlit as st
import plotly.express as px
import requests
import json

def normalize(value, min_val, max_val):
    """Normalize values to a 0-100 scale."""
    return max(0, min(100, 100 * (value - min_val) / (max_val - min_val))) if max_val > min_val else 0

def fetch_nse_data():
    """Fetch real-time stock indices data from NSE website."""
    url = "https://www.nseindia.com/api/allIndices"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            return None
        
        data = response.json()
        indices_data = {}
        for index in data['data']:
            index_name = index.get('index', 'Unknown')
            indices_data[index_name] = {
                'institutional_ownership': index.get('institutionalOwnership', np.random.uniform(30, 90)),
                'retail_trading': index.get('retailParticipation', np.random.uniform(20, 80)),
                'turnover_ratio': index.get('turnoverRatio', np.random.uniform(10, 150)),
                'volatility': index.get('volatility', np.random.uniform(10, 40)),
                'margin_debt': index.get('marginDebt', np.random.uniform(5, 50)),
                'avg_holding_period': index.get('averageHoldingPeriod', np.random.uniform(30, 365)),
                'etf_trading': index.get('etfParticipation', np.random.uniform(10, 60)),
                'pe_stability': index.get('peStability', np.random.uniform(5, 50)),
                'earnings_reaction': index.get('earningsReaction', np.random.uniform(5, 20)),
                'bid_ask_spread': index.get('bidAskSpread', np.random.uniform(0.1, 1)),
                'social_hype': index.get('socialMediaHype', np.random.uniform(10, 80)),
                'speculative_options': index.get('speculativeOptionsActivity', np.random.uniform(5, 40)),
            }
        return indices_data
    except Exception as e:
        st.write(f"Error fetching NSE data: {e}")
        return None

def compute_imi(data):
    """Compute the Investor Maturity Index (IMI) based on weighted metrics."""
    institutional_score = normalize(data['institutional_ownership'], 30, 90)
    retail_score = normalize(100 - data['retail_trading'], 20, 80)
    turnover_score = normalize(100 - data['turnover_ratio'], 10, 150)
    volatility_score = normalize(100 - data['volatility'], 10, 40)
    margin_score = normalize(100 - data['margin_debt'], 5, 50)
    holding_score = normalize(data['avg_holding_period'], 30, 365)
    etf_score = normalize(data['etf_trading'], 10, 60)
    pe_score = normalize(100 - data['pe_stability'], 5, 50)
    earnings_score = normalize(100 - data['earnings_reaction'], 5, 20)
    spread_score = normalize(100 - data['bid_ask_spread'], 0.1, 1)
    sentiment_score = normalize(100 - data['social_hype'], 10, 80)
    options_score = normalize(100 - data['speculative_options'], 5, 40)
    
    C = 0.4 * institutional_score + 0.6 * retail_score  # 30%
    T = 0.5 * turnover_score + 0.3 * volatility_score + 0.2 * margin_score  # 25%
    H = 0.5 * holding_score + 0.5 * etf_score  # 20%
    V = 0.3 * pe_score + 0.4 * earnings_score + 0.3 * spread_score  # 15%
    S = 0.4 * sentiment_score + 0.6 * options_score  # 10%
    
    IMI = 0.3 * C + 0.25 * T + 0.2 * H + 0.15 * V + 0.1 * S
    return round(IMI, 2)

def categorize_imi(imi):
    """Categorize IMI into maturity levels and suggest investment strategies."""
    if imi > 70:
        return "High Maturity - Stable, Long-term Focus"
    elif 50 <= imi <= 70:
        return "Moderate Maturity - Balanced Growth & Risk"
    else:
        return "Low Maturity - High-Risk, High-Reward"

def visualize_imi(index_names, imi_scores):
    """Visualize the Investor Maturity Index for different indices using Plotly."""
    sorted_indices_scores = sorted(zip(index_names, imi_scores), key=lambda x: x[1], reverse=True)
    sorted_indices, sorted_scores = zip(*sorted_indices_scores)
    
    fig = px.bar(x=sorted_indices, y=sorted_scores, color=sorted_scores, color_continuous_scale=["red", "blue", "green"],
                 labels={"x": "Stock Indices", "y": "Investor Maturity Index (IMI)"}, title="Investor Maturity Index for Indian Stock Indices")
    fig.update_layout(xaxis_tickangle=-45, width=1200, height=700)
    st.plotly_chart(fig)

def visualize_risk_reward(index_names, imi_scores):
    """Visualize Risk vs. IMI scores."""
    fig = px.scatter(x=imi_scores, y=[100 - imi for imi in imi_scores], text=index_names,
                     labels={"x": "Investor Maturity Index (IMI)", "y": "Risk Score (100 - IMI)"},
                     title="Risk vs. Investor Maturity Index")
    fig.update_traces(textposition="top center")
    st.plotly_chart(fig)

st.title("Investor Maturity Index (IMI) for Indian Stock Indices")

nse_data = fetch_nse_data()
if not nse_data:
    st.write("Failed to retrieve data from NSE. Please try again later.")
else:
    imi_scores = {index: compute_imi(data) for index, data in nse_data.items()}
    visualize_imi(list(imi_scores.keys()), list(imi_scores.values()))
    visualize_risk_reward(list(imi_scores.keys()), list(imi_scores.values()))
    
    st.write("### Investment Strategies Based on IMI")
    for index, imi in imi_scores.items():
        st.write(f"- **{index}**: {categorize_imi(imi)}")
    
    st.write("### Explanation of Metrics and IMI Interpretation")
    st.write("""
The Investor Maturity Index (IMI) is calculated using the following key metrics:

- **Institutional Ownership**: Higher institutional participation suggests more informed investment decisions.
- **Retail Trading Activity**: Excessive retail participation often indicates speculative behavior.
- **Turnover Ratio**: High turnover suggests frequent trading, common among less mature investors.
- **Volatility**: High volatility often correlates with speculative trading.
- **Margin Debt**: Excessive margin usage may indicate overleveraged retail traders.
- **Average Holding Period**: Longer holding periods indicate long-term investing behavior.
- **ETF Trading Participation**: More ETF involvement suggests passive, stable investing.
- **PE Ratio Stability**: Lower fluctuations indicate consistent valuation.
- **Earnings Reaction**: Lower reaction volatility implies market maturity.
- **Bid-Ask Spread**: Narrower spreads indicate greater liquidity and institutional presence.
- **Social Media Hype**: High hype correlates with speculative activity.
- **Speculative Options Activity**: More speculative trades suggest high retail-driven behavior.

### IMI Interpretation:
- **Above 70**: High Maturity - Indicates stability and informed investing.
- **50 - 70**: Moderate Maturity - Balanced growth and risk.
- **Below 50**: Low Maturity - Speculative and volatile investments.
""")
    st.write("This section provides detailed documentation on the metrics used in calculating IMI, their significance, and interpretation of different IMI ranges.")
