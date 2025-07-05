import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time
from nsepython import nse_get_quote, nse_get_hist

def calculate_ulcer_index(prices, window=14):
    """Calculates the Ulcer Index over a rolling window."""
    max_prices = prices.rolling(window=window, min_periods=1).max()
    drawdowns = ((prices - max_prices) / max_prices) * 100
    ulcer_index = np.sqrt((drawdowns**2).rolling(window=window, min_periods=1).mean())
    return ulcer_index

def fetch_stock_data(ticker_symbol, max_retries=3):
    """Fetch stock data from NSE with retry mechanism."""
    for attempt in range(max_retries):
        try:
            hist_data = nse_get_hist(ticker_symbol, "Equity")
            if hist_data is not None and not hist_data.empty:
                return hist_data
        except Exception as e:
            wait_time = (2 ** attempt)  # Exponential backoff
            st.warning(f"Error fetching data: {str(e)}. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
    return None

# Streamlit UI
st.title("Ulcer Index Stock Analyzer")

# User input for stock symbol
stock_symbol = st.text_input("Enter NSE Stock Symbol (e.g., RELIANCE):", "RELIANCE")
window = st.slider("Select rolling window for Ulcer Index calculation:", 5, 100, 14)

if st.button("Analyze"):
    try:
        data = fetch_stock_data(stock_symbol)
        
        if data is None or data.empty:
            st.error("No data found! Please check the stock symbol or try another one.")
            st.write("🔍 Tips:")
            st.write("- Ensure the stock ticker is correct (e.g., `RELIANCE`, `TCS`, `INFY`).")
            st.write("- The stock may be delisted. Try another stock.")
        else:
            data['Ulcer Index'] = calculate_ulcer_index(data['Close Price'], window=window)
            
            # Plot stock price
            fig_price = px.line(data, x=data.index, y='Close Price', title=f"{stock_symbol} Stock Price", labels={'Close Price': 'Stock Price'})
            st.plotly_chart(fig_price)
            
            # Plot Ulcer Index
            fig_ulcer = px.line(data, x=data.index, y='Ulcer Index', title=f"Ulcer Index for {stock_symbol}", labels={'Ulcer Index': 'Ulcer Index Value'})
            st.plotly_chart(fig_ulcer)
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("- Ensure you have an active internet connection.")
        st.write("- Try a different stock ticker.")
