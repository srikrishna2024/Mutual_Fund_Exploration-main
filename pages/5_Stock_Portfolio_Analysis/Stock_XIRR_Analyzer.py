import streamlit as st
import pandas as pd
import numpy as np
import psycopg
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from scipy.optimize import fsolve
import math

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Portfolio XIRR Analysis", layout="wide")
st.title("📈 Portfolio XIRR & Performance Analysis")

st.markdown("""
This application analyzes your portfolio performance with XIRR calculations, current market values,
and comprehensive metrics including CAGR, dividend gains, and absolute returns.
""")

def connect_to_db():
    """Connect to PostgreSQL database"""
    DB_PARAMS = {
        'dbname': 'postgres',
        'user': 'postgres',
        'password': 'admin123',
        'host': 'localhost',
        'port': '5432'
    }
    try:
        conn = psycopg.connect(**DB_PARAMS)
        return conn
    except Exception as e:
        st.error(f"Database connection failed: {str(e)}")
        return None

def load_portfolio_data():
    """Load portfolio data from database"""
    try:
        with st.spinner("Loading portfolio data..."):
            conn = connect_to_db()
            if conn is None:
                return None
                
            # Query the database
            query = """
            SELECT symbol, trade_date, trade_type, quantity, price 
            FROM portfolio_stocks
            ORDER BY symbol, trade_date
            """
            
            df = pd.read_sql(query, conn)
            conn.close()
            
            if df.empty:
                st.warning("No data found in portfolio_stocks table")
                return None
                
            # Data cleaning
            df.columns = df.columns.str.lower()
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df['quantity'] = pd.to_numeric(df['quantity'])
            df['price'] = pd.to_numeric(df['price'])
            df['amount'] = df['quantity'] * df['price']
            
            return df
            
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def get_last_transaction_prices(df):
    """Get last transaction price for each symbol"""
    last_prices = {}
    
    for symbol in df['symbol'].unique():
        symbol_data = df[df['symbol'] == symbol].sort_values('trade_date')
        if not symbol_data.empty:
            last_prices[symbol] = symbol_data.iloc[-1]['price']
    
    return last_prices

def get_manual_prices(symbols, holdings_info):
    """Get current prices manually or use last transaction price"""
    st.subheader("💰 Current Stock Prices")
    st.write("Enter current market prices for your stocks (or leave blank to use last transaction price):")
    
    current_prices = {}
    
    # Create columns for better layout
    cols = st.columns(3)
    
    for i, symbol in enumerate(symbols):
        with cols[i % 3]:
            # Get last transaction price as default
            last_price = holdings_info.get(symbol, {}).get('last_price', 0)
            
            price_input = st.number_input(
                f"{symbol}",
                min_value=0.0,
                value=float(last_price) if last_price > 0 else 0.0,
                step=0.01,
                format="%.2f",
                key=f"price_{symbol}"
            )
            current_prices[symbol] = price_input
    
    return current_prices

def calculate_xirr(cash_flows, dates, guess=0.1):
    """Calculate XIRR using improved Newton-Raphson method with multiple attempts"""
    try:
        # Input validation
        if len(cash_flows) != len(dates) or len(cash_flows) < 2:
            return None
        
        # Check if we have both positive and negative cash flows
        positive_flows = [cf for cf in cash_flows if cf > 0]
        negative_flows = [cf for cf in cash_flows if cf < 0]
        
        if not positive_flows or not negative_flows:
            return None
        
        # Convert dates to days from first date
        start_date = min(dates)
        days = [(date - start_date).days for date in dates]
        
        # Check for minimum time period (at least 1 day)
        if max(days) == 0:
            return None
        
        def npv(rate):
            try:
                return sum(cf / (1 + rate) ** (day / 365.25) for cf, day in zip(cash_flows, days))
            except:
                return float('inf')
        
        def npv_derivative(rate):
            try:
                return sum(-cf * (day / 365.25) / (1 + rate) ** (day / 365.25 + 1) for cf, day in zip(cash_flows, days))
            except:
                return 0
        
        # Try multiple initial guesses
        guesses = [0.1, -0.1, 0.5, -0.5, 0.01, -0.01]
        
        for initial_guess in guesses:
            try:
                rate = initial_guess
                
                # Newton-Raphson method with bounds checking
                for iteration in range(100):
                    npv_val = npv(rate)
                    
                    # Check for convergence
                    if abs(npv_val) < 1e-6:
                        # Validate the result
                        if -10 <= rate <= 10:  # Reasonable bounds for annual returns
                            return rate
                        break
                    
                    npv_deriv = npv_derivative(rate)
                    
                    # Check for derivative issues
                    if abs(npv_deriv) < 1e-10:
                        break
                    
                    # Calculate new rate
                    new_rate = rate - npv_val / npv_deriv
                    
                    # Apply bounds to prevent extreme values
                    new_rate = max(min(new_rate, 10), -0.99)  # Bounds: -99% to 1000%
                    
                    # Check for convergence in rate
                    if abs(new_rate - rate) < 1e-8:
                        if -10 <= new_rate <= 10:
                            return new_rate
                        break
                    
                    rate = new_rate
                    
            except:
                continue
        
        # If Newton-Raphson fails, try a simple bisection method
        return bisection_xirr(cash_flows, days)
        
    except:
        return None

def bisection_xirr(cash_flows, days):
    """Fallback XIRR calculation using bisection method"""
    try:
        def npv(rate):
            return sum(cf / (1 + rate) ** (day / 365.25) for cf, day in zip(cash_flows, days))
        
        # Set bounds
        low = -0.99  # -99%
        high = 10.0  # 1000%
        
        # Check if solution exists in bounds
        if npv(low) * npv(high) > 0:
            return None
        
        # Bisection method
        for _ in range(100):
            mid = (low + high) / 2
            npv_mid = npv(mid)
            
            if abs(npv_mid) < 1e-6:
                return mid
            
            if npv(low) * npv_mid < 0:
                high = mid
            else:
                low = mid
            
            if abs(high - low) < 1e-8:
                return mid
        
        return None
    except:
        return None

def calculate_cagr(initial_value, final_value, years):
    """Calculate Compound Annual Growth Rate"""
    if initial_value <= 0 or final_value <= 0 or years <= 0:
        return 0
    try:
        return ((final_value / initial_value) ** (1 / years)) - 1
    except:
        return 0

def calculate_holdings_with_metrics(df, current_prices):
    """Calculate holdings with comprehensive metrics"""
    holdings = {}
    
    for symbol in df['symbol'].unique():
        stock_data = df[df['symbol'] == symbol].sort_values('trade_date')
        
        # FIFO calculation
        fifo_queue = []
        cash_flows = []
        cash_flow_dates = []
        total_dividends = 0  # You can add dividend data to your database
        
        for _, row in stock_data.iterrows():
            if row['trade_type'].lower() in ['buy', 'b']:
                fifo_queue.append({
                    'date': row['trade_date'],
                    'quantity': row['quantity'],
                    'price': row['price']
                })
                # Negative cash flow for purchases
                cash_flows.append(-row['amount'])
                cash_flow_dates.append(row['trade_date'])
                
            elif row['trade_type'].lower() in ['sell', 's']:
                remaining_to_sell = row['quantity']
                sell_price = row['price']
                
                while remaining_to_sell > 0 and fifo_queue:
                    oldest_lot = fifo_queue[0]
                    
                    if oldest_lot['quantity'] <= remaining_to_sell:
                        sold_quantity = oldest_lot['quantity']
                        remaining_to_sell -= sold_quantity
                        fifo_queue.pop(0)
                    else:
                        sold_quantity = remaining_to_sell
                        oldest_lot['quantity'] -= remaining_to_sell
                        remaining_to_sell = 0
                
                # Positive cash flow for sales
                cash_flows.append(row['amount'])
                cash_flow_dates.append(row['trade_date'])
        
        # Calculate current holdings
        if fifo_queue:
            total_quantity = sum(lot['quantity'] for lot in fifo_queue)
            total_cost = sum(lot['quantity'] * lot['price'] for lot in fifo_queue)
            avg_price = total_cost / total_quantity if total_quantity > 0 else 0
            
            # Get current price
            current_price = current_prices.get(symbol, 0)
            current_value = total_quantity * current_price
            
            # Add current value as final cash flow for XIRR calculation
            xirr_cash_flows = cash_flows + [current_value]
            xirr_dates = cash_flow_dates + [datetime.now()]
            
            # Calculate metrics - FIX: Check if we have purchase transactions
            purchase_dates = [t.date() for t in cash_flow_dates if cash_flows[cash_flow_dates.index(t)] < 0]
            
            if purchase_dates:  # Only proceed if we have purchase transactions
                first_purchase = min(purchase_dates)
                years_held = (datetime.now().date() - first_purchase).days / 365.25
                
                # Calculate gains
                absolute_gain = current_value - total_cost
                absolute_gain_pct = (absolute_gain / total_cost * 100) if total_cost > 0 else 0
                
                # Calculate CAGR
                cagr = calculate_cagr(total_cost, current_value, years_held) * 100
                
                # Calculate XIRR
                xirr = calculate_xirr(xirr_cash_flows, xirr_dates)
                xirr_pct = xirr * 100 if xirr is not None else 0
                
                # Debug XIRR calculation
                if xirr is not None and (abs(xirr) > 10 or isinstance(xirr, complex)):
                    st.warning(f"Unusual XIRR value for {symbol}: {xirr}")
                    st.write(f"Cash flows: {xirr_cash_flows}")
                    st.write(f"Dates: {[d.strftime('%Y-%m-%d') for d in xirr_dates]}")
                    xirr_pct = 0  # Reset to 0 for display
                
                # Dividend gain (placeholder - add your dividend data)
                dividend_gain_pct = 0  # You can enhance this with actual dividend data
                
                # Total gain
                total_gain_pct = absolute_gain_pct + dividend_gain_pct
                
                holdings[symbol] = {
                    'quantity': total_quantity,
                    'avg_price': avg_price,
                    'current_price': current_price,
                    'total_invested': total_cost,
                    'current_value': current_value,
                    'absolute_gain': absolute_gain,
                    'absolute_gain_pct': absolute_gain_pct,
                    'dividend_gain_pct': dividend_gain_pct,
                    'total_gain_pct': total_gain_pct,
                    'years_held': years_held,
                    'cagr': cagr,
                    'xirr': xirr_pct,
                    'first_purchase_date': first_purchase,
                    'cash_flows': xirr_cash_flows,
                    'cash_flow_dates': xirr_dates
                }
            else:
                # Handle case where no purchases found (only sales)
                st.warning(f"No purchase transactions found for {symbol}. This stock will be excluded from analysis.")
                continue
    
    return holdings

def main():
    # Load data
    df = load_portfolio_data()
    if df is None:
        st.stop()
    
    st.success(f"Loaded {len(df)} transactions from database")
    
    # Debug: Show sample data
    st.subheader("📋 Sample Transaction Data")
    st.dataframe(df.head(10))
    
    # Debug: Show transaction type distribution
    st.subheader("📊 Transaction Type Distribution")
    type_counts = df['trade_type'].value_counts()
    st.write(type_counts)
    
    # Get unique symbols
    symbols = df['symbol'].unique()
    
    # Get last transaction prices for default values
    last_prices = get_last_transaction_prices(df)
    
    # Create a better structure for manual price input
    holdings_info = {}
    for symbol in symbols:
        holdings_info[symbol] = {'last_price': last_prices.get(symbol, 0)}
    
    # Get current prices manually
    current_prices = get_manual_prices(symbols, holdings_info)
    
    # Check if prices are provided
    if not any(current_prices.values()):
        st.warning("Please enter current prices for your stocks to calculate metrics.")
        st.stop()
    
    # Calculate holdings with metrics
    holdings = calculate_holdings_with_metrics(df, current_prices)
    
    if not holdings:
        st.warning("No current holdings found. This might happen if:")
        st.write("1. All stocks have been sold (no remaining quantities)")
        st.write("2. Only sell transactions exist without corresponding buy transactions")
        st.write("3. Data format issues with trade_type column")
        st.stop()
    
    # Create results DataFrame
    results = []
    total_invested = 0
    total_current_value = 0
    
    for symbol, holding in holdings.items():
        results.append({
            'Stock Code': symbol,
            'Weight': 0,  # Will calculate after getting total
            'Avg Years': holding['years_held'],
            'Abs Gain': holding['absolute_gain_pct'],
            'Div Gain': holding['dividend_gain_pct'],
            'Total Gain': holding['total_gain_pct'],
            'CAGR': holding['cagr'],
            'XIRR': holding['xirr'],
            'Current Value': holding['current_value'],
            'Total Invested': holding['total_invested'],
            'Quantity': holding['quantity'],
            'Avg Price': holding['avg_price'],
            'Current Price': holding['current_price']
        })
        
        total_invested += holding['total_invested']
        total_current_value += holding['current_value']
    
    # Calculate weights
    for result in results:
        result['Weight'] = (result['Current Value'] / total_current_value * 100) if total_current_value > 0 else 0
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by weight (descending)
    results_df = results_df.sort_values('Weight', ascending=False)
    
    # Display main metrics table
    st.subheader("📊 Portfolio Performance Metrics")
    
    # Summary metrics
    total_gain = total_current_value - total_invested
    total_gain_pct = (total_gain / total_invested * 100) if total_invested > 0 else 0
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Invested", f"₹{total_invested:,.0f}")
    with col2:
        st.metric("Current Value", f"₹{total_current_value:,.0f}")
    with col3:
        st.metric("Total Gain", f"₹{total_gain:,.0f}")
    with col4:
        st.metric("Total Gain %", f"{total_gain_pct:.2f}%")
    with col5:
        weighted_xirr = sum(r['XIRR'] * r['Weight'] / 100 for r in results if r['XIRR'])
        st.metric("Portfolio XIRR", f"{weighted_xirr:.2f}%")
    
    # Format and display main table
    display_df = results_df.copy()
    
    # Format columns for display
    display_df['Weight'] = display_df['Weight'].apply(lambda x: f"{x:.2f}%")
    display_df['Avg Years'] = display_df['Avg Years'].apply(lambda x: f"{x:.2f}")
    display_df['Abs Gain'] = display_df['Abs Gain'].apply(lambda x: f"{x:.2f}%")
    display_df['Div Gain'] = display_df['Div Gain'].apply(lambda x: f"{x:.2f}%")
    display_df['Total Gain'] = display_df['Total Gain'].apply(lambda x: f"{x:.2f}%")
    display_df['CAGR'] = display_df['CAGR'].apply(lambda x: f"{x:.2f}%")
    display_df['XIRR'] = display_df['XIRR'].apply(lambda x: f"{x:.2f}%" if x != 0 else "N/A")
    
    # Display main table (matching your screenshot format)
    main_display_df = display_df[['Stock Code', 'Weight', 'Avg Years', 'Abs Gain', 'Div Gain', 'Total Gain', 'CAGR', 'XIRR']]
    
    st.dataframe(main_display_df, use_container_width=True, hide_index=True)
    
    # Detailed holdings table
    st.subheader("📋 Detailed Holdings Information")
    
    detailed_df = results_df.copy()
    detailed_df['Current Value'] = detailed_df['Current Value'].apply(lambda x: f"₹{x:,.0f}")
    detailed_df['Total Invested'] = detailed_df['Total Invested'].apply(lambda x: f"₹{x:,.0f}")
    detailed_df['Avg Price'] = detailed_df['Avg Price'].apply(lambda x: f"₹{x:.2f}")
    detailed_df['Current Price'] = detailed_df['Current Price'].apply(lambda x: f"₹{x:.2f}")
    detailed_df['Quantity'] = detailed_df['Quantity'].apply(lambda x: f"{x:,.0f}")
    
    detailed_display_df = detailed_df[['Stock Code', 'Quantity', 'Avg Price', 'Current Price', 'Total Invested', 'Current Value']]
    
    st.dataframe(detailed_display_df, use_container_width=True, hide_index=True)
    
    # Visualizations
    st.subheader("📊 Portfolio Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Portfolio allocation pie chart
        fig_pie = px.pie(
            results_df, 
            values='Current Value', 
            names='Stock Code', 
            title='Portfolio Allocation by Current Value'
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # XIRR comparison
        xirr_data = results_df[results_df['XIRR'] != 0].copy()
        if not xirr_data.empty:
            fig_xirr = px.bar(
                xirr_data,
                x='Stock Code',
                y='XIRR',
                title='XIRR by Stock',
                color='XIRR',
                color_continuous_scale=['red', 'white', 'green']
            )
            fig_xirr.add_hline(y=0, line_dash="dash", line_color="black")
            st.plotly_chart(fig_xirr, use_container_width=True)
    
    # CAGR vs XIRR comparison
    comparison_data = results_df[results_df['XIRR'] != 0].copy()
    if not comparison_data.empty:
        fig_comparison = px.scatter(
            comparison_data,
            x='CAGR',
            y='XIRR',
            size='Current Value',
            color='Total Gain',
            hover_name='Stock Code',
            title='CAGR vs XIRR Comparison',
            color_continuous_scale=['red', 'white', 'green']
        )
        # Add diagonal line (y=x) to show where CAGR equals XIRR
        min_val = min(comparison_data['CAGR'].min(), comparison_data['XIRR'].min())
        max_val = max(comparison_data['CAGR'].max(), comparison_data['XIRR'].max())
        
        fig_comparison.add_shape(
            type="line",
            x0=min_val, y0=min_val,
            x1=max_val, y1=max_val,
            line=dict(color="gray", width=2, dash="dash")
        )
        st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Export functionality
    st.subheader("💾 Export Data")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Export Performance Metrics"):
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download Performance CSV",
                data=csv,
                file_name=f"portfolio_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("Export Transaction History"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Transactions CSV",
                data=csv,
                file_name=f"transaction_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    # Individual stock analysis
    st.subheader("🔍 Individual Stock Analysis")
    
    selected_stock = st.selectbox("Select a stock for detailed cash flow analysis:", list(holdings.keys()))
    
    if selected_stock:
        stock_data = holdings[selected_stock]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Performance Summary:**")
            st.write(f"- **Current Value:** ₹{stock_data['current_value']:,.0f}")
            st.write(f"- **Total Invested:** ₹{stock_data['total_invested']:,.0f}")
            st.write(f"- **Absolute Gain:** ₹{stock_data['absolute_gain']:,.0f} ({stock_data['absolute_gain_pct']:.2f}%)")
            st.write(f"- **CAGR:** {stock_data['cagr']:.2f}%")
            st.write(f"- **XIRR:** {stock_data['xirr']:.2f}%")
            st.write(f"- **Years Held:** {stock_data['years_held']:.2f}")
        
        with col2:
            st.write("**Current Position:**")
            st.write(f"- **Quantity:** {stock_data['quantity']:,.0f} shares")
            st.write(f"- **Average Price:** ₹{stock_data['avg_price']:.2f}")
            st.write(f"- **Current Price:** ₹{stock_data['current_price']:.2f}")
            st.write(f"- **Price Change:** {((stock_data['current_price'] - stock_data['avg_price']) / stock_data['avg_price'] * 100):.2f}%")
        
        # Cash flow analysis
        st.write("**Cash Flow Analysis:**")
        cash_flow_df = pd.DataFrame({
            'Date': [d.strftime('%Y-%m-%d') for d in stock_data['cash_flow_dates']],
            'Cash Flow': stock_data['cash_flows'],
            'Type': ['Purchase' if cf < 0 else 'Sale' if cf > 0 and d < datetime.now() else 'Current Value' 
                    for cf, d in zip(stock_data['cash_flows'], stock_data['cash_flow_dates'])]
        })
        cash_flow_df['Cash Flow'] = cash_flow_df['Cash Flow'].apply(lambda x: f"₹{x:,.0f}")
        st.dataframe(cash_flow_df, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()