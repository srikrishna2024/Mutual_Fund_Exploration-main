import streamlit as st
import pandas as pd
import numpy as np
import psycopg
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Portfolio Holdings Analysis", layout="wide")
st.title("📈 Portfolio Holdings Analysis")

st.markdown("""
This application shows your current stock holdings based on buy/sell transactions 
from your portfolio database.
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

def calculate_holdings_fifo(df):
    """Calculate current holdings using FIFO method for each stock"""
    holdings = {}
    
    for symbol in df['symbol'].unique():
        stock_data = df[df['symbol'] == symbol].sort_values('trade_date')
        
        # Create FIFO queue for buy transactions
        fifo_queue = []
        transactions_summary = []
        total_realized_pnl = 0
        
        for _, row in stock_data.iterrows():
            transaction = {
                'date': row['trade_date'],
                'type': row['trade_type'].lower(),
                'quantity': row['quantity'],
                'price': row['price'],
                'amount': row['amount']
            }
            transactions_summary.append(transaction)
            
            if row['trade_type'].lower() in ['buy', 'b']:
                # Add to FIFO queue
                fifo_queue.append({
                    'date': row['trade_date'],
                    'quantity': row['quantity'],
                    'price': row['price']
                })
                
            elif row['trade_type'].lower() in ['sell', 's']:
                # Process sell using FIFO
                remaining_to_sell = row['quantity']
                sell_price = row['price']
                
                while remaining_to_sell > 0 and fifo_queue:
                    oldest_lot = fifo_queue[0]
                    
                    if oldest_lot['quantity'] <= remaining_to_sell:
                        # Sell entire lot
                        sold_quantity = oldest_lot['quantity']
                        cost_price = oldest_lot['price']
                        realized_pnl = sold_quantity * (sell_price - cost_price)
                        total_realized_pnl += realized_pnl
                        
                        remaining_to_sell -= sold_quantity
                        fifo_queue.pop(0)
                    else:
                        # Partial sale of lot
                        sold_quantity = remaining_to_sell
                        cost_price = oldest_lot['price']
                        realized_pnl = sold_quantity * (sell_price - cost_price)
                        total_realized_pnl += realized_pnl
                        
                        oldest_lot['quantity'] -= remaining_to_sell
                        remaining_to_sell = 0
        
        # Calculate current holdings
        if fifo_queue:
            total_quantity = sum(lot['quantity'] for lot in fifo_queue)
            total_cost = sum(lot['quantity'] * lot['price'] for lot in fifo_queue)
            avg_price = total_cost / total_quantity if total_quantity > 0 else 0
            
            # Calculate investment timeline
            first_purchase = min(t['date'] for t in transactions_summary if t['type'] in ['buy', 'b'])
            last_transaction = max(t['date'] for t in transactions_summary)
            holding_period = (datetime.now() - first_purchase).days
            
            holdings[symbol] = {
                'quantity': total_quantity,
                'avg_price': avg_price,
                'total_invested': total_cost,
                'realized_pnl': total_realized_pnl,
                'first_purchase_date': first_purchase,
                'last_transaction_date': last_transaction,
                'holding_period_days': holding_period,
                'fifo_lots': fifo_queue,
                'all_transactions': transactions_summary
            }
    
    return holdings

def main():
    # Load data
    df = load_portfolio_data()
    if df is None:
        st.stop()
    
    st.success(f"Loaded {len(df)} transactions from database")
    
    # Show basic transaction summary
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Transactions", len(df))
    with col2:
        buy_transactions = len(df[df['trade_type'].str.lower().isin(['buy', 'b'])])
        st.metric("Buy Transactions", buy_transactions)
    with col3:
        sell_transactions = len(df[df['trade_type'].str.lower().isin(['sell', 's'])])
        st.metric("Sell Transactions", sell_transactions)
    with col4:
        st.metric("Unique Stocks", df['symbol'].nunique())
    
    # Calculate holdings
    holdings = calculate_holdings_fifo(df)
    
    if not holdings:
        st.warning("No current holdings found")
        st.stop()
    
    st.subheader("📊 Current Portfolio Holdings")
    
    # Create results list
    results = []
    total_invested = 0
    total_realized_pnl = 0
    
    for symbol, holding in holdings.items():
        quantity = holding['quantity']
        avg_price = holding['avg_price']
        invested = holding['total_invested']
        realized_pnl = holding['realized_pnl']
        holding_period = holding['holding_period_days']
        
        results.append({
            'Symbol': symbol,
            'Quantity': quantity,
            'Avg Price': avg_price,
            'Total Invested': invested,
            'Realized P&L': realized_pnl,
            'First Purchase': holding['first_purchase_date'].strftime('%Y-%m-%d'),
            'Last Transaction': holding['last_transaction_date'].strftime('%Y-%m-%d'),
            'Holding Period (Days)': holding_period,
            'Holding Period (Years)': round(holding_period / 365.25, 1)
        })
        
        total_invested += invested
        total_realized_pnl += realized_pnl
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Display summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current Holdings", len(holdings))
    with col2:
        st.metric("Total Invested", f"₹{total_invested:,.0f}")
    with col3:
        st.metric("Total Realized P&L", f"₹{total_realized_pnl:,.0f}")
    with col4:
        avg_holding_period = results_df['Holding Period (Days)'].mean()
        st.metric("Avg Holding Period", f"{avg_holding_period:.0f} days")
    
    # Display holdings table
    st.subheader("📋 Detailed Holdings")
    
    # Format the dataframe for display
    display_df = results_df.copy()
    display_df['Avg Price'] = display_df['Avg Price'].apply(lambda x: f"₹{x:.2f}")
    display_df['Total Invested'] = display_df['Total Invested'].apply(lambda x: f"₹{x:,.0f}")
    display_df['Realized P&L'] = display_df['Realized P&L'].apply(lambda x: f"₹{x:,.0f}")
    
    st.dataframe(display_df, use_container_width=True)
    
    # Visualizations
    st.subheader("📊 Portfolio Visualizations")
    
    # Portfolio allocation pie chart
    col1, col2 = st.columns(2)
    
    with col1:
        fig_pie = px.pie(
            results_df, 
            values='Total Invested', 
            names='Symbol', 
            title='Portfolio Allocation by Investment Amount'
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Realized P&L bar chart
        fig_bar = px.bar(
            results_df,
            x='Symbol',
            y='Realized P&L',
            title='Realized P&L by Stock',
            color='Realized P&L',
            color_continuous_scale=['red', 'white', 'green']
        )
        fig_bar.add_hline(y=0, line_dash="dash", line_color="black")
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Holding period analysis
    fig_holding = px.scatter(
        results_df,
        x='Holding Period (Years)',
        y='Total Invested',
        size='Quantity',
        color='Realized P&L',
        hover_name='Symbol',
        title='Holding Period vs Investment Amount',
        color_continuous_scale=['red', 'white', 'green']
    )
    st.plotly_chart(fig_holding, use_container_width=True)
    
    # Detailed stock analysis
    st.subheader("🔍 Detailed Stock Analysis")
    
    selected_stock = st.selectbox("Select a stock for detailed analysis:", list(holdings.keys()))
    
    if selected_stock:
        stock_details = holdings[selected_stock]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Current Holdings Summary:**")
            st.write(f"- **Quantity:** {stock_details['quantity']:,.0f} shares")
            st.write(f"- **Average Price:** ₹{stock_details['avg_price']:.2f}")
            st.write(f"- **Total Invested:** ₹{stock_details['total_invested']:,.0f}")
            st.write(f"- **Realized P&L:** ₹{stock_details['realized_pnl']:,.0f}")
            st.write(f"- **Holding Period:** {stock_details['holding_period_days']} days ({stock_details['holding_period_days']/365.25:.1f} years)")
        
        with col2:
            st.write("**FIFO Lots Breakdown:**")
            lots_df = pd.DataFrame(stock_details['fifo_lots'])
            if not lots_df.empty:
                lots_df['Date'] = pd.to_datetime(lots_df['date']).dt.strftime('%Y-%m-%d')
                lots_df['Investment'] = (lots_df['quantity'] * lots_df['price']).apply(lambda x: f"₹{x:,.0f}")
                lots_df['Price'] = lots_df['price'].apply(lambda x: f"₹{x:.2f}")
                st.dataframe(lots_df[['Date', 'quantity', 'Price', 'Investment']], use_container_width=True)
        
        # Transaction history for selected stock
        st.write("**Transaction History:**")
        transactions_df = pd.DataFrame(stock_details['all_transactions'])
        transactions_df['Date'] = pd.to_datetime(transactions_df['date']).dt.strftime('%Y-%m-%d')
        transactions_df['Price'] = transactions_df['price'].apply(lambda x: f"₹{x:.2f}")
        transactions_df['Amount'] = transactions_df['amount'].apply(lambda x: f"₹{x:,.0f}")
        
        display_transactions = transactions_df[['Date', 'type', 'quantity', 'Price', 'Amount']].copy()
        display_transactions.columns = ['Date', 'Type', 'Quantity', 'Price', 'Amount']
        st.dataframe(display_transactions, use_container_width=True)
    
    # Summary statistics
    st.subheader("📈 Portfolio Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Top Holdings by Investment:**")
        top_investments = results_df.nlargest(5, 'Total Invested')[['Symbol', 'Total Invested']]
        for _, row in top_investments.iterrows():
            investment_pct = (row['Total Invested'] / total_invested) * 100
            st.write(f"**{row['Symbol']}**: ₹{row['Total Invested']:,.0f} ({investment_pct:.1f}%)")
    
    with col2:
        st.write("**Best Realized P&L:**")
        best_pnl = results_df.nlargest(5, 'Realized P&L')[['Symbol', 'Realized P&L']]
        for _, row in best_pnl.iterrows():
            st.write(f"**{row['Symbol']}**: ₹{row['Realized P&L']:,.0f}")
    
    with col3:
        st.write("**Longest Holdings:**")
        longest_holdings = results_df.nlargest(5, 'Holding Period (Days)')[['Symbol', 'Holding Period (Years)']]
        for _, row in longest_holdings.iterrows():
            st.write(f"**{row['Symbol']}**: {row['Holding Period (Years)']} years")
    
    # Export functionality
    st.subheader("💾 Export Data")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Export Holdings to CSV"):
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download Holdings CSV",
                data=csv,
                file_name=f"portfolio_holdings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("Export All Transactions to CSV"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Transactions CSV",
                data=csv,
                file_name=f"all_transactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()