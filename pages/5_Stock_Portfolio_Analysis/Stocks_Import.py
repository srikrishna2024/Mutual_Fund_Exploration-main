import streamlit as st
import pandas as pd
import psycopg
from datetime import datetime

st.set_page_config(page_title="Stock Portfolio Uploader", layout="wide")
st.title("Stock Portfolio Database Uploader")

# Database connection parameters
DB_PARAMS = {
    'dbname': 'postgres',
    'user': 'postgres',
    'password': 'admin123',
    'host': 'localhost',
    'port': '5432'
}

def connect_to_db():
    """Establish connection to PostgreSQL database"""
    return psycopg.connect(**DB_PARAMS)

def create_stocks_table():
    """Create portfolio_stocks table if it doesn't exist"""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS portfolio_stocks (
        id SERIAL PRIMARY KEY,
        symbol VARCHAR(20) NOT NULL,
        isin VARCHAR(20),
        trade_date DATE NOT NULL,
        exchange VARCHAR(10),
        segment VARCHAR(20),
        series VARCHAR(5),
        trade_type VARCHAR(10) NOT NULL,
        auction VARCHAR(5),
        quantity NUMERIC NOT NULL,
        price NUMERIC NOT NULL,
        trade_id VARCHAR(50),
        order_id VARCHAR(50),
        order_execution_time TIMESTAMP,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """
    try:
        with connect_to_db() as conn:
            with conn.cursor() as cur:
                cur.execute(create_table_sql)
                conn.commit()
        st.success("portfolio_stocks table is ready")
    except Exception as e:
        st.error(f"Error creating table: {str(e)}")

def get_latest_execution_time():
    """Get the most recent order_execution_time from the database"""
    try:
        with connect_to_db() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT MAX(order_execution_time) 
                    FROM portfolio_stocks
                    WHERE order_execution_time IS NOT NULL
                """)
                result = cur.fetchone()
                return result[0] if result else None
    except Exception as e:
        st.error(f"Error fetching latest execution time: {str(e)}")
        return None

def insert_stocks_data(df):
    """Insert stock transactions into database"""
    insert_sql = """
    INSERT INTO portfolio_stocks (
        symbol, isin, trade_date, exchange, segment, series,
        trade_type, auction, quantity, price, trade_id, order_id, order_execution_time
    ) VALUES (
        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
    )
    """
    try:
        with connect_to_db() as conn:
            with conn.cursor() as cur:
                # Convert DataFrame to list of tuples for executemany
                data = df[[
                    'symbol', 'isin', 'trade_date', 'exchange', 'segment', 'series',
                    'trade_type', 'auction', 'quantity', 'price', 'trade_id', 'order_id', 'order_execution_time'
                ]].values.tolist()
                
                cur.executemany(insert_sql, data)
                conn.commit()
        return True, len(df)
    except Exception as e:
        return False, f"Error inserting data: {str(e)}"

def load_and_validate_data(file):
    """Load and validate the uploaded file"""
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        
        # Standardize column names (case insensitive)
        df.columns = df.columns.str.lower()
        
        # Check for required columns
        required_columns = {'symbol', 'trade_date', 'trade_type', 'quantity', 'price'}
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            st.error(f"Missing required columns: {', '.join(missing)}")
            return None
        
        # Convert date columns to datetime
        date_cols = ['trade_date', 'order_execution_time']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Convert numeric columns
        numeric_cols = ['quantity', 'price']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Fill empty strings with None for database
        df = df.replace('', None)
        
        return df
    
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

def filter_new_records(df, cutoff_time):
    """Filter records to only those after the cutoff time"""
    if cutoff_time is None or 'order_execution_time' not in df.columns:
        return df
    
    # Convert cutoff_time to pandas Timestamp if it's not already
    cutoff_time = pd.to_datetime(cutoff_time)
    
    # Filter records
    new_records = df[df['order_execution_time'] > cutoff_time]
    
    if len(new_records) < len(df):
        st.info(f"Filtered out {len(df) - len(new_records)} existing records (already in database)")
    
    return new_records

# Main application
def main():
    st.markdown("""
    This tool uploads your stock portfolio transactions to the database.
    Supported file formats: CSV, Excel (XLS/XLSX)
    Required columns: symbol, trade_date, trade_type, quantity, price
    """)
    
    # Create the table if it doesn't exist
    create_stocks_table()
    
    # File upload section
    st.subheader("📂 Upload Your Stock Transactions")
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["csv", "xlsx", "xls"],
        key="stocks_uploader"
    )
    
    if uploaded_file:
        with st.spinner("Processing your file..."):
            df = load_and_validate_data(uploaded_file)
            
            if df is not None:
                st.success(f"File loaded successfully with {len(df)} records")
                
                # Show preview
                with st.expander("Preview your data"):
                    st.dataframe(df.head())
                
                # Check for duplicates in the file itself
                if st.checkbox("Check for duplicate trade_id entries within this file"):
                    if 'trade_id' in df.columns:
                        duplicates = df[df.duplicated('trade_id', keep=False)]
                        if not duplicates.empty:
                            st.warning(f"Found {len(duplicates)} duplicate trade_id entries in file")
                            st.dataframe(duplicates)
                        else:
                            st.success("No duplicate trade_id entries found in file")
                    else:
                        st.info("trade_id column not available for duplicate check")
                
                # Check for existing records in database
                if st.checkbox("Check for existing records in database (by order_execution_time)", True):
                    latest_time = get_latest_execution_time()
                    
                    if latest_time:
                        st.info(f"Latest order_execution_time in database: {latest_time}")
                        
                        # Filter to only new records
                        df = filter_new_records(df, latest_time)
                        
                        if df.empty:
                            st.warning("No new records to import - all records already exist in database")
                            return
                    else:
                        st.info("No existing records found in database - will import all records")
                
                # Upload to database
                if st.button("Upload to Database") and not df.empty:
                    with st.spinner("Uploading data to database..."):
                        success, result = insert_stocks_data(df)
                        
                        if success:
                            st.success(f"Successfully inserted {result} records into portfolio_stocks table")
                            
                            # Show success metrics
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Total Records Uploaded", result)
                            with col2:
                                st.metric("Unique Stocks", df['symbol'].nunique())
                            
                            # Show date range
                            min_date = df['trade_date'].min().strftime('%Y-%m-%d')
                            max_date = df['trade_date'].max().strftime('%Y-%m-%d')
                            st.info(f"Date range in uploaded data: {min_date} to {max_date}")
                            
                            # Show earliest and latest execution times
                            if 'order_execution_time' in df.columns:
                                min_exec = df['order_execution_time'].min().strftime('%Y-%m-%d %H:%M:%S')
                                max_exec = df['order_execution_time'].max().strftime('%Y-%m-%d %H:%M:%S')
                                st.info(f"Execution time range: {min_exec} to {max_exec}")
                        else:
                            st.error(result)

if __name__ == "__main__":
    main()