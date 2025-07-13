import streamlit as st
import pandas as pd
import psycopg
from io import StringIO
import tempfile
from datetime import datetime

# Database connection parameters from the existing script
DB_PARAMS = {
    'dbname': 'postgres',
    'user': 'postgres',
    'password': 'admin123',
    'host': 'localhost',
    'port': '5432'
}

def connect_to_db():
    """Create database connection"""
    return psycopg.connect(**DB_PARAMS)

def create_portfolio_table():
    """Create portfolio_data table if it doesn't exist"""
    with connect_to_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_data (
                    id SERIAL PRIMARY KEY,
                    date DATE NOT NULL,
                    scheme_name VARCHAR(255) NOT NULL,
                    code VARCHAR(50) NOT NULL,
                    transaction_type VARCHAR(10) CHECK (transaction_type IN ('invest', 'switch', 'redeem')),
                    value NUMERIC(20, 4),
                    units NUMERIC(20, 4),
                    amount NUMERIC(20, 2),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            conn.commit()

def clean_numeric_data(df):
    """Clean numeric columns by removing commas and converting to numeric"""
    # Remove commas and convert to numeric for value, units, and amount columns
    numeric_columns = ['value', 'units', 'amount']
    for col in numeric_columns:
        df[col] = df[col].astype(str).str.replace(',', '').str.replace('₹', '').str.strip()
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove commas from code column
    df['code'] = df['code'].astype(str).str.replace(',', '')
    
    return df

def convert_date_format(df):
    """Convert date from MM/DD/YYYY to YYYY-MM-DD format"""
    try:
        # Convert date strings to datetime objects and then to desired format
        df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y').dt.strftime('%Y-%m-%d')
        # Convert back to datetime.date objects for database compatibility
        df['Date'] = pd.to_datetime(df['Date']).dt.date
        return df
    except Exception as e:
        raise Exception(f"Error converting date format: {str(e)}")

def standardize_column_names(df):
    """Standardize column names to expected format"""
    # Create a mapping of possible variations to standard names
    column_mapping = {
        'date': 'Date',
        'scheme_name': 'scheme_name',
        'code': 'code',
        'transaction_type': 'Transaction Type',
        'transaction': 'Transaction Type',
        'type': 'Transaction Type',
        'value': 'value',
        'units': 'units',
        'amount': 'amount'
    }
    
    # Convert column names to lowercase for case-insensitive matching
    df.columns = [col.lower().strip() for col in df.columns]
    
    # Rename columns based on mapping
    df = df.rename(columns=column_mapping)
    
    return df

def validate_dataframe(df):
    """Validate the uploaded dataframe format and data"""
    try:
        # Standardize column names
        df = standardize_column_names(df)
        
        required_columns = ['Date', 'scheme_name', 'code', 'Transaction Type', 'value', 'units', 'amount']
        
        # Check if all required columns exist
        if not all(col in df.columns for col in required_columns):
            missing_cols = [col for col in required_columns if col not in df.columns]
            return False, f"Missing columns: {', '.join(missing_cols)}", None

        # Convert date format
        try:
            df = convert_date_format(df)
        except Exception as e:
            return False, f"Date format error: {str(e)}", None
        
        # Check transaction types
        valid_types = {'invest', 'switch', 'redeem'}
        invalid_types = set(df['Transaction Type'].str.lower().unique()) - valid_types
        if invalid_types:
            return False, f"Invalid transaction types found: {', '.join(invalid_types)}. Allowed types: invest, switch, redeem", None
        
        # Standardize transaction types to lowercase
        df['Transaction Type'] = df['Transaction Type'].str.lower()
        
        # Clean and validate numeric data
        try:
            df = clean_numeric_data(df)
            
            # Check for any remaining NaN values in numeric columns
            numeric_columns = ['value', 'units', 'amount']
            for col in numeric_columns:
                if df[col].isna().any():
                    invalid_rows = df[df[col].isna()].index.tolist()
                    return False, f"Invalid numeric values found in {col} column at rows: {invalid_rows}", None
            
            return True, "Validation successful", df
        except Exception as e:
            return False, f"Error processing numeric data: {str(e)}", None
            
    except Exception as e:
        return False, f"Error validating dataframe: {str(e)}", None

def insert_portfolio_data(df):
    """Insert validated data into portfolio_data table"""
    with connect_to_db() as conn:
        with conn.cursor() as cur:
            # Prepare the values for insertion
            values = [
                (
                    row['Date'],
                    row['scheme_name'],
                    row['code'],
                    row['Transaction Type'],
                    float(row['value']),
                    float(row['units']),
                    float(row['amount'])
                )
                for _, row in df.iterrows()
            ]
            
            # Insert all rows using executemany
            cur.executemany("""
                INSERT INTO portfolio_data 
                (date, scheme_name, code, transaction_type, value, units, amount)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, values)
            
            conn.commit()

def get_portfolio_data():
    """Retrieve all records from portfolio_data table"""
    with connect_to_db() as conn:
        query = """
            SELECT date, scheme_name, code, transaction_type, value, units, amount 
            FROM portfolio_data 
            ORDER BY date DESC, created_at DESC
        """
        return pd.read_sql(query, conn)

def download_portfolio_data(df, file_format):
    """Generate downloadable file from portfolio data"""
    if df.empty:
        return None
    
    if file_format == 'CSV':
        # Convert DataFrame to CSV
        csv = df.to_csv(index=False)
        return csv.encode('utf-8')
    else:  # XLSX
        # Create Excel file in memory
        output = StringIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Portfolio Data')
        return output.getvalue()

def main():
    """
    Main function to handle the Portfolio Transaction Upload page.
    """
    st.set_page_config(page_title='Portfolio Transaction Upload', layout='wide')
    st.title('Portfolio Transaction Upload')
    
    # Create table if it doesn't exist
    create_portfolio_table()
    
    # File upload section
    st.subheader('Upload Transaction Data')
    st.info("""
    Expected columns and formats:
    - Date: MM/DD/YYYY format (e.g., 01/31/2024)
    - scheme_name: Name of the scheme
    - code: Scheme code
    - Transaction Type: 'invest', 'switch', or 'redeem'
    - value: Numeric value
    - units: Number of units
    - amount: Transaction amount
    """)
    
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=['csv', 'xlsx'])
    
    if uploaded_file is not None:
        try:
            # Read the file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Show raw data
            st.subheader('Raw Data Preview')
            st.dataframe(df)
            
            # Validate data
            is_valid, message, cleaned_df = validate_dataframe(df)
            
            # Show preview of the processed data
            st.subheader('Processed Data Preview')
            if is_valid:
                st.success(message)
                st.dataframe(cleaned_df)
                
                # Process button
                if st.button('Insert Data to Database'):
                    try:
                        insert_portfolio_data(cleaned_df)
                        st.success(f'Successfully inserted {len(cleaned_df)} records into the database!')
                    except Exception as e:
                        st.error(f'Error inserting data: {str(e)}')
            else:
                st.error(message)
                
        except Exception as e:
            st.error(f'Error processing file: {str(e)}')
    
    # Show existing records
    st.subheader('Current Portfolio Data')
    portfolio_data = get_portfolio_data()
    if not portfolio_data.empty:
        st.dataframe(portfolio_data)
        
        # Download section
        st.subheader('Download Portfolio Data')
        col1, col2 = st.columns([1, 2])
        with col1:
            file_format = st.selectbox('Select Format', ['CSV', 'XLSX'])
        with col2:
            if st.button('Download Data'):
                file_data = download_portfolio_data(portfolio_data, file_format)
                
                if file_data is not None:
                    file_extension = 'csv' if file_format == 'CSV' else 'xlsx'
                    file_name = f'portfolio_data.{file_extension}'
                    
                    # Create download button
                    st.download_button(
                        label=f"Click here to download {file_format}",
                        data=file_data,
                        file_name=file_name,
                        mime='text/csv' if file_format == 'CSV' else 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                    )
    else:
        st.info('No records found in the database. Please upload a file to add data.')

if __name__ == "__main__":
    main()