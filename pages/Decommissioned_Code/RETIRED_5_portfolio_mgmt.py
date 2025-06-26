import streamlit as st
import psycopg
import pandas as pd
from datetime import datetime
import csv
import io
import locale
import decimal
import plotly.express as px

# Set locale for Indian number formatting
locale.setlocale(locale.LC_ALL, 'en_IN')

# Database configuration
DB_PARAMS = {
    'dbname': 'postgres',
    'user': 'postgres',
    'password': 'admin123',
    'host': 'localhost',
    'port': '5432'
}

def get_db_connection():
    """Establish connection to the PostgreSQL database"""
    return psycopg.connect(**DB_PARAMS)

def initialize_database():
    """Initialize database views and ensure table structure is correct"""
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            # Update transaction_type constraint
            cursor.execute("""
                ALTER TABLE portfolio_data 
                DROP CONSTRAINT IF EXISTS portfolio_data_transaction_type_check;
            """)
            
            cursor.execute("""
                ALTER TABLE portfolio_data 
                ADD CONSTRAINT portfolio_data_transaction_type_check 
                CHECK (transaction_type IN ('invest', 'redeem', 'switch_out', 'switch_in'));
            """)
            
            # Remove target scheme columns if they exist
            cursor.execute("""
                ALTER TABLE portfolio_data 
                DROP COLUMN IF EXISTS target_scheme_code,
                DROP COLUMN IF EXISTS target_scheme_name;
            """)
            conn.commit()
            
            # Create or replace the portfolio_holdings view with corrected calculations
            cursor.execute("""
                CREATE OR REPLACE VIEW portfolio_holdings AS
                WITH transaction_summary AS (
                    SELECT 
                        code,
                        scheme_name,
                        SUM(CASE 
                            WHEN transaction_type = 'invest' THEN units 
                            WHEN transaction_type = 'redeem' THEN -units 
                            WHEN transaction_type = 'switch_out' THEN -units 
                            WHEN transaction_type = 'switch_in' THEN units 
                            ELSE 0 
                        END) AS current_units,
                        SUM(CASE 
                            WHEN transaction_type = 'invest' THEN amount 
                            WHEN transaction_type = 'redeem' THEN -amount 
                            WHEN transaction_type = 'switch_out' THEN -amount 
                            WHEN transaction_type = 'switch_in' THEN amount 
                            ELSE 0 
                        END) AS total_investment
                    FROM portfolio_data
                    GROUP BY code, scheme_name
                )
                SELECT 
                    t.code,
                    t.scheme_name,
                    t.current_units,
                    COALESCE(m.value, 0) AS latest_nav,
                    CASE 
                        WHEN t.current_units > 0 THEN t.current_units * COALESCE(m.value, 0)
                        ELSE 0
                    END AS current_value,
                    m.nav AS nav_date,
                    t.total_investment
                FROM transaction_summary t
                LEFT JOIN LATERAL (
                    SELECT value, nav
                    FROM mutual_fund_nav
                    WHERE code = t.code
                    ORDER BY nav DESC
                    LIMIT 1
                ) m ON true
                WHERE t.current_units > 0
                ORDER BY t.current_units * COALESCE(m.value, 0) DESC;
            """)
            conn.commit()

def parse_date(date_str):
    """Parse date from string in multiple possible formats"""
    date_formats = [
        '%Y-%m-%d', '%m/%d/%Y', '%d-%m-%Y', '%d/%m/%Y', '%m-%d-%Y'
    ]
    for fmt in date_formats:
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Date '{date_str}' doesn't match any known format")

def get_scheme_list():
    """Get list of available mutual fund schemes"""
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT DISTINCT code, scheme_name 
                FROM mutual_fund_nav 
                ORDER BY scheme_name
            """)
            return cursor.fetchall()

def get_portfolio_schemes():
    """Get list of schemes currently in the portfolio"""
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT DISTINCT code, scheme_name 
                FROM portfolio_holdings
                ORDER BY scheme_name
            """)
            return cursor.fetchall()

def get_scheme_nav(scheme_code, date):
    """Get NAV for a specific scheme on a specific date"""
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT value 
                FROM mutual_fund_nav 
                WHERE code = %s AND nav <= %s
                ORDER BY nav DESC
                LIMIT 1
            """, (scheme_code, date))
            result = cursor.fetchone()
            return result[0] if result else None

def get_latest_nav(scheme_code):
    """Get the latest NAV for a scheme"""
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT value 
                FROM mutual_fund_nav 
                WHERE code = %s
                ORDER BY nav DESC
                LIMIT 1
            """, (scheme_code,))
            result = cursor.fetchone()
            return result[0] if result else None

def add_transaction(transaction_type, scheme_code, scheme_name, date, 
                   amount=None, units=None, value=None):
    """Add a transaction to the database"""
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            # Store absolute values - sign handled in view
            cursor.execute("""
                INSERT INTO portfolio_data (
                    date, scheme_name, code, transaction_type,
                    value, units, amount
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (
                date, scheme_name, scheme_code, transaction_type,
                abs(float(value)) if value is not None else None,
                abs(float(units)) if units is not None else None,
                abs(float(amount)) if amount is not None else None
            ))
            conn.commit()

def format_indian(number):
    """Format number with Indian comma separators"""
    try:
        return locale.format_string("%.2f", number, grouping=True)
    except:
        return str(number)

def parse_number(value):
    """Parse a number that might have commas or other formatting"""
    if isinstance(value, str):
        value = ''.join(c for c in value if c.isdigit() or c == '.' or c == '-')
        if not value:
            return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None

def get_portfolio_holdings():
    """Get current portfolio holdings"""
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT * FROM portfolio_holdings")
            columns = [desc[0] for desc in cursor.description]
            df = pd.DataFrame(cursor.fetchall(), columns=columns)
            for col in ['current_units', 'latest_nav', 'current_value', 'total_investment']:
                df[col] = df[col].astype(float).round(2)
            return df

def get_transaction_history():
    """Get all transactions"""
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT date, scheme_name, code, transaction_type,
                       value, units, amount
                FROM portfolio_data
                ORDER BY date DESC
            """)
            columns = [desc[0] for desc in cursor.description]
            df = pd.DataFrame(cursor.fetchall(), columns=columns)
            for col in ['value', 'units', 'amount']:
                if col in df.columns:
                    df[col] = df[col].apply(
                        lambda x: round(float(x), 2) if x is not None else None
                    )
            return df

def import_transactions_from_csv(uploaded_file):
    """Import transactions from CSV file"""
    try:
        content = uploaded_file.read().decode('utf-8-sig')
        csv_reader = csv.DictReader(io.StringIO(content))
        
        transactions = []
        for row in csv_reader:
            cleaned_row = {k.strip('\ufeff').strip(): v for k, v in row.items()}
            transactions.append(cleaned_row)
        
        success_count = 0
        error_count = 0
        errors = []
        
        for txn in transactions:
            try:
                transaction_type = txn.get('transaction_type', '').lower().strip()
                scheme_code = txn.get('code', '').strip()
                scheme_name = txn.get('scheme_name', '').strip()
                date_str = txn.get('date', '').strip()
                
                if not all([transaction_type, scheme_code, scheme_name, date_str]):
                    raise ValueError("Missing required fields")
                
                date = parse_date(date_str)
                amount = parse_number(txn.get('amount'))
                units = parse_number(txn.get('units'))
                value = parse_number(txn.get('value'))
                
                if transaction_type not in ['invest', 'redeem', 'switch_out', 'switch_in']:
                    raise ValueError(f"Invalid transaction type: {transaction_type}")
                
                add_transaction(
                    transaction_type=transaction_type,
                    scheme_code=scheme_code,
                    scheme_name=scheme_name,
                    date=date,
                    amount=amount,
                    units=units,
                    value=value
                )
                success_count += 1
            except Exception as e:
                error_count += 1
                errors.append(f"Error processing transaction: {txn}. Error: {str(e)}")
        
        return True, f"Imported {success_count} transactions successfully. {error_count} failed.", errors
    except Exception as e:
        return False, f"Failed to import transactions: {str(e)}", []

def main():
    st.set_page_config(page_title="Mutual Fund Portfolio Manager", layout="wide")
    st.title("Mutual Fund Portfolio Manager")
    
    initialize_database()
    all_schemes = get_scheme_list()
    portfolio_schemes = get_portfolio_schemes()
    
    all_scheme_dict = {f"{s[1]} ({s[0]})": s[0] for s in all_schemes}
    portfolio_scheme_dict = {f"{s[1]} ({s[0]})": s[0] for s in portfolio_schemes}
    
    all_scheme_names = list(all_scheme_dict.keys())
    portfolio_scheme_names = list(portfolio_scheme_dict.keys())
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Dashboard", "Add Investment", "Add Redemption", "Add Switch", "Import Transactions"
    ])
    
    with tab1:
        st.header("Portfolio Dashboard")
        # Add refresh button
        if st.button("Refresh Data"):
            st.rerun()
    
        st.subheader("Current Holdings")
        holdings = get_portfolio_holdings()
        
        if not holdings.empty:
            total_current = holdings['current_value'].sum()
            total_invested = holdings['total_investment'].sum()
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Schemes", len(holdings))
            col2.metric("Invested Value", f"₹{format_indian(total_invested)}")
            col3.metric("Current Value", f"₹{format_indian(total_current)}")
            
            display_df = holdings[[
                'scheme_name', 'code', 'current_units', 'latest_nav', 
                'current_value', 'total_investment', 'nav_date'
            ]]
            
            styled_df = display_df.style.format({
                'current_units': lambda x: format_indian(x),
                'latest_nav': lambda x: f"₹{format_indian(x)}",
                'current_value': lambda x: f"₹{format_indian(x)}",
                'total_investment': lambda x: f"₹{format_indian(x)}"
            })
            
            st.dataframe(styled_df, use_container_width=True)
            
            st.subheader("Portfolio Allocation")
            holdings_sorted = holdings.sort_values('current_value', ascending=True)
            fig = px.bar(
                holdings_sorted,
                x='current_value',
                y='scheme_name',
                orientation='h',
                title='Portfolio Allocation by Current Value',
                color='scheme_name',
                color_discrete_sequence=px.colors.qualitative.Set3,
                text='current_value'
            )
            fig.update_traces(
                texttemplate='₹%{text:,.2f}',
                textposition='outside'
            )
            fig.update_layout(
                showlegend=False,
                yaxis_title="Scheme",
                xaxis_title="Current Value (₹)",
                height=600,  # Adjust height as needed
                margin=dict(l=100, r=50, t=50, b=50)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No holdings found. Add transactions to see your portfolio.")
        
        st.subheader("Transaction History")
        transactions = get_transaction_history()
        if not transactions.empty:
            format_dict = {
                'amount': lambda x: f"₹{format_indian(x)}" if pd.notnull(x) else "",
                'units': lambda x: format_indian(x) if pd.notnull(x) else "",
                'value': lambda x: f"₹{format_indian(x)}" if pd.notnull(x) else ""
            }
            st.dataframe(
                transactions.style.format({k: v for k, v in format_dict.items() if k in transactions.columns}),
                use_container_width=True
            )
        else:
            st.info("No transactions found.")
    
    with tab2:
        st.header("Add New Investment")
        with st.form("investment_form"):
            selected_scheme = st.selectbox("Select Scheme", all_scheme_names)
            date = st.date_input("Transaction Date", datetime.today())
            
            col1, col2 = st.columns(2)
            amount = col1.number_input("Amount (₹)", min_value=0.0, step=1000.0, format="%.2f")
            units = col2.number_input("Units (optional)", min_value=0.0, step=0.01, format="%.2f")
            
            value = st.number_input("NAV (optional)", min_value=0.0, step=0.01, format="%.2f")
            
            submitted = st.form_submit_button("Add Investment")
            if submitted:
                scheme_code = all_scheme_dict[selected_scheme]
                scheme_name = selected_scheme.split(' (')[0]
                
                if not value:
                    value = get_scheme_nav(scheme_code, date) or get_latest_nav(scheme_code)
                    if isinstance(value, decimal.Decimal):
                        value = float(value)
                
                if not units and amount and value:
                    units = round(amount / value, 2)
                
                if not units and not amount:
                    st.error("Please provide either amount or units.")
                else:
                    add_transaction(
                        transaction_type="invest",
                        scheme_code=scheme_code,
                        scheme_name=scheme_name,
                        date=date,
                        amount=amount,
                        units=units,
                        value=value
                    )
                    st.success("Investment added successfully!")
    
    with tab3:
        st.header("Add Redemption")
        if not portfolio_schemes:
            st.warning("No schemes in your portfolio. Add investments first.")
        else:
            with st.form("redemption_form"):
                selected_scheme = st.selectbox("Select Scheme", portfolio_scheme_names)
                date = st.date_input("Transaction Date", datetime.today())
                
                col1, col2 = st.columns(2)
                amount = col1.number_input("Amount (₹ - optional)", min_value=0.0, step=1000.0, format="%.2f")
                units = col2.number_input("Units", min_value=0.0, step=0.01, format="%.2f")
                
                value = st.number_input("NAV (optional)", min_value=0.0, step=0.01, format="%.2f")
                
                submitted = st.form_submit_button("Add Redemption")
                if submitted:
                    scheme_code = portfolio_scheme_dict[selected_scheme]
                    scheme_name = selected_scheme.split(' (')[0]
                    
                    if not value:
                        value = get_scheme_nav(scheme_code, date) or get_latest_nav(scheme_code)
                        if isinstance(value, decimal.Decimal):
                            value = float(value)
                    
                    if not amount and units and value:
                        amount = round(units * value, 2)
                    
                    if not units:
                        st.error("Please provide units to redeem.")
                    else:
                        add_transaction(
                            transaction_type="redeem",
                            scheme_code=scheme_code,
                            scheme_name=scheme_name,
                            date=date,
                            amount=amount,
                            units=units,
                            value=value
                        )
                        st.success("Redemption added successfully!")
    
    with tab4:
        st.header("Add Switch Transaction")
        st.info("Switch transfers units from one scheme to another")
        
        if not portfolio_schemes:
            st.warning("No schemes in your portfolio. Add investments first.")
        else:
            with st.form("switch_form"):
                selected_scheme = st.selectbox("From Scheme", portfolio_scheme_names)
                date = st.date_input("Transaction Date", datetime.today())
                
                col1, col2 = st.columns(2)
                amount = col1.number_input("Amount (₹ - optional)", min_value=0.0, step=1000.0, format="%.2f")
                units = col2.number_input("Units", min_value=0.0, step=0.01, format="%.2f")
                
                value = st.number_input("NAV (optional)", min_value=0.0, step=0.01, format="%.2f")
                
                submitted = st.form_submit_button("Add Switch Out")
                if submitted:
                    scheme_code = portfolio_scheme_dict[selected_scheme]
                    scheme_name = selected_scheme.split(' (')[0]
                    
                    if not value:
                        value = get_scheme_nav(scheme_code, date) or get_latest_nav(scheme_code)
                        if isinstance(value, decimal.Decimal):
                            value = float(value)
                    
                    if not amount and units and value:
                        amount = round(units * value, 2)
                    
                    if not units:
                        st.error("Please provide units to switch.")
                    else:
                        add_transaction(
                            transaction_type="switch_out",
                            scheme_code=scheme_code,
                            scheme_name=scheme_name,
                            date=date,
                            amount=amount,
                            units=units,
                            value=value
                        )
                        st.success("Switch Out transaction added successfully!")
                        st.info("Now add a new investment (Switch In) for the target scheme in the 'Add Investment' tab")

    with tab5:
        st.header("Import Transactions from CSV")
        st.info("""
            Upload a CSV file with transactions. Required columns:
            - date (YYYY-MM-DD, MM/DD/YYYY, DD-MM-YYYY, etc.)
            - scheme_name
            - code (scheme code)
            - transaction_type (invest, redeem, switch_out, switch_in)
            Optional columns:
            - value (NAV)
            - units
            - amount
        """)
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            success, message, errors = import_transactions_from_csv(uploaded_file)
            if success:
                st.success(message)
                if errors:
                    st.warning("Some transactions had errors:")
                    for error in errors:
                        st.error(error)
            else:
                st.error(message)

if __name__ == "__main__":
    main()