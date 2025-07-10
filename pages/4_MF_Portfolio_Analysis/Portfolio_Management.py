import streamlit as st
import psycopg
import pandas as pd
from datetime import datetime
import csv
import io
import locale
import decimal
import plotly.express as px
from pandas.tseries.offsets import BDay

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

def format_indian_currency(amount):
    """Format amount in Indian currency style (lakhs, crores)"""
    def format_number(num):
        if num < 0:
            return f"-{format_number(abs(num))}"
        if num < 1000:
            return str(num)
        elif num < 100000:
            return f"{num/1000:.2f}K"
        elif num < 10000000:
            return f"{num/100000:.2f}L"
        else:
            return f"{num/10000000:.2f}Cr"
    
    return f"₹{format_number(float(amount))}"

def check_and_update_goals_schema():
    """Check if goals table exists and has required columns, update if necessary"""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'goals'
                );
            """)
            table_exists = cur.fetchone()[0]
            
            if not table_exists:
                cur.execute("""
                    CREATE TABLE goals (
                        id SERIAL PRIMARY KEY,
                        goal_name VARCHAR(100),
                        investment_type VARCHAR(20),
                        scheme_name VARCHAR(200),
                        scheme_code VARCHAR(50),
                        current_value DECIMAL(15,2),
                        is_manual_entry BOOLEAN DEFAULT FALSE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
            else:
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.columns 
                        WHERE table_name = 'goals' AND column_name = 'is_manual_entry'
                    );
                """)
                has_manual_entry = cur.fetchone()[0]
                
                if not has_manual_entry:
                    cur.execute("""
                        ALTER TABLE goals 
                        ADD COLUMN is_manual_entry BOOLEAN DEFAULT FALSE
                    """)
            
        conn.commit()

def get_portfolio_data_for_goals():
    """Retrieve current portfolio data with latest NAVs for goal mapping"""
    with get_db_connection() as conn:
        query = """
            WITH transaction_units AS (
                SELECT 
                    scheme_name, 
                    code,
                    CASE 
                        WHEN transaction_type IN ('switch_out', 'redeem') THEN -units
                        WHEN transaction_type IN ('invest', 'switch_in') THEN units
                        ELSE 0 
                    END as units_change
                FROM portfolio_data
            ),
            cumulative_units AS (
                SELECT 
                    scheme_name,
                    code,
                    SUM(units_change) as total_units
                FROM transaction_units
                GROUP BY scheme_name, code
                HAVING SUM(units_change) > 0
            ),
            latest_nav AS (
                SELECT code, value as nav_value
                FROM mutual_fund_nav
                WHERE (code, nav) IN (
                    SELECT code, MAX(nav)
                    FROM mutual_fund_nav
                    GROUP BY code
                )
            )
            SELECT 
                cu.scheme_name,
                cu.code as scheme_code,
                cu.total_units * ln.nav_value as current_value
            FROM cumulative_units cu
            JOIN latest_nav ln ON cu.code = ln.code
        """
        return pd.read_sql(query, conn)

def check_existing_mapping(scheme_name, scheme_code):
    """Check if a fund is already mapped to any goal"""
    with get_db_connection() as conn:
        query = """
            SELECT goal_name 
            FROM goals 
            WHERE scheme_name = %s AND scheme_code = %s
            AND (is_manual_entry IS NULL OR is_manual_entry = FALSE)
        """
        df = pd.read_sql(query, conn, params=(scheme_name, scheme_code))
        return df['goal_name'].iloc[0] if not df.empty else None

def insert_goal_mapping(goal_name, investment_type, scheme_name, scheme_code, current_value, is_manual_entry=False):
    """Insert a new goal mapping into the goals table"""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO goals 
                    (goal_name, investment_type, scheme_name, scheme_code, current_value, is_manual_entry)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (goal_name, investment_type, scheme_name, scheme_code, current_value, is_manual_entry))
                
                cur.fetchone()
                conn.commit()
                return True
    except Exception as e:
        print(f"Error inserting record: {str(e)}")
        return False

def get_existing_goals():
    """Retrieve existing goal mappings"""
    with get_db_connection() as conn:
        query = """
            SELECT 
                goal_name, 
                investment_type, 
                scheme_name, 
                scheme_code, 
                current_value,
                COALESCE(is_manual_entry, FALSE) as is_manual_entry
            FROM goals
            ORDER BY goal_name, scheme_name
        """
        return pd.read_sql(query, conn)

def initialize_database():
    """Initialize database views and ensure table structure is correct"""
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                ALTER TABLE portfolio_data 
                DROP CONSTRAINT IF EXISTS portfolio_data_transaction_type_check;
            """)
            
            cursor.execute("""
                ALTER TABLE portfolio_data 
                ADD CONSTRAINT portfolio_data_transaction_type_check 
                CHECK (transaction_type IN ('invest', 'redeem', 'switch_out', 'switch_in'));
            """)
            
            cursor.execute("""
                ALTER TABLE portfolio_data 
                DROP COLUMN IF EXISTS target_scheme_code,
                DROP COLUMN IF EXISTS target_scheme_name;
            """)
            conn.commit()
            
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
                ),
                latest_nav AS (
                    SELECT 
                        code,
                        value,
                        nav
                    FROM (
                        SELECT 
                            code,
                            value,
                            nav,
                            ROW_NUMBER() OVER (PARTITION BY code ORDER BY nav DESC) as rn
                        FROM mutual_fund_nav
                    ) t
                    WHERE rn = 1
                )
                SELECT 
                    t.code,
                    t.scheme_name,
                    t.current_units,
                    COALESCE(l.value, 0) AS latest_nav,
                    CASE 
                        WHEN t.current_units > 0 THEN t.current_units * COALESCE(l.value, 0)
                        ELSE 0
                    END AS current_value,
                    l.nav AS nav_date,
                    t.total_investment
                FROM transaction_summary t
                LEFT JOIN latest_nav l ON t.code = l.code
                WHERE t.current_units > 0
                ORDER BY t.current_units * COALESCE(l.value, 0) DESC;
            """)
            conn.commit()
            
            check_and_update_goals_schema()

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
    """Get current portfolio holdings with goal names"""
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT 
                    ph.*,
                    g.goal_name
                FROM portfolio_holdings ph
                LEFT JOIN goals g ON ph.code = g.scheme_code
                ORDER BY ph.current_value DESC
            """)
            columns = [desc[0] for desc in cursor.description]
            df = pd.DataFrame(cursor.fetchall(), columns=columns)
            for col in ['current_units', 'latest_nav', 'current_value', 'total_investment']:
                df[col] = df[col].astype(float).round(2)
            return df

def get_fund_performance():
    """Get performance metrics for each fund with fallback to latest NAV"""
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT CURRENT_DATE")
            current_date = cursor.fetchone()[0]
            
            cursor.execute("SELECT code, scheme_name FROM portfolio_holdings")
            funds = cursor.fetchall()
            
            performance_data = []
            
            for fund_code, fund_name in funds:
                cursor.execute("""
                    SELECT current_value FROM portfolio_holdings WHERE code = %s
                """, (fund_code,))
                current_value = cursor.fetchone()[0] or 0
                
                def get_historical_fund_value(target_date):
                    query = """
                        WITH holdings_at_date AS (
                            SELECT 
                                SUM(CASE 
                                    WHEN transaction_type = 'invest' THEN units 
                                    WHEN transaction_type = 'redeem' THEN -units 
                                    WHEN transaction_type = 'switch_out' THEN -units 
                                    WHEN transaction_type = 'switch_in' THEN units 
                                    ELSE 0 
                                END) AS units_held
                            FROM portfolio_data
                            WHERE code = %s AND date <= %s
                        ),
                        nav_at_date AS (
                            SELECT value as nav_value
                            FROM mutual_fund_nav
                            WHERE code = %s AND nav <= %s
                            ORDER BY nav DESC
                            LIMIT 1
                        )
                        SELECT 
                            COALESCE(h.units_held * n.nav_value, 0)
                        FROM holdings_at_date h, nav_at_date n
                    """
                    cursor.execute(query, (fund_code, target_date, fund_code, target_date))
                    result = cursor.fetchone()[0]
                    
                    if result == 0:
                        cursor.execute("""
                            SELECT value 
                            FROM mutual_fund_nav 
                            WHERE code = %s AND nav < %s
                            ORDER BY nav DESC 
                            LIMIT 1
                        """, (fund_code, target_date))
                        fallback_nav = cursor.fetchone()
                        if fallback_nav:
                            cursor.execute("""
                                SELECT 
                                    SUM(CASE 
                                        WHEN transaction_type = 'invest' THEN units 
                                        WHEN transaction_type = 'redeem' THEN -units 
                                        WHEN transaction_type = 'switch_out' THEN -units 
                                        WHEN transaction_type = 'switch_in' THEN units 
                                        ELSE 0 
                                    END)
                                FROM portfolio_data
                                WHERE code = %s AND date <= %s
                            """, (fund_code, target_date))
                            units = cursor.fetchone()[0] or 0
                            result = units * fallback_nav[0]
                    
                    return result
                
                if current_date.weekday() == 0:
                    prev_day = current_date - BDay(1)
                else:
                    prev_day = current_date - BDay(1)
                
                daily_value = get_historical_fund_value(prev_day)
                week_ago = current_date - BDay(7)
                weekly_value = get_historical_fund_value(week_ago)
                month_ago = current_date - BDay(30)
                monthly_value = get_historical_fund_value(month_ago)
                ytd_date = datetime(current_date.year, 1, 1).date()
                ytd_value = get_historical_fund_value(ytd_date)
                
                cursor.execute("SELECT goal_name FROM goals WHERE scheme_code = %s LIMIT 1", (fund_code,))
                goal_result = cursor.fetchone()
                goal_name = goal_result[0] if goal_result else None
                
                performance_data.append({
                    'scheme_name': fund_name,
                    'code': fund_code,
                    'goal_name': goal_name,
                    'current_value': current_value,
                    'daily_change': current_value - daily_value,
                    'daily_pct': ((current_value - daily_value) / daily_value * 100) if daily_value != 0 else 0,
                    'weekly_change': current_value - weekly_value,
                    'weekly_pct': ((current_value - weekly_value) / weekly_value * 100) if weekly_value != 0 else 0,
                    'monthly_change': current_value - monthly_value,
                    'monthly_pct': ((current_value - monthly_value) / monthly_value * 100) if monthly_value != 0 else 0,
                    'ytd_change': current_value - ytd_value,
                    'ytd_pct': ((current_value - ytd_value) / ytd_value * 100) if ytd_value != 0 else 0
                })
            
            return pd.DataFrame(performance_data)

def get_performance_metrics():
    """Get performance metrics (daily, weekly, monthly, YTD returns)"""
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT CURRENT_DATE")
            current_date = cursor.fetchone()[0]
            
            cursor.execute("SELECT SUM(current_value) FROM portfolio_holdings")
            current_value = cursor.fetchone()[0] or 0
            
            metrics = {
                'current': current_value,
                'daily': 0,
                'weekly': 0,
                'monthly': 0,
                'ytd': 0
            }
            
            def get_historical_value(target_date):
                query = """
                    WITH holdings_at_date AS (
                        SELECT 
                            code,
                            SUM(CASE 
                                WHEN transaction_type = 'invest' THEN units 
                                WHEN transaction_type = 'redeem' THEN -units 
                                WHEN transaction_type = 'switch_out' THEN -units 
                                WHEN transaction_type = 'switch_in' THEN units 
                                ELSE 0 
                            END) AS units_held
                        FROM portfolio_data
                        WHERE date <= %s
                        GROUP BY code
                        HAVING SUM(CASE 
                            WHEN transaction_type = 'invest' THEN units 
                            WHEN transaction_type = 'redeem' THEN -units 
                            WHEN transaction_type = 'switch_out' THEN -units 
                            WHEN transaction_type = 'switch_in' THEN units 
                            ELSE 0 
                        END) > 0
                    ),
                    nav_at_date AS (
                        SELECT 
                            code,
                            value as nav_value
                        FROM mutual_fund_nav
                        WHERE (code, nav) IN (
                            SELECT code, MAX(nav)
                            FROM mutual_fund_nav
                            WHERE nav <= %s
                            GROUP BY code
                        )
                    )
                    SELECT 
                        COALESCE(SUM(h.units_held * n.nav_value), 0)
                    FROM holdings_at_date h
                    JOIN nav_at_date n ON h.code = n.code
                """
                cursor.execute(query, (target_date, target_date))
                return cursor.fetchone()[0]
            
            if current_date.weekday() == 0:
                prev_day = current_date - BDay(1)
            else:
                prev_day = current_date - BDay(1)
            metrics['daily'] = get_historical_value(prev_day)
            
            week_ago = current_date - BDay(7)
            metrics['weekly'] = get_historical_value(week_ago)
            
            month_ago = current_date - BDay(30)
            metrics['monthly'] = get_historical_value(month_ago)
            
            ytd_date = datetime(current_date.year, 1, 1).date()
            metrics['ytd'] = get_historical_value(ytd_date)
            
            return metrics

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

def display_goal_dashboard():
    """Display the goal mapping dashboard"""
    st.subheader("Goal Dashboard")
    existing_goals = get_existing_goals()
    
    if not existing_goals.empty:
        goal_summary = existing_goals.groupby('goal_name').agg({
            'current_value': 'sum'
        }).reset_index()
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Goal-wise Summary")
            summary_cols = st.columns(len(goal_summary))
            for idx, goal in goal_summary.iterrows():
                with summary_cols[idx]:
                    st.metric(
                        goal['goal_name'],
                        format_indian_currency(goal['current_value'])
                    )
        
        with col2:
            st.subheader("Total Portfolio Value")
            st.metric(
                "Total",
                format_indian_currency(goal_summary['current_value'].sum())
            )
        
        st.subheader("Detailed Mappings")
        display_df = existing_goals.copy()
        display_df['current_value'] = display_df['current_value'].apply(format_indian_currency)
        display_df['Source'] = display_df['is_manual_entry'].map({True: 'Manual Entry', False: 'Portfolio'})
        display_df = display_df.drop('is_manual_entry', axis=1)
        st.dataframe(display_df)
    else:
        st.info("No goal mappings exist yet. Use the forms above to create your first mapping.")

def main():
    st.set_page_config(page_title="Mutual Fund Portfolio Manager", layout="wide")
    st.title("Mutual Fund Portfolio Manager")
    
    initialize_database()

    # Get all schemes and portfolio schemes first
    all_schemes = get_scheme_list()
    portfolio_schemes = get_portfolio_schemes()
    
    # Create dictionaries and name lists
    all_scheme_dict = {f"{s[1]} ({s[0]})": s[0] for s in all_schemes}
    portfolio_scheme_dict = {f"{s[1]} ({s[0]})": s[0] for s in portfolio_schemes}

    all_scheme_names = list(all_scheme_dict.keys())
    portfolio_scheme_names = list(portfolio_scheme_dict.keys())
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Dashboard", "Add Investment", "Add Redemption", "Add Switch", "Import Transactions",
        "Map Investments to Goals", "Manual Investments to Goals"
    ])
    
    with tab1:
        st.header("Portfolio Dashboard")
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
                'scheme_name', 'code', 'goal_name', 'current_units', 'latest_nav', 
                'current_value', 'total_investment', 'nav_date'
            ]]
            
            styled_df = display_df.style.format({
                'current_units': lambda x: format_indian(x),
                'latest_nav': lambda x: f"₹{format_indian(x)}",
                'current_value': lambda x: f"₹{format_indian(x)}",
                'total_investment': lambda x: f"₹{format_indian(x)}"
            })
            
            st.dataframe(styled_df, use_container_width=True)
            
            st.subheader("Fund Performance Analysis")
            fund_performance = get_fund_performance()
            
            if not fund_performance.empty:
                def color_negative_red(val):
                    if isinstance(val, str):
                        try:
                            val = float(val.strip('%'))
                        except:
                            return ''
                    color = 'red' if val < 0 else 'green'
                    return f'color: {color}'
                
                # Filter out rows with invalid or zero current_value
                fund_performance = fund_performance[
                    (fund_performance['current_value'] > 0) & 
                    (fund_performance['current_value'].notna())
                ]

                if not fund_performance.empty:
                    performance_display = fund_performance[[
                        'scheme_name', 'code', 'goal_name', 'current_value',
                        'daily_pct', 'weekly_pct','monthly_pct','ytd_pct'
                    ]].rename(columns={
                        'scheme_name': 'Scheme',
                        'code': 'Code',
                        'goal_name': 'Goal',
                        'current_value': 'Current Value',
                        'daily_pct': 'Daily',
                        'weekly_pct': 'Weekly',
                        'monthly_pct': 'Monthly',
                        'ytd_pct': 'YTD'
                    })

                    # Calculate dynamic height based on number of rows (max 600px)
                    num_rows = len(performance_display)
                    table_height = min(35 + num_rows * 35, 600)  # 35px per row + header
                    
                    st.dataframe(
                        performance_display.style
                        .applymap(color_negative_red, subset=['Daily', 'Weekly', 'Monthly', 'YTD'])
                        .format({
                            'Current Value': lambda x: f"₹{format_indian(x)}",
                            'Daily': lambda x: f"{x:.2f}%",
                            'Weekly': lambda x: f"{x:.2f}%",
                            'Monthly': lambda x: f"{x:.2f}%",
                            'YTD': lambda x: f"{x:.2f}%"
                        }),
                        column_order=["Scheme", "Code", "Goal", "Current Value", "Daily", "Weekly", "Monthly", "YTD"],
                        use_container_width=True,
                        height=table_height,
                        hide_index=True
                    )
            
            st.subheader("Recent Performance Insights")
            perf_metrics = get_performance_metrics()
            current_value = perf_metrics['current']
            
            cols = st.columns(4)
            cols[0].metric(
                "Daily Change", 
                f"₹{format_indian(current_value - perf_metrics['daily'])}", 
                f"{((current_value - perf_metrics['daily'])/perf_metrics['daily']*100) if perf_metrics['daily'] != 0 else 0:.2f}%"
            )
            cols[1].metric(
                "Weekly Change",
                f"₹{format_indian(current_value - perf_metrics['weekly'])}",
                f"{((current_value - perf_metrics['weekly'])/perf_metrics['weekly']*100) if perf_metrics['weekly'] != 0 else 0:.2f}%"
            )
            cols[2].metric(
                "Monthly Change",
                f"₹{format_indian(current_value - perf_metrics['monthly'])}",
                f"{((current_value - perf_metrics['monthly'])/perf_metrics['monthly']*100) if perf_metrics['monthly'] != 0 else 0:.2f}%"
            )
            cols[3].metric(
                "YTD Change",
                f"₹{format_indian(current_value - perf_metrics['ytd'])}",
                f"{((current_value - perf_metrics['ytd'])/perf_metrics['ytd']*100) if perf_metrics['ytd'] != 0 else 0:.2f}%"
            )
        
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

    with tab6:
        st.header("Map Investments to Goals")
        portfolio_df = get_portfolio_data_for_goals()
        
        if portfolio_df.empty:
            st.warning("No portfolio data found. Please ensure your portfolio data is up to date.")
        else:
            with st.form("goal_mapping_form"):
                goal_name = st.text_input("Goal Name")
                investment_type = st.selectbox("Investment Type", ["Equity", "Debt"])
                
                scheme_options = portfolio_df.apply(
                    lambda x: f"{x['scheme_name']} ({format_indian_currency(x['current_value'])})", 
                    axis=1
                ).tolist()
                selected_scheme = st.selectbox("Select Fund", scheme_options)
                
                if selected_scheme:
                    scheme_name = selected_scheme.split(" (₹")[0]
                    scheme_details = portfolio_df[portfolio_df['scheme_name'] == scheme_name].iloc[0]
                    current_value = scheme_details['current_value']
                    scheme_code = scheme_details['scheme_code']
                
                submitted = st.form_submit_button("Map to Goal")
                
                if submitted and goal_name and investment_type and selected_scheme:
                    existing_goal = check_existing_mapping(scheme_name, scheme_code)
                    if existing_goal:
                        st.error(f"This fund is already mapped to goal: {existing_goal}")
                    else:
                        insert_success = insert_goal_mapping(
                            goal_name, 
                            investment_type, 
                            scheme_name,
                            scheme_code,
                            current_value
                        )
                        if insert_success:
                            st.success(f"Successfully mapped {scheme_name} to goal: {goal_name}")
                        else:
                            st.error("Failed to map investment to goal. Please try again.")
        
        display_goal_dashboard()

    with tab7:
        st.header("Add Manual Investments to Goals")
        with st.form("manual_investment_form"):
            manual_goal_name = st.text_input("Goal Name")
            manual_scheme_type = st.selectbox(
                "Investment Type",
                ["PPF", "EPF", "NPS", "Fixed Deposit", "Other"]
            )
            manual_scheme_description = st.text_input(
                "Description",
                help="Enter additional details like FD duration, bank name etc."
            )
            manual_amount = st.number_input("Current Value", min_value=0.0, step=1000.0)
            
            manual_submitted = st.form_submit_button("Add Investment")
            
            if manual_submitted and manual_goal_name and manual_scheme_description and manual_amount > 0:
                insert_success = insert_goal_mapping(
                    manual_goal_name,
                    "Debt",
                    manual_scheme_type,
                    "9999",
                    manual_amount,
                    is_manual_entry=True
                )
                if insert_success:
                    st.success(f"Successfully added {manual_scheme_type} investment to goal: {manual_goal_name}")
                else:
                    st.error("Failed to add investment. Please try again.")
        
        display_goal_dashboard()

if __name__ == "__main__":
    main()