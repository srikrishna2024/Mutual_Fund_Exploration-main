import psycopg
import requests
from datetime import datetime, timedelta
import time
import streamlit as st
from typing import List, Tuple, Optional, Dict, Any
from queue import Queue, Empty
import threading
import os
import locale

# Set UTF-8 encoding globally
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

# Configuration
LOG_FILE = "nav_update_log.txt"
BATCH_SIZE = 50
API_DELAY = 0.5

# Initialize UTF-8 log file
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", encoding='utf-8-sig') as f:
        f.write("Mutual Fund NAV Update Log\n")

# Initialize Streamlit UI
st.set_page_config(page_title="Mutual Fund NAV Updater", layout="wide")
st.title("Mutual Fund NAV Updater")

# Database Configuration
DB_CONFIG = {
    'dbname': 'postgres',
    'user': 'postgres',
    'password': 'admin123',
    'host': 'localhost',
    'port': '5432'
}

# Thread-safe log queue
log_queue = Queue()
update_stats = {
    'total_schemes': 0,
    'success': 0,
    'failed': 0,
    'records_updated': 0
}

def is_weekday(date: datetime.date) -> bool:
    """Check if date is a weekday (Monday-Friday)"""
    return date.weekday() < 5  # 0=Monday, 4=Friday

def get_next_business_day(last_date: datetime.date) -> datetime.date:
    """Get the next business day (excluding weekends)"""
    next_day = last_date + timedelta(days=1)
    while not is_weekday(next_day):
        next_day += timedelta(days=1)
    return next_day

def write_log(message: str):
    """Thread-safe logging with UTF-8 support"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    
    with open(LOG_FILE, "a", encoding='utf-8') as file:
        file.write(log_entry + "\n")
    log_queue.put(log_entry)

def log_display_thread():
    """Continuously update the log display"""
    log_placeholder = st.empty()
    while True:
        try:
            logs = []
            while True:
                try:
                    logs.append(log_queue.get_nowait())
                except Empty:
                    break
            
            if logs:
                with log_placeholder.container():
                    st.subheader("Live Update Logs")
                    st.text_area("Log Output", "\n".join(logs[-100:]), 
                                height=300, 
                                key="live_logs")
            
            time.sleep(0.1)
        except:
            pass

if 'log_thread' not in st.session_state:
    st.session_state.log_thread = threading.Thread(target=log_display_thread, daemon=True)
    st.session_state.log_thread.start()

def reset_stats():
    global update_stats
    update_stats = {
        'total_schemes': 0,
        'success': 0,
        'failed': 0,
        'records_updated': 0
    }

def show_completion_message():
    """Display final summary"""
    with st.expander("✅ Update Summary", expanded=True):
        st.write(f"""
        - **Total schemes processed:** {update_stats['total_schemes']}
        - **Successfully updated:** {update_stats['success']}
        - **Failed updates:** {update_stats['failed']}
        - **Total NAV records added:** {update_stats['records_updated']}
        """)
    write_log(f"Update summary: {update_stats}")

def fetch_schemes_to_update(cursor, specific_code: Optional[str] = None) -> List[Tuple]:
    """Fetch schemes needing updates with their last NAV date"""
    query = """
        SELECT code, scheme_name, MAX(nav) as last_date
        FROM mutual_fund_nav
        {where_clause}
        GROUP BY code, scheme_name
    """.format(
        where_clause="WHERE code = %s" if specific_code else ""
    )
    
    params = (specific_code,) if specific_code else ()
    cursor.execute(query, params)
    return cursor.fetchall()

def fetch_portfolio_schemes(cursor) -> List[Tuple]:
    """Fetch portfolio schemes with their last NAV date"""
    query = """
        SELECT p.code, p.scheme_name, MAX(m.nav) as last_date
        FROM portfolio_data p
        LEFT JOIN mutual_fund_nav m ON p.code = m.code
        WHERE p.code IS NOT NULL
        GROUP BY p.code, p.scheme_name
    """
    cursor.execute(query)
    return cursor.fetchall()

def fetch_nav_data(scheme_code: str, retries: int = 3) -> Optional[Dict]:
    """Fetch NAV data with rate limiting"""
    api_url = f"https://api.mfapi.in/mf/{scheme_code}"
    
    for attempt in range(retries):
        try:
            time.sleep(API_DELAY)
            response = requests.get(api_url, timeout=10)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                write_log(f"❌ Scheme {scheme_code} not found in API")
                return None
        except requests.exceptions.RequestException as e:
            write_log(f"❌ Error fetching NAV data for {scheme_code}: {str(e)}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    
    write_log(f"❌ Failed to fetch NAV data for {scheme_code} after {retries} attempts")
    return None

def update_nav_data_batch(cursor, schemes: List[Tuple]) -> int:
    """Process schemes and only insert the next business day's NAV"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    update_stats['total_schemes'] = len(schemes)
    
    for i, (scheme_code, scheme_name, last_date) in enumerate(schemes):
        status_text.text(f"Processing: {scheme_name} ({scheme_code})")
        progress_bar.progress((i + 1) / len(schemes))
        
        try:
            nav_data = fetch_nav_data(scheme_code)
            if not nav_data or 'data' not in nav_data:
                update_stats['failed'] += 1
                continue
            
            # Find the next business day after last recorded date
            next_date = get_next_business_day(last_date) if last_date else None
            
            batch_values = []
            for nav_entry in nav_data['data']:
                try:
                    nav_date = datetime.strptime(nav_entry['date'], "%d-%m-%Y").date()
                    
                    # Only insert if it's the next business day or newer data is needed
                    if next_date and nav_date < next_date:
                        continue
                        
                    if not is_weekday(nav_date):
                        continue
                        
                    nav_value = float(nav_entry['nav'])
                    batch_values.append((scheme_code, scheme_name, nav_date, nav_value))
                    
                    # Stop after finding the first valid record (most recent)
                    break
                    
                except (ValueError, KeyError):
                    continue
            
            if batch_values:
                cursor.executemany("""
                    INSERT INTO mutual_fund_nav (code, scheme_name, nav, value)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT ON CONSTRAINT unique_code_nav DO NOTHING;
                """, batch_values)
                update_stats['records_updated'] += len(batch_values)
                update_stats['success'] += 1
                write_log(f"✔ Updated {scheme_name} with NAV for {batch_values[0][2]}")
            else:
                update_stats['failed'] += 1
                write_log(f"⚠ No new weekday data for {scheme_name}")
                
        except Exception as e:
            update_stats['failed'] += 1
            write_log(f"❌ Failed to process {scheme_name}: {str(e)}")
    
    return update_stats['records_updated']

# Main UI
with st.sidebar:
    st.header("Update Options")
    update_all = st.button("Update All Schemes")
    update_portfolio = st.button("Update Portfolio Schemes")
    specific_code = st.text_input("Update Specific Scheme (Enter Code)")
    update_specific = st.button("Update Specific Scheme")

log_placeholder = st.empty()
summary_placeholder = st.empty()

# Process updates
if update_all or update_portfolio or (update_specific and specific_code):
    reset_stats()
    try:
        with psycopg.connect(**DB_CONFIG) as connection:
            with connection.cursor() as cursor:
                schemes_to_update = []
                
                if update_all:
                    schemes_to_update = fetch_schemes_to_update(cursor)
                elif update_portfolio:
                    schemes_to_update = fetch_portfolio_schemes(cursor)
                elif update_specific and specific_code:
                    schemes_to_update = fetch_schemes_to_update(cursor, specific_code)
                
                if not schemes_to_update:
                    write_log("ℹ No eligible schemes found for update")
                    summary_placeholder.warning("No schemes needed updating")
                else:
                    write_log(f"🚀 Starting update for {len(schemes_to_update)} schemes")
                    total_updated = update_nav_data_batch(cursor, schemes_to_update)
                    connection.commit()
                    show_completion_message()
    
    except Exception as e:
        write_log(f"❌ Critical error: {str(e)}")
        summary_placeholder.error(f"Update failed: {str(e)}")