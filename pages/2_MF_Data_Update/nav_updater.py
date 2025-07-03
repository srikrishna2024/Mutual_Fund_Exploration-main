import os
import psycopg
import requests
from datetime import datetime
import time

LOG_FILE = "last_downloaded_scheme.log"

# Set date format explicitly for parsing NAV dates
def parse_date(date_str):
    """Parses a date string in DD-MM-YYYY format."""
    try:
        return datetime.strptime(date_str, "%d-%m-%Y").date()
    except ValueError:
        print(f"Invalid date format: {date_str}")
        return None

def create_nav_table_if_not_exists(cursor):
    """Creates the mutual_fund_nav table if it doesn't exist."""
    cursor.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_name = 'mutual_fund_nav'
        );
    """)
    table_exists = cursor.fetchone()[0]

    if not table_exists:
        cursor.execute("""
            CREATE TABLE mutual_fund_nav (
                id SERIAL PRIMARY KEY,
                code TEXT NOT NULL,
                scheme_name TEXT NOT NULL,
                nav DATE NOT NULL,
                value FLOAT NOT NULL
            );
        """)
        print("Table 'mutual_fund_nav' created.")

    # Ensure the unique constraint exists
    cursor.execute("""
        SELECT COUNT(*)
        FROM information_schema.table_constraints
        WHERE table_name = 'mutual_fund_nav' AND constraint_name = 'unique_code_nav';
    """)
    constraint_exists = cursor.fetchone()[0]

    if not constraint_exists:
        cursor.execute("""
            ALTER TABLE mutual_fund_nav
            ADD CONSTRAINT unique_code_nav UNIQUE (code, nav);
        """)
        print("Constraint 'unique_code_nav' added.")

def fetch_new_open_ended_schemes(cursor):
    """
    Fetches all open-ended schemes that exist in master_data but not in nav table.
    """
    cursor.execute("""
        SELECT mf.code, mf.scheme_name
        FROM mutual_fund_master_data mf
        WHERE mf.scheme_type = 'Open Ended'
        AND NOT EXISTS (
            SELECT 1
            FROM mutual_fund_nav nav
            WHERE nav.code = mf.code
        );
    """)
    schemes = cursor.fetchall()
    print(f"Fetched {len(schemes)} new open-ended schemes without NAV data.")
    return schemes

def fetch_all_open_ended_schemes(cursor):
    """Fetches all open-ended schemes."""
    cursor.execute("""
        SELECT code, scheme_name
        FROM mutual_fund_master_data
        WHERE scheme_type = 'Open Ended';
    """)
    schemes = cursor.fetchall()
    print(f"Fetched {len(schemes)} open-ended schemes.")
    return schemes

def fetch_nav_data(scheme_code, retries=3):
    """Fetches NAV data for a specific scheme using MFAPI with retry logic."""
    api_url = f"https://api.mfapi.in/mf/{scheme_code}"
    for attempt in range(retries):
        try:
            response = requests.get(api_url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if 'data' in data and data['data']:
                    return data
                else:
                    print(f"Empty data received for scheme {scheme_code}")
            else:
                print(f"Failed to fetch NAV data for scheme {scheme_code}: Status {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"Error fetching NAV data for scheme {scheme_code}: {e}")
        
        if attempt < retries - 1:
            print(f"Retrying... (Attempt {attempt + 1} of {retries})")
            time.sleep(1)  # Add a small delay between retries
    
    return None

def update_nav_data(cursor, connection, schemes, limit=None, offset=0):
    """Updates NAV data for the given list of schemes."""
    schemes_to_fetch = schemes[offset:offset+limit] if limit else schemes
    updated_count = 0
    inserted_count = 0
    skipped_count = 0
    last_successful_scheme = None

    for i, scheme in enumerate(schemes_to_fetch):
        scheme_code, scheme_name = scheme
        print(f"Processing scheme {i+1}/{len(schemes_to_fetch)}: {scheme_code} - {scheme_name}")
        
        nav_data = fetch_nav_data(scheme_code)
        if nav_data and 'data' in nav_data and nav_data['data']:
            entries_added = 0
            # Check if this scheme already has entries
            cursor.execute("SELECT COUNT(*) FROM mutual_fund_nav WHERE code = %s", (scheme_code,))
            existing_count = cursor.fetchone()[0]
            
            for nav_entry in nav_data['data']:
                nav_date = parse_date(nav_entry['date'])
                if not nav_date:
                    continue
                
                try:
                    nav_value = float(nav_entry['nav'])
                    cursor.execute("""
                        INSERT INTO mutual_fund_nav (code, scheme_name, nav, value)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT ON CONSTRAINT unique_code_nav DO NOTHING
                        RETURNING id;
                    """, (scheme_code, scheme_name, nav_date, nav_value))
                    
                    # Check if a row was inserted
                    result = cursor.fetchone()
                    if result:
                        entries_added += 1
                except Exception as e:
                    print(f"Error inserting NAV data for {scheme_code} on {nav_date}: {e}")
            
            # Commit after each scheme to save progress
            connection.commit()
            
            if entries_added > 0:
                print(f"Added {entries_added} NAV entries for scheme {scheme_code}")
                inserted_count += 1
            elif existing_count > 0:
                print(f"Scheme {scheme_code} already had data, no new entries added")
                skipped_count += 1
            else:
                print(f"No new NAV entries added for scheme {scheme_code}")
                skipped_count += 1
                
            updated_count += 1
            last_successful_scheme = scheme_code
            write_last_downloaded_scheme(last_successful_scheme)
        else:
            print(f"No valid NAV data found for scheme {scheme_code}.")
            skipped_count += 1
        
        # Add a small delay to avoid overwhelming the API
        time.sleep(0.2)
    
    print(f"Process summary:")
    print(f"Total schemes processed: {updated_count}")
    print(f"Schemes with new data inserted: {inserted_count}")
    print(f"Schemes skipped or with no new data: {skipped_count}")
    
    return last_successful_scheme

def read_last_downloaded_scheme():
    """Reads the last downloaded scheme code from the log file."""
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as file:
            return file.read().strip()
    return None

def write_last_downloaded_scheme(scheme_code):
    """Writes the last downloaded scheme code to the log file."""
    with open(LOG_FILE, "w") as file:
        file.write(str(scheme_code))

def verify_scheme_update(cursor, scheme_code):
    """Verifies if a scheme has entries in the NAV table."""
    cursor.execute("""
        SELECT COUNT(*)
        FROM mutual_fund_nav
        WHERE code = %s;
    """, (scheme_code,))
    count = cursor.fetchone()[0]
    return count > 0

def nav_updater(db_config):
    """
    Updates the Net Asset Value (NAV) data for mutual fund schemes in a PostgreSQL database.
    Args:
        db_config (dict): A dictionary containing the database configuration with keys:
            - 'dbname': The name of the database.
            - 'user': The username used to authenticate.
            - 'password': The password used to authenticate.
            - 'host': The host address of the database.
            - 'port': The port number on which the database is listening.
    
    The function now has an additional option to identify and update only new schemes
    that exist in the master data table but not in the NAV table.
    """
    try:
        # Connect to the PostgreSQL database
        with psycopg.connect(
            dbname=db_config['dbname'],
            user=db_config['user'],
            password=db_config['password'],
            host=db_config['host'],
            port=db_config['port']
        ) as connection:
            with connection.cursor() as cursor:
                # Ensure NAV table exists
                create_nav_table_if_not_exists(cursor)
                connection.commit()

                # Get user's choice
                print("Choose an option:")
                print("1. Update all open-ended schemes")
                print("2. Update 5000 schemes from last position")
                print("3. Update a specific scheme")
                print("4. Update only new schemes (in master data but not in NAV table)")
                print("5. Verify a specific scheme has NAV data")
                choice = input("Enter your choice (1/2/3/4/5): ")

                if choice == "1":
                    all_schemes = fetch_all_open_ended_schemes(cursor)
                    update_nav_data(cursor, connection, all_schemes)
                elif choice == "2":
                    all_schemes = fetch_all_open_ended_schemes(cursor)
                    # Determine the starting point
                    last_downloaded_scheme = read_last_downloaded_scheme()
                    if last_downloaded_scheme:
                        offset = next((i for i, scheme in enumerate(all_schemes) if scheme[0] == last_downloaded_scheme), 0) + 1
                    else:
                        offset = 0
                    
                    limit = 5000
                    print(f"Starting from position {offset} (after scheme {last_downloaded_scheme})")
                    last_scheme = update_nav_data(cursor, connection, all_schemes, limit=limit, offset=offset)
                    if last_scheme:
                        write_last_downloaded_scheme(last_scheme)
                elif choice == "3":
                    all_schemes = fetch_all_open_ended_schemes(cursor)
                    scheme_code = input("Enter the scheme code: ")
                    specific_scheme = [scheme for scheme in all_schemes if scheme[0] == scheme_code]
                    if specific_scheme:
                        update_nav_data(cursor, connection, specific_scheme)
                    else:
                        print(f"No scheme found with code {scheme_code}.")
                elif choice == "4":
                    # Option to update only schemes that don't have NAV data
                    new_schemes = fetch_new_open_ended_schemes(cursor)
                    if new_schemes:
                        print(f"Found {len(new_schemes)} new schemes without NAV data. Updating...")
                        update_nav_data(cursor, connection, new_schemes)
                    else:
                        print("No new schemes found that need NAV updates.")
                elif choice == "5":
                    # Verify a specific scheme
                    scheme_code = input("Enter the scheme code to verify: ")
                    has_data = verify_scheme_update(cursor, scheme_code)
                    if has_data:
                        print(f"Scheme {scheme_code} has NAV data in the database.")
                    else:
                        print(f"Scheme {scheme_code} does NOT have any NAV data in the database.")
                else:
                    print("Invalid choice. Exiting.")

                print("NAV update completed.")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    DB_PARAMS = {
        'dbname': 'postgres',
        'user': 'postgres',
        'password': 'admin123',
        'host': 'localhost',
        'port': '5432'
    }

    nav_updater(DB_PARAMS)