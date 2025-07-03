import psycopg
import pandas as pd
from datetime import datetime

def upload_csv_to_postgresql(file_path, db_config, update_option):
    """
    Uploads data from a CSV file to a PostgreSQL database table with different update options.
    
    Args:
        file_path (str): The path to the CSV file containing the data to be uploaded.
        db_config (dict): A dictionary containing the database configuration.
        update_option (int): The update option selected by the user:
                            1 - Update Open Ended schemes only
                            2 - Update Closed Ended/Interval schemes only
                            3 - Update all schemes
                            4 - Refresh all data (delete and re-upload)
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

                # Check if the table exists
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'mutual_fund_master_data'
                    );
                """)
                table_exists = cursor.fetchone()[0]

                if not table_exists:
                    # Create the table if it doesn't exist
                    cursor.execute("""
                        CREATE TABLE mutual_fund_master_data (
                            id SERIAL PRIMARY KEY,
                            amc TEXT,
                            code TEXT UNIQUE,
                            scheme_type TEXT,
                            scheme_category TEXT,
                            scheme_name TEXT,
                            launch_date DATE,
                            closure_date DATE
                        );
                    """)
                    connection.commit()
                    print("Table 'mutual_fund_master_data' created.")

                # Load CSV data
                df = pd.read_csv(file_path)

                # Ensure date columns are properly parsed with specific format
                date_columns = ['launch_date', 'closure_date']
                date_format = '%Y-%m-%d'  # Adjust this format based on your actual date format
                
                for col in date_columns:
                    if col in df.columns:
                        try:
                            df[col] = pd.to_datetime(df[col], format=date_format, errors='coerce')
                        except ValueError:
                            df[col] = pd.to_datetime(df[col], errors='coerce')
                    else:
                        print(f"Warning: Column '{col}' not found in CSV file")

                # Replace NaT with None for proper insertion into the database
                df = df.where(pd.notnull(df), None)

                # Update closure_date to 9999-12-31 for blank or invalid entries
                if 'closure_date' in df.columns:
                    df['closure_date'] = df['closure_date'].apply(
                        lambda x: datetime.strptime('9999-12-31', '%Y-%m-%d').date() 
                        if x is None or pd.isna(x) else x
                    )

                if update_option == 4:  # Refresh all data
                    # Delete all existing records
                    cursor.execute("TRUNCATE TABLE mutual_fund_master_data RESTART IDENTITY")
                    connection.commit()
                    print("All existing records deleted.")

                    # Insert all data from CSV
                    inserted_count = 0
                    for _, row in df.iterrows():
                        try:
                            cursor.execute("""
                                INSERT INTO mutual_fund_master_data 
                                (amc, code, scheme_type, scheme_category, scheme_name, launch_date, closure_date)
                                VALUES (%s, %s, %s, %s, %s, %s, %s)
                            """, (
                                row['amc'], row['code'], row['scheme_type'],
                                row['scheme_category'], row['scheme_name'],
                                row['launch_date'] if 'launch_date' in df.columns else None,
                                row['closure_date'] if 'closure_date' in df.columns else datetime.strptime('9999-12-31', '%Y-%m-%d').date()
                            ))
                            inserted_count += 1
                        except psycopg.IntegrityError:
                            connection.rollback()
                            continue
                        except Exception as e:
                            print(f"Error inserting record {row['code']}: {e}")
                            connection.rollback()
                            continue
                    
                    connection.commit()
                    print(f"Successfully inserted {inserted_count} records.")
                    return

                # For options 1-3, get existing codes from database
                cursor.execute("SELECT code FROM mutual_fund_master_data")
                existing_codes = {row[0] for row in cursor.fetchall()}

                # Filter new schemes based on update option
                if update_option == 1:  # Open Ended only
                    new_schemes = df[(~df['code'].isin(existing_codes)) & 
                                    (df['scheme_type'].str.strip().str.lower() == 'open ended')]
                    print(f"Found {len(new_schemes)} new Open Ended schemes to add.")
                elif update_option == 2:  # Closed Ended/Interval only
                    new_schemes = df[(~df['code'].isin(existing_codes)) & 
                                    (df['scheme_type'].str.strip().str.lower().isin(['closed ended', 'close ended', 'interval fund']))]
                    print(f"Found {len(new_schemes)} new Closed Ended/Interval schemes to add.")
                elif update_option == 3:  # All new schemes
                    new_schemes = df[~df['code'].isin(existing_codes)]
                    print(f"Found {len(new_schemes)} new schemes to add (all types).")

                # Insert new schemes
                if not new_schemes.empty:
                    inserted_count = 0
                    for _, row in new_schemes.iterrows():
                        try:
                            cursor.execute("""
                                INSERT INTO mutual_fund_master_data 
                                (amc, code, scheme_type, scheme_category, scheme_name, launch_date, closure_date)
                                VALUES (%s, %s, %s, %s, %s, %s, %s)
                            """, (
                                row['amc'], row['code'], row['scheme_type'],
                                row['scheme_category'], row['scheme_name'],
                                row['launch_date'] if 'launch_date' in df.columns else None,
                                row['closure_date'] if 'closure_date' in df.columns else datetime.strptime('9999-12-31', '%Y-%m-%d').date()
                            ))
                            inserted_count += 1
                        except psycopg.IntegrityError:
                            connection.rollback()
                            continue
                        except Exception as e:
                            print(f"Error inserting record {row['code']}: {e}")
                            connection.rollback()
                            continue
                    
                    connection.commit()
                    print(f"Successfully added {inserted_count} new schemes.")
                else:
                    print("No new schemes to add based on the selected option.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    print("Mutual Fund Master Data Update Tool")
    print("----------------------------------")
    file_path = input("Enter the path to the CSV file: ")
    
    print("\nSelect update option:")
    print("1. Update master data - Open Ended schemes only")
    print("2. Update master data - Closed Ended/Interval schemes only")
    print("3. Update master data - All schemes")
    print("4. Refresh all data (delete existing and upload entire CSV)")
    
    while True:
        try:
            option = int(input("Enter your choice (1-4): "))
            if 1 <= option <= 4:
                break
            else:
                print("Please enter a number between 1 and 4.")
        except ValueError:
            print("Please enter a valid number.")

    DB_PARAMS = {
        'dbname': 'postgres',
        'user': 'postgres',
        'password': 'admin123',
        'host': 'localhost',
        'port': '5432'
    }

    upload_csv_to_postgresql(file_path, DB_PARAMS, option)