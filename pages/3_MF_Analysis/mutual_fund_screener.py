import streamlit as st
import pandas as pd
import psycopg

# Database connection parameters
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

def get_categories():
    """Fetch unique scheme categories"""
    with connect_to_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT DISTINCT scheme_category 
                FROM mutual_fund_master_data 
                ORDER BY scheme_category;
            """)
            return [row[0] for row in cur.fetchall()]

def get_fund_metrics(category):
    """
    Fetch and calculate fund metrics for the selected category
    """
    with connect_to_db() as conn:
        query = """
        WITH latest_dates AS (
            SELECT code, MAX(nav) as latest_date
            FROM mutual_fund_nav
            WHERE value::float > 0
            GROUP BY code
        ),
        peak_metrics AS (
            SELECT 
                code,
                MAX(value::float) as peak_value,
                (
                    SELECT nav
                    FROM mutual_fund_nav n2
                    WHERE n2.code = n1.code
                    AND n2.value::float = MAX(n1.value::float)
                    ORDER BY nav DESC
                    LIMIT 1
                ) as peak_date
            FROM mutual_fund_nav n1
            WHERE value::float > 0
            GROUP BY code
        )
        SELECT 
            m.scheme_name,
            ld.latest_date as current_date,
            n1.value::float as current_nav,
            pm.peak_date,
            pm.peak_value as peak_nav,
            CASE 
                WHEN pm.peak_value > 0 THEN 
                    ((n1.value::float - pm.peak_value) / pm.peak_value * 100)
                ELSE NULL 
            END as peak_change
        FROM mutual_fund_master_data m
        JOIN latest_dates ld ON m.code = ld.code
        JOIN mutual_fund_nav n1 ON m.code = n1.code AND n1.nav = ld.latest_date
        JOIN peak_metrics pm ON m.code = pm.code
        WHERE m.scheme_category = %s
        ORDER BY m.scheme_name;
        """
        
        df = pd.read_sql(query, conn, params=(category,))
        return df

def main():
    """
    Main function to run the Mutual Fund Category Screener application.
    This function sets up the Streamlit page configuration, displays the title,
    and provides a sidebar for user inputs. It allows users to select a mutual
    fund category, choose sorting options, and analyze the selected category.
    The results are displayed in a table format, and users can download the
    results as a CSV file. Additionally, summary statistics for the selected
    category are displayed.
    Sidebar Inputs:
        - Select Mutual Fund Category: Dropdown to select the mutual fund category.
        - Sort By: Dropdown to select the column to sort by.
        - Sort Order: Dropdown to select the sort order (Ascending/Descending).
        - Analyze Category: Button to trigger the analysis.
    Displays:
        - Screening Results: A table of the mutual funds in the selected category,
          sorted based on the selected criteria.
        - Download Results as CSV: Button to download the screening results as a CSV file.
        - Category Summary: A table displaying summary statistics for the selected category.
    Raises:
        - Displays an error message if no data is found for the selected category.
    """
    st.set_page_config(page_title='Mutual Fund Category Screener', layout='wide')
    st.title('Mutual Fund Category Screener')

    # Sidebar for inputs
    st.sidebar.header('Select Parameters')
    
    # Get categories
    categories = get_categories()
    selected_category = st.sidebar.selectbox('Select Mutual Fund Category', categories)
    
    # Add sorting options
    sort_options = {
        'Scheme Name': 'scheme_name',
        'Current NAV': 'current_nav',
        'Peak NAV': 'peak_nav',
        'Change from Peak': 'peak_change'
    }
    sort_by = st.sidebar.selectbox('Sort By', list(sort_options.keys()))
    sort_order = st.sidebar.selectbox('Sort Order', ['Ascending', 'Descending'])

    analyze_button = st.sidebar.button('Analyze Category')

    if analyze_button:
        with st.spinner('Analyzing funds...'):
            # Get fund metrics
            df = get_fund_metrics(selected_category)
            
            if df.empty:
                st.error('No data found for the selected category.')
                return
            
            # Ensure the columns are of type float
            df['current_nav'] = df['current_nav'].astype(float)
            df['peak_nav'] = df['peak_nav'].astype(float)
            
            # Sort the dataframe first using numerical values
            is_ascending = sort_order == 'Ascending'
            sort_column = sort_options[sort_by]
            df = df.sort_values(
                by=sort_column,
                ascending=is_ascending,
                na_position='last'
            )
            
            # Format the dates and numeric values for display after sorting
            display_df = pd.DataFrame({
                'Scheme Name': df['scheme_name'],
                'Latest NAV Date': pd.to_datetime(df['current_date']).dt.strftime('%Y-%m-%d'),
                'Current NAV': df['current_nav'].apply(lambda x: f"{round(float(x), 1)}"),  # Round to 1 decimal place and format as string
                'Peak NAV Date': pd.to_datetime(df['peak_date']).dt.strftime('%Y-%m-%d'),
                'Peak NAV': df['peak_nav'].apply(lambda x: f"{round(float(x), 1)}"),  # Round to 1 decimal place and format as string
                'Change from Peak (%)': df['peak_change'].apply(
                    lambda x: f"{x:.2f}%" if pd.notnull(x) else "N/A"
                )
            })
            
            # Display results as a static table (disables header sorting)
            st.subheader(f'Screening Results for {selected_category}')
            st.table(display_df)  # Use st.table instead of st.dataframe
            
            # Add download button
            csv = display_df.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name=f"fund_screener_{selected_category}.csv",
                mime="text/csv"
            )
            
            # Display summary statistics
            st.subheader('Category Summary')
            summary_df = pd.DataFrame({
                'Metric': [
                    'Average Change from Peak (%)',
                    'Funds at All-Time High',
                    'Total Funds Analyzed'
                ],
                'Value': [
                    f"{df['peak_change'].mean():.2f}%",
                    f"{(df['peak_change'] >= -0.01).sum()}",
                    f"{len(df)}"
                ]
            })
            st.table(summary_df)  # Use st.table for summary as well

if __name__ == "__main__":
    main()