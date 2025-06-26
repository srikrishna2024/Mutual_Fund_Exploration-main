import streamlit as st
import psycopg
import pandas as pd
from datetime import datetime
import plotly.express as px

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
    
    if amount is None:
        return "₹0.00"
    return f"₹{format_number(float(amount))}"

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

def get_goals_mappings():
    """Get all goal mappings"""
    with get_db_connection() as conn:
        query = """
            SELECT 
                goal_name, 
                investment_type, 
                scheme_name, 
                scheme_code, 
                current_value,
                COALESCE(is_manual_entry, FALSE) as is_manual_entry,
                id
            FROM goals
            ORDER BY goal_name, scheme_name
        """
        return pd.read_sql(query, conn)

def get_unmapped_funds():
    """Get funds that are not mapped to any goal"""
    with get_db_connection() as conn:
        query = """
        WITH portfolio AS (
            SELECT 
                scheme_name,
                code as scheme_code,
                current_value
            FROM portfolio_holdings
        ),
        mapped_funds AS (
            SELECT 
                scheme_name,
                scheme_code
            FROM goals
            WHERE is_manual_entry = FALSE
        )
        SELECT 
            p.scheme_name,
            p.scheme_code,
            p.current_value
        FROM portfolio p
        LEFT JOIN mapped_funds m 
        ON p.scheme_name = m.scheme_name AND p.scheme_code = m.scheme_code
        WHERE m.scheme_name IS NULL
        ORDER BY p.current_value DESC
        """
        return pd.read_sql(query, conn)

def get_existing_funds():
    """
    Identify funds that have been partially redeemed or switched out 
    but are still mapped to goals
    """
    with get_db_connection() as conn:
        query = """
        WITH mapped_funds AS (
            SELECT 
                scheme_name,
                scheme_code,
                current_value as mapped_value,
                id
            FROM goals
            WHERE is_manual_entry = FALSE
        ),
        current_holdings AS (
            SELECT 
                scheme_name,
                code as scheme_code,
                current_value as actual_value
            FROM portfolio_holdings
        )
        SELECT 
            m.id,
            m.scheme_name,
            m.scheme_code,
            m.mapped_value,
            COALESCE(c.actual_value, 0) as actual_value
        FROM mapped_funds m
        LEFT JOIN current_holdings c 
        ON m.scheme_name = c.scheme_name AND m.scheme_code = c.scheme_code
        WHERE (COALESCE(c.actual_value, 0) != m.mapped_value) 
              OR c.actual_value IS NULL
        ORDER BY m.scheme_name
        """
        return pd.read_sql(query, conn)

def get_fully_redeemed_funds():
    """
    Identify funds that have been fully redeemed but are still mapped to goals
    """
    with get_db_connection() as conn:
        query = """
        WITH mapped_funds AS (
            SELECT 
                scheme_name,
                scheme_code,
                current_value,
                id,
                goal_name
            FROM goals
            WHERE is_manual_entry = FALSE
        ),
        current_holdings AS (
            SELECT 
                scheme_name,
                code as scheme_code
            FROM portfolio_holdings
        )
        SELECT 
            m.id,
            m.scheme_name,
            m.scheme_code,
            m.current_value,
            m.goal_name
        FROM mapped_funds m
        LEFT JOIN current_holdings c 
        ON m.scheme_name = c.scheme_name AND m.scheme_code = c.scheme_code
        WHERE c.scheme_name IS NULL
        ORDER BY m.scheme_name
        """
        return pd.read_sql(query, conn)

def get_new_funds():
    """
    Identify new funds added to portfolio that are not mapped to any goal
    """
    return get_unmapped_funds()

def update_goal_mapping(goal_id, new_value):
    """Update the current value of a goal mapping"""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE goals
                SET current_value = %s
                WHERE id = %s
            """, (new_value, goal_id))
            conn.commit()
            return True

def delete_goal_mapping(goal_id):
    """Delete a goal mapping"""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                DELETE FROM goals
                WHERE id = %s
            """, (goal_id,))
            conn.commit()
            return True

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
                
                inserted_id = cur.fetchone()[0]
                conn.commit()
                return True
    except Exception as e:
        print(f"Error inserting record: {str(e)}")
        return False

def create_unmapped_goal_category():
    """Create or check for the 'Unmapped Funds' goal category"""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            # Check if unmapped category exists
            cur.execute("""
                SELECT COUNT(*) 
                FROM goals 
                WHERE goal_name = 'Unmapped Funds'
            """)
            count = cur.fetchone()[0]
            if count == 0:
                # Create the unmapped goal category
                cur.execute("""
                    INSERT INTO goals 
                    (goal_name, investment_type, scheme_name, scheme_code, current_value, is_manual_entry)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, ('Unmapped Funds', 'Mixed', 'Unmapped Investments', '0000', 0.0, True))
                conn.commit()
            return True

def sync_goals():
    """
    Synchronize goals with current portfolio 
    Handle fully redeemed, partially redeemed, and new funds
    """
    # 1. Handle fully redeemed funds
    fully_redeemed = get_fully_redeemed_funds()
    if not fully_redeemed.empty:
        for _, fund in fully_redeemed.iterrows():
            # Delete the mapping
            delete_goal_mapping(fund['id'])
            st.info(f"Removed mapping for {fund['scheme_name']} which has been fully redeemed.")

    # 2. Handle partially redeemed funds
    existing_funds = get_existing_funds()
    if not existing_funds.empty:
        for _, fund in existing_funds.iterrows():
            if fund['actual_value'] == 0:
                # Fund no longer exists but was handled in the fully redeemed section
                continue
            
            # Update the current value to actual value
            update_goal_mapping(fund['id'], fund['actual_value'])
            st.info(f"Updated {fund['scheme_name']} value from {format_indian_currency(fund['mapped_value'])} to {format_indian_currency(fund['actual_value'])}")
    
    # 3. New funds are shown in the interface for the user to map

    return True

def display_goal_dashboard():
    """Display the goal mapping dashboard with visual data"""
    st.subheader("Goal Dashboard")
    existing_goals = get_goals_mappings()
    
    if not existing_goals.empty:
        # Group by goal and calculate total value
        goal_summary = existing_goals.groupby('goal_name').agg({
            'current_value': 'sum'
        }).reset_index()
        
        # Display goal summary
        col1, col2 = st.columns([2, 1])
        with col1:
            # Create pie chart for goal allocation
            fig = px.pie(
                goal_summary, 
                values='current_value', 
                names='goal_name',
                title='Goal Allocation',
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.subheader("Goal-wise Summary")
            for _, goal in goal_summary.iterrows():
                st.metric(
                    goal['goal_name'],
                    format_indian_currency(goal['current_value'])
                )
            
            st.subheader("Total Portfolio Value")
            st.metric(
                "Total",
                format_indian_currency(goal_summary['current_value'].sum())
            )
        
        # Display detailed mappings
        st.subheader("Detailed Mappings")
        # Format the dataframe for display
        display_df = existing_goals.copy()
        display_df['current_value'] = display_df['current_value'].apply(format_indian_currency)
        display_df['Source'] = display_df['is_manual_entry'].map({True: 'Manual Entry', False: 'Portfolio'})
        display_df = display_df.drop(['is_manual_entry', 'id'], axis=1)
        st.dataframe(display_df)
    else:
        st.info("No goal mappings exist yet. Use the forms above to create your first mapping.")

def main():
    st.set_page_config(page_title="Goal Sync Manager", layout="wide")
    st.title("Goal Sync Manager")
    
    # Create unmapped category if it doesn't exist
    create_unmapped_goal_category()
    
    tab1, tab2, tab3 = st.tabs([
        "Dashboard", "Sync Goals", "Map New Funds"
    ])
    
    with tab1:
        st.header("Goal Dashboard")
        if st.button("Refresh Data", key="refresh_dashboard"):
            st.rerun()
        
        display_goal_dashboard()
    
    with tab2:
        st.header("Sync Goals with Portfolio")
        st.info("This will update your goal mappings based on recent portfolio changes.")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Sync Now", key="sync_goals"):
                with st.spinner("Synchronizing goals..."):
                    if sync_goals():
                        st.success("Goals synchronized successfully!")
                    else:
                        st.error("Failed to synchronize goals. Please try again.")
        
        with col2:
            if st.button("Refresh After Sync", key="refresh_after_sync"):
                st.rerun()
        
        # Show existing funds with different values
        st.subheader("Existing Funds")
        existing_funds = get_existing_funds()
        
        if not existing_funds.empty:
            # Filter out fully redeemed ones which will be handled separately
            existing_funds = existing_funds[existing_funds['actual_value'] > 0]
            
            if not existing_funds.empty:
                # Format for display
                display_df = existing_funds.copy()
                display_df['mapped_value'] = display_df['mapped_value'].apply(format_indian_currency)
                display_df['actual_value'] = display_df['actual_value'].apply(format_indian_currency)
                display_df = display_df.drop('id', axis=1)
                
                st.dataframe(display_df, use_container_width=True)
            else:
                st.info("No existing funds found with value differences.")
        else:
            st.info("No existing funds need updating.")
        
        # Show fully redeemed funds
        st.subheader("Fully Redeemed Funds")
        fully_redeemed = get_fully_redeemed_funds()
        
        if not fully_redeemed.empty:
            # Format for display
            display_df = fully_redeemed.copy()
            display_df['current_value'] = display_df['current_value'].apply(format_indian_currency)
            display_df = display_df.drop('id', axis=1)
            
            st.dataframe(display_df, use_container_width=True)
        else:
            st.info("No fully redeemed funds found.")
    
    with tab3:
        st.header("Map New Funds to Goals")
        
        # Get all goals for mapping
        all_goals = get_goals_mappings()
        if all_goals.empty:
            st.warning("No goals exist. Please create goals first.")
        else:
            goal_names = all_goals['goal_name'].unique().tolist()
            
            # Get unmapped funds
            unmapped_funds = get_new_funds()
            
            if unmapped_funds.empty:
                st.info("No unmapped funds found in your portfolio.")
            else:
                st.subheader("Unmapped Funds")
                
                for _, fund in unmapped_funds.iterrows():
                    with st.expander(f"{fund['scheme_name']} - {format_indian_currency(fund['current_value'])}"):
                        with st.form(f"map_fund_{fund['scheme_code']}"):
                            selected_goal = st.selectbox(
                                "Select Goal", 
                                goal_names,
                                key=f"goal_for_{fund['scheme_code']}"
                            )
                            
                            investment_type = st.selectbox(
                                "Investment Type", 
                                ["Equity", "Debt", "Hybrid"],
                                key=f"type_for_{fund['scheme_code']}"
                            )
                            
                            submitted = st.form_submit_button("Map to Goal")
                            
                            if submitted:
                                if selected_goal == "Unmapped Funds":
                                    st.warning("Please select a valid goal.")
                                else:
                                    success = insert_goal_mapping(
                                        selected_goal,
                                        investment_type,
                                        fund['scheme_name'],
                                        fund['scheme_code'],
                                        fund['current_value'],
                                        False
                                    )
                                    
                                    if success:
                                        st.success(f"Successfully mapped {fund['scheme_name']} to {selected_goal}")
                                        st.rerun()
                                    else:
                                        st.error("Failed to map fund. Please try again.")

if __name__ == "__main__":
    main()