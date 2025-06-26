import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import psycopg
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

# -------------------- DATABASE CONFIG --------------------
DB_PARAMS = {
    'dbname': 'postgres',
    'user': 'postgres',
    'password': 'admin123',
    'host': 'localhost',
    'port': '5432'
}

def get_db_connection():
    return psycopg.connect(**DB_PARAMS)

# -------------------- HELPER FUNCTIONS --------------------
def calculate_rolling_returns(df_nav_pivot, window_days):
    """Calculate rolling returns for given window"""
    if len(df_nav_pivot) < window_days:
        return df_nav_pivot.pct_change(periods=min(window_days, len(df_nav_pivot)//2)).dropna()
    
    rolling_returns = (df_nav_pivot.pct_change(periods=window_days) + 1).pow(252/window_days) - 1
    return rolling_returns.dropna()

def calculate_consistency_score(returns_series):
    """Calculate consistency score - higher score means more consistent returns"""
    if len(returns_series) == 0 or returns_series.std() == 0:
        return 0
    
    mean_return = returns_series.mean()
    std_return = returns_series.std()
    sharpe_like = mean_return / std_return if std_return > 0 else 0
    return_bonus = max(0, mean_return) * 2 - abs(min(0, mean_return)) * 3
    return sharpe_like + return_bonus

def get_category_metrics(rolling_period_years=5):
    """Calculate median consistency and returns for all categories"""
    with get_db_connection() as conn:
        # Get all categories
        categories_df = pd.read_sql(
            "SELECT DISTINCT scheme_category FROM mutual_fund_master_data WHERE scheme_category IS NOT NULL",
            conn
        )
        categories = categories_df['scheme_category'].dropna().unique().tolist()
        
        category_metrics = []
        category_corr_data = {}
        
        for category in categories:
            # Get all funds in this category
            funds_query = """
                SELECT DISTINCT code
                FROM mutual_fund_master_data
                WHERE scheme_category = %s
            """
            df_funds = pd.read_sql(funds_query, conn, params=(category,))
            
            if df_funds.empty:
                continue
                
            fund_codes = df_funds['code'].tolist()
            
            # Get NAV data for these funds
            nav_query = """
                SELECT nav, code, value
                FROM mutual_fund_nav
                WHERE code = ANY(%s)
                ORDER BY nav ASC
            """
            df_nav = pd.read_sql(nav_query, conn, params=(fund_codes,))
            
            if df_nav.empty:
                continue
            
            # Process NAV data
            df_nav_pivot = df_nav.pivot(index='nav', columns='code', values='value')
            min_data_points = int(0.7 * len(df_nav_pivot))
            df_nav_pivot = df_nav_pivot.dropna(thresh=min_data_points, axis=1)
            df_nav_pivot = df_nav_pivot.fillna(method='ffill').fillna(method='bfill')
            
            if df_nav_pivot.empty:
                continue
                
            # Calculate rolling returns
            window = int(252 * rolling_period_years)
            if len(df_nav_pivot) < window:
                window = max(252, len(df_nav_pivot) // 2)
                
            rolling_returns = calculate_rolling_returns(df_nav_pivot, window)
            
            if rolling_returns.empty:
                continue
                
            # Calculate metrics for each fund
            fund_metrics = []
            for fund_code in rolling_returns.columns:
                fund_returns = rolling_returns[fund_code].dropna()
                if len(fund_returns) < 10:
                    continue
                    
                consistency = calculate_consistency_score(fund_returns)
                avg_return = fund_returns.mean()
                volatility = fund_returns.std()
                
                fund_metrics.append({
                    'code': fund_code,
                    'consistency': consistency,
                    'return': avg_return,
                    'volatility': volatility
                })
            
            if not fund_metrics:
                continue
                
            # Calculate median metrics for the category
            metrics_df = pd.DataFrame(fund_metrics)
            median_consistency = metrics_df['consistency'].median()
            median_return = metrics_df['return'].median()
            median_volatility = metrics_df['volatility'].median()
            
            category_metrics.append({
                'category': category,
                'median_consistency': median_consistency,
                'median_return': median_return,
                'median_volatility': median_volatility,
                'fund_count': len(fund_metrics)
            })
            
            # Store returns for correlation calculation
            if len(rolling_returns.columns) >= 3:  # Need at least 3 funds for representative category returns
                category_returns = rolling_returns.mean(axis=1)  # Average returns across all funds in category
                category_corr_data[category] = category_returns
                
        # Calculate correlation matrix between categories
        if category_corr_data:
            corr_df = pd.DataFrame(category_corr_data)
            corr_matrix = corr_df.corr()
            # Ensure matrix is perfectly symmetric
            corr_matrix = (corr_matrix + corr_matrix.T) / 2
        else:
            corr_matrix = pd.DataFrame()
            
        return pd.DataFrame(category_metrics), corr_matrix

def cluster_categories(corr_matrix):
    """Cluster categories based on correlation matrix"""
    if corr_matrix.empty:
        return {}
        
    try:
        # Convert correlation to distance matrix (0 = identical, 2 = opposite)
        distance_matrix = np.sqrt(2 * (1 - corr_matrix.abs()))
        np.fill_diagonal(distance_matrix.values, 0)
        
        # Ensure perfect symmetry
        if not np.allclose(distance_matrix, distance_matrix.T, atol=1e-8):
            distance_matrix = (distance_matrix + distance_matrix.T) / 2
        
        # Convert to condensed form
        dist_array = squareform(distance_matrix)
        
        # Perform hierarchical clustering
        linkage_matrix = linkage(dist_array, method='average')
        
        # Get cluster labels
        dendro = dendrogram(linkage_matrix, no_plot=True)
        clusters = {}
        for idx, cluster_id in enumerate(dendro['leaves']):
            category = corr_matrix.columns[cluster_id]
            clusters[category] = idx
            
        return clusters, distance_matrix
    
    except Exception as e:
        st.error(f"Clustering failed: {str(e)}")
        return {}, None

def identify_combinations(corr_matrix, category_metrics, n=3):
    """Identify low, medium, and high correlation combinations"""
    combinations = {
        'low_risk': [],
        'medium_risk': [],
        'high_risk': []
    }
    
    if corr_matrix.empty or len(corr_matrix) < 2:
        return combinations
        
    # Get all possible pairs
    categories = corr_matrix.columns.tolist()
    n_categories = len(categories)
    
    # Create DataFrame with category metrics
    metrics_df = category_metrics.set_index('category')
    
    # Evaluate all possible pairs
    for i in range(n_categories):
        for j in range(i+1, n_categories):
            cat1 = categories[i]
            cat2 = categories[j]
            
            corr = corr_matrix.loc[cat1, cat2]
            avg_volatility = (metrics_df.loc[cat1, 'median_volatility'] + metrics_df.loc[cat2, 'median_volatility']) / 2
            avg_return = (metrics_df.loc[cat1, 'median_return'] + metrics_df.loc[cat2, 'median_return']) / 2
            
            combination = {
                'categories': [cat1, cat2],
                'correlation': corr,
                'avg_volatility': avg_volatility,
                'avg_return': avg_return
            }
            
            if abs(corr) < 0.3:
                combinations['low_risk'].append(combination)
            elif abs(corr) < 0.7:
                combinations['medium_risk'].append(combination)
            else:
                combinations['high_risk'].append(combination)
    
    # Sort each risk category by best risk-return profile
    for risk_type in combinations:
        combinations[risk_type].sort(
            key=lambda x: x['avg_return'] / (x['avg_volatility'] + 1e-6), 
            reverse=True
        )
    
    return combinations

# -------------------- STREAMLIT APP --------------------
st.set_page_config(page_title="Category Analyzer", layout="wide")
st.title("📊 Mutual Fund Category Analysis")
st.markdown("Analyze all fund categories by consistency, returns, and correlations")

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")
rolling_period_years = st.sidebar.slider(
    "Rolling Period (Years)", 
    min_value=1, 
    max_value=10, 
    value=5, 
    step=1
)

analyze_button = st.sidebar.button("🔍 Analyze Categories", type="primary")

if analyze_button:
    with st.spinner("Analyzing all fund categories. This may take a few minutes..."):
        # Get category metrics and correlations
        category_metrics, corr_matrix = get_category_metrics(rolling_period_years)
        
        if category_metrics.empty:
            st.error("No category data available. Please check your database connection.")
            st.stop()
            
        # Display category metrics
        st.subheader("📋 Category Performance Metrics")
        
        # Format metrics for display
        display_metrics = category_metrics.copy()
        display_metrics['Median Return'] = display_metrics['median_return'].map('{:.2%}'.format)
        display_metrics['Median Volatility'] = display_metrics['median_volatility'].map('{:.2%}'.format)
        display_metrics['Median Consistency'] = display_metrics['median_consistency'].map('{:.2f}'.format)
        display_metrics['Fund Count'] = display_metrics['fund_count']
        
        st.dataframe(
            display_metrics[['category', 'Median Return', 'Median Volatility', 'Median Consistency', 'Fund Count']]
            .sort_values('Median Consistency', ascending=False),
            use_container_width=True,
            height=600
        )
        
        # Correlation analysis
        st.subheader("🔗 Category Correlation Analysis")
        
        if not corr_matrix.empty:
            # Plot correlation heatmap
            st.subheader("🎯 Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(16, 12))
            sns.heatmap(
                corr_matrix, 
                annot=True, 
                cmap="coolwarm", 
                fmt=".2f", 
                ax=ax, 
                square=True, 
                linewidths=0.5,
                vmin=-1, 
                vmax=1
            )
            ax.set_title(f"{rolling_period_years}-Year Rolling Return Correlations", fontsize=14)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            st.pyplot(fig)
            
            # Cluster analysis
            st.subheader("🌐 Category Clustering")
            clusters, distance_matrix = cluster_categories(corr_matrix)
            
            if clusters and distance_matrix is not None:
                # Create cluster visualization
                fig, ax = plt.subplots(figsize=(16, 8))
                dendrogram(
                    linkage(squareform(distance_matrix)), 
                    labels=corr_matrix.columns,
                    orientation='top',
                    leaf_rotation=90,
                    ax=ax
                )
                ax.set_title("Hierarchical Clustering of Categories")
                ax.set_ylabel("Correlation Distance")
                st.pyplot(fig)
                
                # Display clusters
                cluster_df = pd.DataFrame.from_dict(clusters, orient='index', columns=['Cluster'])
                st.write("Categories grouped by return pattern similarity:")
                st.dataframe(cluster_df.sort_values('Cluster'), use_container_width=True)
            else:
                st.warning("Could not perform hierarchical clustering on these categories")
            
            # Identify best combinations
            st.subheader("💡 Optimal Category Combinations")
            combinations = identify_combinations(corr_matrix, category_metrics)
            
            # Low risk combinations
            with st.expander("✅ Low Risk Combinations (Correlation < 0.3)", expanded=True):
                if combinations['low_risk']:
                    low_risk_df = pd.DataFrame(combinations['low_risk'][:10])  # Show top 10
                    low_risk_df['Correlation'] = low_risk_df['correlation'].map('{:.2f}'.format)
                    low_risk_df['Avg Return'] = low_risk_df['avg_return'].map('{:.2%}'.format)
                    low_risk_df['Avg Volatility'] = low_risk_df['avg_volatility'].map('{:.2%}'.format)
                    low_risk_df['Categories'] = low_risk_df['categories'].apply(lambda x: " + ".join(x))
                    
                    st.dataframe(
                        low_risk_df[['Categories', 'Correlation', 'Avg Return', 'Avg Volatility']],
                        use_container_width=True
                    )
                    
                    # Visualize best low-risk combination
                    best_low_risk = combinations['low_risk'][0]
                    cat1, cat2 = best_low_risk['categories']
                    
                    st.markdown(f"**Best Low-Risk Pair:** {cat1} + {cat2}")
                    st.markdown(f"- Correlation: {best_low_risk['correlation']:.2f}")
                    st.markdown(f"- Expected Return: {best_low_risk['avg_return']:.2%}")
                    st.markdown(f"- Expected Volatility: {best_low_risk['avg_volatility']:.2%}")
                else:
                    st.warning("No low-risk combinations found")
            
            # Medium risk combinations
            with st.expander("⚠️ Medium Risk Combinations (Correlation 0.3-0.7)"):
                if combinations['medium_risk']:
                    medium_risk_df = pd.DataFrame(combinations['medium_risk'][:10])
                    medium_risk_df['Correlation'] = medium_risk_df['correlation'].map('{:.2f}'.format)
                    medium_risk_df['Avg Return'] = medium_risk_df['avg_return'].map('{:.2%}'.format)
                    medium_risk_df['Avg Volatility'] = medium_risk_df['avg_volatility'].map('{:.2%}'.format)
                    medium_risk_df['Categories'] = medium_risk_df['categories'].apply(lambda x: " + ".join(x))
                    
                    st.dataframe(
                        medium_risk_df[['Categories', 'Correlation', 'Avg Return', 'Avg Volatility']],
                        use_container_width=True
                    )
                else:
                    st.warning("No medium-risk combinations found")
            
            # High risk combinations
            with st.expander("🔥 High Risk Combinations (Correlation > 0.7)"):
                if combinations['high_risk']:
                    high_risk_df = pd.DataFrame(combinations['high_risk'][:10])
                    high_risk_df['Correlation'] = high_risk_df['correlation'].map('{:.2f}'.format)
                    high_risk_df['Avg Return'] = high_risk_df['avg_return'].map('{:.2%}'.format)
                    high_risk_df['Avg Volatility'] = high_risk_df['avg_volatility'].map('{:.2%}'.format)
                    high_risk_df['Categories'] = high_risk_df['categories'].apply(lambda x: " + ".join(x))
                    
                    st.dataframe(
                        high_risk_df[['Categories', 'Correlation', 'Avg Return', 'Avg Volatility']],
                        use_container_width=True
                    )
                else:
                    st.warning("No high-risk combinations found")
            
            # Risk-return scatter plot with Plotly
            st.subheader("📈 Risk-Return Profile by Category")
            try:
                import plotly.express as px
                
                # Prepare data for Plotly
                plot_data = category_metrics.copy()
                plot_data['Volatility (%)'] = plot_data['median_volatility'] * 100
                plot_data['Return (%)'] = plot_data['median_return'] * 100
                plot_data['Consistency'] = plot_data['median_consistency']
                plot_data['Fund Count'] = plot_data['fund_count']
                
                fig = px.scatter(
                    plot_data,
                    x='Volatility (%)',
                    y='Return (%)',
                    color='Consistency',
                    color_continuous_scale='viridis',
                    size='Fund Count',
                    hover_name='category',
                    hover_data={
                        'Volatility (%)': ':.2f',
                        'Return (%)': ':.2f',
                        'Consistency': ':.2f',
                        'Fund Count': True,
                        'category': False
                    },
                    labels={
                        'Volatility (%)': 'Volatility (%)',
                        'Return (%)': 'Return (%)',
                        'Consistency': 'Consistency Score',
                        'Fund Count': 'Number of Funds'
                    },
                    title="Category Risk-Return Characteristics (Hover for details)"
                )
                
                # Customize layout
                fig.update_layout(
                    hovermode='closest',
                    xaxis_title="Volatility (%)",
                    yaxis_title="Return (%)",
                    coloraxis_colorbar_title="Consistency",
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            except ImportError:
                st.warning("Plotly not available, showing static plot instead")
                # Fallback to Matplotlib if Plotly not available
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # Color by median consistency (no labels)
                scatter = ax.scatter(
                    x=category_metrics['median_volatility'] * 100,
                    y=category_metrics['median_return'] * 100,
                    c=category_metrics['median_consistency'],
                    cmap='viridis',
                    s=100,
                    alpha=0.7
                )
                
                ax.set_xlabel("Volatility (%)")
                ax.set_ylabel("Return (%)")
                ax.set_title("Category Risk-Return Characteristics")
                ax.grid(True, alpha=0.3)
                
                # Add colorbar
                cbar = plt.colorbar(scatter)
                cbar.set_label("Consistency Score")
                
                st.pyplot(fig)
            
        else:
            st.warning("Insufficient data to calculate category correlations")

# -------------------- INSTRUCTIONS --------------------
st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 How to Use This Tool")
with st.sidebar.expander("Click for instructions"):
    st.markdown("""
    1. **Select rolling period** (1-10 years) in the sidebar
    2. Click **"Analyze Categories"** to process all fund data
    3. View **category performance metrics** table
    4. Examine **correlation heatmap** to understand relationships
    5. Check **optimal combinations** for different risk levels
    6. Use **interactive risk-return plot** (hover for details)
    
    **Key Features:**
    - Median consistency, return, and volatility for each category
    - Correlation analysis between all category pairs
    - Hierarchical clustering of similar categories
    - Recommended low, medium, and high-risk combinations
    - Interactive risk-return profile visualization
    """)

st.sidebar.markdown("### 🎯 Key Metrics")
st.sidebar.info("""
- **Consistency Score**: Higher = more stable returns
- **Correlation**: 
  - <0.3 = Excellent diversification
  - 0.3-0.7 = Moderate diversification
  - >0.7 = Limited diversification
- **Risk-Return**: Higher returns typically come with higher volatility
""")