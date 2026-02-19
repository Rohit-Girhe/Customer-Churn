import streamlit as st
import pandas as pd
import plotly.express as px

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="Telecom Churn Dashboard", page_icon="📊", layout="wide")

# ==========================================
# 2. DATA LOADING & PREPARATION
# ==========================================
@st.cache_data
def load_data():
    file_path = r"D:\Customer-Churn\Dataset\churn_dataset_cleaned.csv"
    return pd.read_csv(file_path)

df = load_data()

# Separate columns into categorical and numerical for dynamic dropdowns
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# ==========================================
# 3. SIDEBAR NAVIGATION
# ==========================================
st.sidebar.title("🧭 Navigation")
st.sidebar.markdown("Select a phase of the analysis to explore:")

analysis_phase = st.sidebar.radio(
    "Go to:",
    ["Dataset Overview", "Univariate Analysis", "Bivariate Analysis", "Multivariate Analysis"]
)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip:** Use the dynamic dropdowns in each section to explore different variables on the fly!")

# ==========================================
# 4. MAIN DASHBOARD LOGIC
# ==========================================
st.title("📊 Modern Telecom Churn Dashboard")

# ------------------------------------------
# PHASE 1: DATASET OVERVIEW
# ------------------------------------------
if analysis_phase == "Dataset Overview":
    st.header("1️⃣ Dataset Overview & Statistical Summary")
    st.markdown("Before diving into visual analytics, explore the raw data and its statistical properties.")
    
    # KPIs
    st.subheader("At a Glance")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{len(df):,}")
    col2.metric("Total Features", f"{len(df.columns)}")
    if 'Churn' in df.columns:
        col3.metric("Overall Churn Rate", f"{(df['Churn'] == 'Yes').mean() * 100:.1f}%")
    col4.metric("Avg Monthly Bill", f"${df['MonthlyCharges'].mean():.2f}")
    st.markdown("---")
    
    # Tabs for Data and Stats
    tab1, tab2, tab3 = st.tabs(["🔍 View Dataset", "📈 Numerical Statistics", "🔠 Categorical Statistics"])
    
    with tab1:
        st.markdown("### The Cleaned Dataset")
        st.dataframe(df, use_container_width=True)
    with tab2:
        st.markdown("### Numerical Summary")
        st.dataframe(df.describe().T, use_container_width=True) 
    with tab3:
        st.markdown("### Categorical Summary")
        st.dataframe(df.describe(include='object').T, use_container_width=True)

# ------------------------------------------
# PHASE 2: UNIVARIATE ANALYSIS
# ------------------------------------------
elif analysis_phase == "Univariate Analysis":
    st.header("2️⃣ Univariate Analysis")
    st.markdown("Explore the distribution of individual features across your customer base.")
    
    # Dynamic column selector
    selected_col = st.selectbox("Select a feature to analyze:", df.columns)
    
    col1, col2 = st.columns(2)
    
    if selected_col in cat_cols:
        with col1:
            fig_pie = px.pie(df, names=selected_col, title=f"Proportion of {selected_col}", hole=0.4, color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig_pie, use_container_width=True)
        with col2:
            fig_bar = px.histogram(df, x=selected_col, title=f"Count of {selected_col}", color=selected_col, text_auto=True)
            fig_bar.update_layout(showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)
            
    else: # If numerical
        with col1:
            fig_hist = px.histogram(df, x=selected_col, nbins=30, title=f"Distribution of {selected_col}", color_discrete_sequence=['#42A5F5'])
            st.plotly_chart(fig_hist, use_container_width=True)
        with col2:
            fig_box = px.box(df, y=selected_col, title=f"Spread & Outliers of {selected_col}", color_discrete_sequence=['#FFA726'])
            st.plotly_chart(fig_box, use_container_width=True)

# ------------------------------------------
# PHASE 3: BIVARIATE ANALYSIS
# ------------------------------------------
elif analysis_phase == "Bivariate Analysis":
    st.header("3️⃣ Bivariate Analysis")
    st.markdown("Analyze the relationship between two different variables.")
    
    # User selects the type of relationship
    biv_type = st.radio("Select Relationship Type to Explore:", 
                        ["Categorical vs Categorical", "Categorical vs Numerical", "Numerical vs Numerical"], horizontal=True)
    st.markdown("---")
    
    if biv_type == "Categorical vs Categorical":
        col1, col2 = st.columns(2)
        cat_feat1 = col1.selectbox("Select First Categorical Feature (X-axis):", cat_cols, index=cat_cols.index('Contract') if 'Contract' in cat_cols else 0)
        cat_feat2 = col2.selectbox("Select Second Categorical Feature (Grouping):", cat_cols, index=cat_cols.index('Churn') if 'Churn' in cat_cols else 1)
        
        fig = px.histogram(df, x=cat_feat1, color=cat_feat2, barmode='group', text_auto=True, 
                           title=f"Relationship between {cat_feat1} and {cat_feat2}")
        st.plotly_chart(fig, use_container_width=True)

    elif biv_type == "Categorical vs Numerical":
        col1, col2 = st.columns(2)
        cat_feat = col1.selectbox("Select Categorical Feature (X-axis):", cat_cols)
        num_feat = col2.selectbox("Select Numerical Feature (Y-axis):", num_cols)
        
        fig = px.box(df, x=cat_feat, y=num_feat, color=cat_feat, 
                     title=f"Distribution of {num_feat} grouped by {cat_feat}")
        st.plotly_chart(fig, use_container_width=True)

    elif biv_type == "Numerical vs Numerical":
        col1, col2 = st.columns(2)
        num_feat1 = col1.selectbox("Select First Numerical Feature (X-axis):", num_cols, index=0)
        num_feat2 = col2.selectbox("Select Second Numerical Feature (Y-axis):", num_cols, index=1 if len(num_cols)>1 else 0)
        
        # We will color it by Churn by default to make it insightful!
        color_target = 'Churn' if 'Churn' in cat_cols else None
        
        fig = px.scatter(df, x=num_feat1, y=num_feat2, color=color_target, opacity=0.6,
                         title=f"Correlation between {num_feat1} and {num_feat2}")
        st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------
# PHASE 4: MULTIVARIATE ANALYSIS
# ------------------------------------------
elif analysis_phase == "Multivariate Analysis":
    st.header("4️⃣ Multivariate Analysis")
    st.markdown("Discover complex mathematical correlations across multiple variables.")
    
    # We create a temporary dataframe just for correlation to include the target variable
    corr_df = df.copy()
    if 'Churn' in corr_df.columns:
        corr_df['Churn_Num'] = corr_df['Churn'].map({'No': 0, 'Yes': 1})
        cols_to_correlate = num_cols + ['Churn_Num']
    else:
        cols_to_correlate = num_cols
        
    corr_matrix = corr_df[cols_to_correlate].corr()
    
    # Plotly Heatmap
    fig_corr = px.imshow(corr_matrix, text_auto=".2f", aspect="auto", 
                         color_continuous_scale="RdBu_r", origin="lower",
                         title="Correlation Heatmap (Numerical Features + Target)")
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Business insight box
    st.info("""
    **💡 Business Insight:** Look for highly correlated variables (close to 1 or -1). 
    For example, high positive correlation between `tenure` and `TotalCharges` suggests Multicollinearity, which we must handle before building our Machine Learning model!
    """)