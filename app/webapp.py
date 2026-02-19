import streamlit as st
import pandas as pd
import plotly.express as px

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="Telecom Churn Dashboard", page_icon="📊", layout="wide")

@st.cache_data
def load_data():
    file_path = r"D:\Customer-Churn\Dataset\churn_dataset_cleaned.csv"
    return pd.read_csv(file_path)

df = load_data()

cat_cols = df.select_dtypes(include=['object']).columns.tolist()
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# ==========================================
# 2. SIDEBAR NAVIGATION
# ==========================================
st.sidebar.title("🧭 Navigation")
st.sidebar.markdown("Select a phase of the analysis to explore:")

analysis_phase = st.sidebar.radio(
    "Go to:",
    ["Dataset Overview", "Univariate Analysis", "Bivariate Analysis", "Multivariate Analysis"]
)
st.sidebar.markdown("---")

st.title("📊 Modern Telecom Churn Dashboard")

# ------------------------------------------
# PHASE 1: DATASET OVERVIEW
# ------------------------------------------
if analysis_phase == "Dataset Overview":
    st.header("1️⃣ Dataset Overview & Statistical Summary")
    
    st.subheader("At a Glance")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{len(df):,}")
    col2.metric("Total Features", f"{len(df.columns)}")
    if 'Churn' in df.columns:
        col3.metric("Overall Churn Rate", f"{(df['Churn'] == 'Yes').mean() * 100:.1f}%")
    col4.metric("Avg Monthly Bill", f"${df['MonthlyCharges'].mean():.2f}")
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["🔍 View Dataset", "📈 Numerical Statistics", "🔠 Categorical Statistics"])
    
    with tab1:
        st.dataframe(df, use_container_width=True)
    with tab2:
        st.dataframe(df.describe().T, use_container_width=True) 
    with tab3:
        st.dataframe(df.describe(include='object').T, use_container_width=True)

# ------------------------------------------
# PHASE 2: UNIVARIATE ANALYSIS
# ------------------------------------------
elif analysis_phase == "Univariate Analysis":
    st.header("2️⃣ Univariate Analysis")
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
            
    else: 
        with col1:
            fig_hist = px.histogram(df, x=selected_col, nbins=30, title=f"Distribution of {selected_col}", color_discrete_sequence=['#42A5F5'])
            st.plotly_chart(fig_hist, use_container_width=True)
        with col2:
            fig_box = px.box(df, y=selected_col, title=f"Spread & Outliers of {selected_col}", color_discrete_sequence=['#FFA726'])
            st.plotly_chart(fig_box, use_container_width=True)

    # --- DYNAMIC UNIVARIATE INSIGHTS ---
    if selected_col == 'Contract':
        st.warning("**💡 Stakeholder Insight:** Over half the customer base (55%) is on a Month-to-month contract. This represents a massive, highly volatile risk segment for the business.")
    elif selected_col == 'tenure':
        st.info("**💡 Stakeholder Insight:** Tenure is highly bimodal. We attract many new customers (0-5 months), but also have a solid base of fiercely loyal users (70+ months). The middle ground is hollow.")
    elif selected_col == 'MonthlyCharges':
        st.info("**💡 Stakeholder Insight:** A significant portion of the base subscribes to the absolute cheapest bare-bones plans (~$20/month).")
    else:
        st.caption(f"Showing univariate distribution for {selected_col}.")

# ------------------------------------------
# PHASE 3: BIVARIATE ANALYSIS
# ------------------------------------------
elif analysis_phase == "Bivariate Analysis":
    st.header("3️⃣ Bivariate Analysis")
    biv_type = st.radio("Select Relationship Type:", ["Categorical vs Categorical", "Categorical vs Numerical", "Numerical vs Numerical"], horizontal=True)
    st.markdown("---")
    
    if biv_type == "Categorical vs Categorical":
        col1, col2 = st.columns(2)
        cat_feat1 = col1.selectbox("X-axis (Feature):", cat_cols, index=cat_cols.index('Contract') if 'Contract' in cat_cols else 0)
        cat_feat2 = col2.selectbox("Grouping (Target):", cat_cols, index=cat_cols.index('Churn') if 'Churn' in cat_cols else 1)
        
        fig = px.histogram(df, x=cat_feat1, color=cat_feat2, barmode='group', text_auto=True, title=f"{cat_feat1} vs {cat_feat2}")
        st.plotly_chart(fig, use_container_width=True)

        # --- DYNAMIC CAT VS CAT INSIGHTS ---
        if cat_feat1 == 'Contract' and cat_feat2 == 'Churn':
            st.error("**🚨 The Contract Trap:** Month-to-month users have an alarming ~42.7% churn rate. **Recommendation:** Offer 'first month free' promotions to upgrade users to 1-Year or 2-Year contracts.")
        elif cat_feat1 == 'PaymentMethod' and cat_feat2 == 'Churn':
            st.warning("**⚠️ Billing Friction:** Electronic check users churn at over 45%. **Recommendation:** Nudge users toward automatic bank transfers with a $5 recurring discount.")
        elif cat_feat1 == 'InternetService' and cat_feat2 == 'Churn':
            st.warning("**⚠️ Infrastructure Risk:** Fiber Optic users have the highest churn rate among internet types. Investigate service outages or pricing misalignments immediately.")

    elif biv_type == "Categorical vs Numerical":
        col1, col2 = st.columns(2)
        cat_feat = col1.selectbox("Categorical Feature:", cat_cols)
        num_feat = col2.selectbox("Numerical Feature:", num_cols)
        
        fig = px.box(df, x=cat_feat, y=num_feat, color=cat_feat, title=f"{num_feat} grouped by {cat_feat}")
        st.plotly_chart(fig, use_container_width=True)

        # --- DYNAMIC CAT VS NUM INSIGHTS ---
        if cat_feat == 'Churn' and num_feat == 'tenure':
            st.success("**🎯 Retention Strategy:** The median tenure for churners is roughly 10 months. **Recommendation:** Implement a 'First-Year Loyalty Program' with milestone rewards to push them past the 12-month danger zone.")
        elif cat_feat == 'InternetService' and num_feat == 'MonthlyCharges':
            st.info("**💡 Pricing Insight:** Fiber Optic users pay the highest premium (median ~$90). Combined with their high churn rate, they are experiencing 'bill shock' without perceived value.")

    elif biv_type == "Numerical vs Numerical":
        col1, col2 = st.columns(2)
        num_feat1 = col1.selectbox("X-axis:", num_cols, index=0)
        num_feat2 = col2.selectbox("Y-axis:", num_cols, index=1 if len(num_cols)>1 else 0)
        
        color_target = 'Churn' if 'Churn' in cat_cols else None
        fig = px.scatter(df, x=num_feat1, y=num_feat2, color=color_target, opacity=0.6, title=f"{num_feat1} vs {num_feat2}")
        st.plotly_chart(fig, use_container_width=True)

        # --- DYNAMIC NUM VS NUM INSIGHTS ---
        if num_feat1 == 'tenure' and num_feat2 == 'MonthlyCharges':
            st.error("**🚨 High-Risk Profile:** Notice the concentration of Churn (red dots) in the top-left quadrant. Our highest flight risk is newly acquired customers (low tenure) on expensive plans (high charges).")

# ------------------------------------------
# PHASE 4: MULTIVARIATE ANALYSIS
# ------------------------------------------
elif analysis_phase == "Multivariate Analysis":
    st.header("4️⃣ Multivariate Analysis")
    
    corr_df = df.copy()
    if 'Churn' in corr_df.columns:
        corr_df['Churn_Num'] = corr_df['Churn'].map({'No': 0, 'Yes': 1})
        cols_to_correlate = num_cols + ['Churn_Num']
    else:
        cols_to_correlate = num_cols
        
    corr_matrix = corr_df[cols_to_correlate].corr()
    
    fig_corr = px.imshow(corr_matrix, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", origin="lower")
    st.plotly_chart(fig_corr, use_container_width=True)
    