import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Page Settings
st.set_page_config(page_title="Data Analytics Portfolio", layout="wide")

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("🧭 Project Navigation")
st.sidebar.markdown("Switch between completed data assignments below:")
page = st.sidebar.radio("Go to:", ["1. Sales & Revenue Dashboard", "2. Customer Segmentation", "3. Predictive Forecasting"])

st.sidebar.write("---")
st.sidebar.info("💡 **Tip:** Adjust filters inside each project page to see metrics change live.")

# ==============================================================================
# PAGE 1: SALES & REVENUE DASHBOARD
# ==============================================================================
if page == "1. Sales & Revenue Dashboard":
    st.title("📊 Sales & Revenue Analysis Dashboard")
    st.markdown("Interactive business intelligence monitoring tool tracking overarching sales metrics.")
    st.write("---")
    
    try:
        df_sales = pd.read_csv("sales_data.csv")
        df_sales['Order_Date'] = pd.to_datetime(df_sales['Order_Date'])
        
        # Slicers/Filters
        st.sidebar.subheader("Dashboard Slicers")
        categories = ["All Categories"] + list(df_sales['Category'].unique())
        selected_cat = st.sidebar.selectbox("Select Product Category", categories)
        
        filtered_sales = df_sales if selected_cat == "All Categories" else df_sales[df_sales['Category'] == selected_cat]
        
        # KPIs
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("💰 Total Revenue", f"${filtered_sales['Revenue'].sum():,.2f}")
        kpi2.metric("📦 Total Units Sold", f"{filtered_sales['Quantity'].sum():,}")
        top_p = filtered_sales.groupby('Product')['Revenue'].sum().idxmax() if not filtered_sales.empty else "N/A"
        kpi3.metric("🏆 Top Product", top_p)
        
        st.write("---")
        
        # Visuals
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📈 Revenue Trend Over Time")
            trend_df = filtered_sales.groupby('Order_Date')['Revenue'].sum().reset_index()
            fig_line = px.line(trend_df, x='Order_Date', y='Revenue', markers=True)
            st.plotly_chart(fig_line, use_container_width=True)
        with col2:
            st.subheader("🥇 Top Performing Products")
            prod_df = filtered_sales.groupby('Product')['Revenue'].sum().reset_index().sort_values(by='Revenue', ascending=True)
            fig_bar = px.bar(prod_df, x='Revenue', y='Product', orientation='h')
            st.plotly_chart(fig_bar, use_container_width=True)
            
    except FileNotFoundError:
        st.error("Missing file: 'sales_data.csv' not found.")

# ==============================================================================
# PAGE 2: CUSTOMER SEGMENTATION
# ==============================================================================
elif page == "2. Customer Segmentation":
    st.title("👥 Customer Segmentation Model (K-Means)")
    st.markdown("Machine learning clustering algorithm partitioning customer profiles based on purchasing behaviors.")
    st.write("---")
    
    try:
        df_cust = pd.read_csv("customer_data.csv")
        
        # Clustering Math
        X = df_cust[['Annual_Income_k', 'Spending_Score']]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        df_cust['Cluster'] = kmeans.fit_predict(X_scaled)
        
        # Metrics & Summary Profiles
        st.subheader("📝 Automated Cluster Profiling")
        summary = df_cust.groupby('Cluster')[['Age', 'Annual_Income_k', 'Spending_Score']].mean()
        
        sc1, sc2, sc3 = st.columns(3)
        sc1.dataframe(summary.style.highlight_max(axis=0, color='#d4edda'))
        with sc2:
            st.markdown("**Cluster Descriptions:**\n* **Group 0:** Premium demographic / High income earners.\n* **Group 1:** Frugal shoppers / Minimal spending.\n* **Group 2:** Target core / Moderate income, high spending.")
            
        st.write("---")
        st.subheader("🎯 Machine Learning Segment Scatter Plot")
        
        # Plotly Interactivity for ML
        fig_scatter = px.scatter(
            df_cust, x='Annual_Income_k', y='Spending_Score', color=df_cust['Cluster'].astype(str),
            title="Income vs. Spending Score Segment Map",
            labels={'color': 'Customer Group', 'Annual_Income_k': 'Annual Income ($k)', 'Spending_Score': 'Spending Score (1-100)'},
            size_max=15
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    except FileNotFoundError:
        st.error("Missing file: 'customer_data.csv' not found.")

# ==============================================================================
# PAGE 3: PREDICTIVE FORECASTING
# ==============================================================================
elif page == "3. Predictive Forecasting":
    st.title("🔮 Predictive Analytics Revenue Modeling")
    st.markdown("Linear regression calculations plotting historical budgeting trends into future quarterly estimates.")
    st.write("---")
    
    try:
        df_hist = pd.read_csv("historical_data.csv")
        X = df_hist[['Marketing_Spend_k']]
        y = df_hist['Sales_Revenue_k']
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Forecasting calculations
        future_budgets = np.array([[60], [65], [70]])
        future_forecasts = model.predict(future_budgets)
        
        # Displays
        fcol1, fcol2 = st.columns(2)
        with fcol1:
            st.subheader("🎯 Future Multi-Scenario Projections")
            for b, f in zip(future_budgets, future_forecasts):
                st.info(f"If Marketing Invests **${b[0]}k** ➡️ Expected Sales Revenue: **${f:.2f}k**")
        with fcol2:
            st.subheader("📊 Model Accuracy Performance")
            st.metric("Model Baseline Connection (R-Squared Value)", f"{model.score(X, y)*100:.2f}%")
            
        st.write("---")
        st.subheader("📈 Projected Growth Trendpath Line")
        
        # Create full continuous line graph using plotly
        min_spend = float(X.min().iloc[0])
        full_x = np.linspace(min_spend, 75, 100).reshape(-1, 1)
        full_y = model.predict(full_x)
        df_line = pd.DataFrame({'Spend': full_x.flatten(), 'Revenue': full_y.flatten()})
        
        fig_trend = px.scatter(df_hist, x='Marketing_Spend_k', y='Sales_Revenue_k', labels={'Marketing_Spend_k': 'Spend ($k)', 'Sales_Revenue_k': 'Revenue ($k)'})
        fig_trend.add_traces(px.line(df_line, x='Spend', y='Revenue').data[0])
        fig_trend.data[1].line.color = 'orange'
        fig_trend.data[1].name = 'Trend Projection Line'
        
        st.plotly_chart(fig_trend, use_container_width=True)
        
    except FileNotFoundError:
        st.error("Missing file: 'historical_data.csv' not found.")