import streamlit as st
import pandas as pd
import plotly.express as px

# Set up page configuration
st.set_page_config(page_title="Sales & Revenue Analysis Dashboard", layout="wide")

# App Header
st.title("📊 Sales & Revenue Analysis Dashboard")
st.markdown("Import data, track business KPIs, and explore interactive revenue trends.")
st.write("---")

# 1. FEATURE: Import Data from CSV
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("sales_data.csv")
        df['Order_Date'] = pd.to_datetime(df['Order_Date'])
        return df
    except FileNotFoundError:
        st.error("Error: 'sales_data.csv' not found. Please create the file first!")
        return None

df = load_data()

if df is not None:
    # 2. FEATURE: Interactive Slicers and Filters (Sidebar)
    st.sidebar.header("🎯 Dashboard Filters")
    
    # Filter by Product Category
    categories = ["All Categories"] + list(df['Category'].unique())
    selected_category = st.sidebar.selectbox("Select Product Category", categories)
    
    # Apply Slicer Filter to DataFrame
    if selected_category != "All Categories":
        filtered_df = df[df['Category'] == selected_category]
    else:
        filtered_df = df

    # 3. FEATURE: Visualize Key KPIs (Top Row)
    total_revenue = filtered_df['Revenue'].sum()
    total_units = filtered_df['Quantity'].sum()
    
    # Find top performing product safely based on filter
    if not filtered_df.empty:
        top_product = filtered_df.groupby('Product')['Revenue'].sum().idxmax()
    else:
        top_product = "N/A"

    # Layout KPIs into 3 columns
    kpi1, kpi2, kpi3 = st.columns(3)
    with kpi1:
        st.metric(label="💰 Total Revenue", value=f"${total_revenue:,.2f}")
    with kpi2:
        st.metric(label="📦 Total Units Sold", value=f"{total_units:,}")
    with kpi3:
        st.metric(label="🏆 Top Product", value=top_product)

    st.write("---")

    # 4. FEATURE: Interactive Charts & Trend Lines
    chart1, chart2 = st.columns(2)

    with chart1:
        st.subheader("📈 Revenue Trend Over Time")
        # Aggregate revenue by date for smooth line plotting
        trend_df = filtered_df.groupby('Order_Date')['Revenue'].sum().reset_index()
        fig_line = px.line(
            trend_df, 
            x='Order_Date', 
            y='Revenue', 
            markers=True,
            labels={'Order_Date': 'Date', 'Revenue': 'Revenue ($)'}
        )
        st.plotly_chart(fig_line, use_container_width=True)

    with chart2:
        st.subheader("🥇 Top Performing Products")
        # Aggregate total revenue per unique product name
        product_df = filtered_df.groupby('Product')['Revenue'].sum().reset_index().sort_values(by='Revenue', ascending=True)
        fig_bar = px.bar(
            product_df, 
            x='Revenue', 
            y='Product', 
            orientation='h',
            labels={'Revenue': 'Total Sales ($)', 'Product': 'Product'}
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Display Raw Data View at the bottom
    with st.expander("🔍 View Filtered Raw Data"):
        st.dataframe(filtered_df, use_container_width=True)