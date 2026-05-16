import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="SmartOps", layout="wide")

# Sidebar
st.sidebar.title("SmartOps")
st.sidebar.markdown("AI-Powered E-Commerce Operations Intelligence")
page = st.sidebar.selectbox("Navigate", [
    "Overview",
    "Demand Forecast",
    "Customer Segments",
    "Anomaly Detection",
    "AI Reports"
])

# Load data
@st.cache_data
def load_data():
    master   = pd.read_csv("../data/processed/master_data.csv")
    rfm      = pd.read_csv("../data/processed/rfm_segments.csv")
    forecast = pd.read_csv("../data/processed/demand_forecast.csv")
    anomalies= pd.read_csv("../data/processed/seller_anomalies.csv")
    return master, rfm, forecast, anomalies

master, rfm, forecast, anomalies = load_data()
master['order_purchase_timestamp'] = pd.to_datetime(master['order_purchase_timestamp'])

# ── PAGE 1: OVERVIEW ──────────────────────────────────
if page == "Overview":
    st.title("SmartOps — Operations Intelligence Dashboard")
    st.markdown("### Key Business Metrics")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Orders",    f"{master['order_id'].nunique():,}")
    col2.metric("Total Revenue",   f"R${master['payment_value'].sum():,.0f}")
    col3.metric("Delivery Rate",   f"{(master['order_status']=='delivered').mean()*100:.1f}%")
    col4.metric("Anomalies Found", f"{(anomalies['anomaly']==-1).sum():,}")

    st.markdown("### Monthly Order Volume")
    master['order_month'] = master['order_purchase_timestamp'].dt.to_period('M').astype(str)
    monthly = master.groupby('order_month')['order_id'].nunique().reset_index()
    monthly.columns = ['Month', 'Orders']
    fig = px.bar(monthly, x='Month', y='Orders', color='Orders',
                 color_continuous_scale='Blues')
    st.plotly_chart(fig, use_container_width=True)

# ── PAGE 2: DEMAND FORECAST ───────────────────────────
elif page == "Demand Forecast":
    st.title("Module 1 — Demand Forecasting Engine")
    forecast['ds'] = pd.to_datetime(forecast['ds'])

    fig = px.line(forecast, x='ds', y='yhat',
                  title='90-Day Demand Forecast',
                  labels={'ds':'Date','yhat':'Predicted Orders'})
    fig.add_scatter(x=forecast['ds'], y=forecast['yhat_upper'],
                    mode='lines', name='Upper Bound',
                    line=dict(dash='dash', color='lightblue'))
    fig.add_scatter(x=forecast['ds'], y=forecast['yhat_lower'],
                    mode='lines', name='Lower Bound',
                    line=dict(dash='dash', color='lightblue'))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Next 30 Days Forecast")
    next30 = forecast.tail(30)[['ds','yhat','yhat_lower','yhat_upper']]
    next30.columns = ['Date','Predicted Orders','Lower','Upper']
    st.dataframe(next30, use_container_width=True)

# ── PAGE 3: CUSTOMER SEGMENTS ─────────────────────────
elif page == "Customer Segments":
    st.title("Module 2 — Customer Behavior Analytics")

    col1, col2 = st.columns(2)
    with col1:
        seg_counts = rfm['Segment'].value_counts().reset_index()
        seg_counts.columns = ['Segment','Count']
        fig = px.pie(seg_counts, names='Segment', values='Count',
                     title='Customer Segments',
                     color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Segment Summary")
        summary = rfm.groupby('Segment').agg(
            Customers  = ('customer_id','count'),
            Avg_Recency= ('Recency','mean'),
            Avg_Monetary=('Monetary','mean')
        ).round(1).reset_index()
        st.dataframe(summary, use_container_width=True)

    st.markdown("### Churn Risk")
    churned = rfm['Churned'].sum()
    total   = len(rfm)
    st.metric("Churned Customers", f"{churned:,}", f"{churned/total*100:.1f}% of total")

# ── PAGE 4: ANOMALY DETECTION ─────────────────────────
elif page == "Anomaly Detection":
    st.title("Module 3 — Operations Anomaly Detector")

    fig = px.scatter(anomalies,
                     x='avg_delay', y='total_revenue',
                     color='anomaly_label',
                     color_discrete_map={'Normal':'steelblue','Anomaly':'red'},
                     title='Seller Anomaly Detection',
                     labels={'avg_delay':'Avg Delay (days)',
                             'total_revenue':'Total Revenue (BRL)'})
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Flagged Anomalous Sellers")
    flagged = anomalies[anomalies['anomaly']==-1][
        ['seller_id','avg_delay','total_orders','total_revenue']
    ].sort_values('total_revenue', ascending=False)
    st.dataframe(flagged, use_container_width=True)

# ── PAGE 5: AI REPORTS ────────────────────────────────
elif page == "AI Reports":
    st.title("Module 4 — AI Report Generator")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Weekly Ops Report")
        try:
            with open("../outputs/weekly_report.txt") as f:
                st.text_area("", f.read(), height=400)
        except:
            st.warning("Run Module 4 notebook first to generate report.")

    with col2:
        st.markdown("### Supplier Email")
        try:
            with open("../outputs/supplier_email.txt") as f:
                st.text_area("", f.read(), height=400)
        except:
            st.warning("Run Module 4 notebook first to generate email.")
        