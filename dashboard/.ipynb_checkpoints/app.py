import streamlit as st
import pandas as pd
import plotly.express as px
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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
   master    = pd.read_csv(os.path.join(BASE_DIR, "data", "processed", "master_data.csv"))
   rfm       = pd.read_csv(os.path.join(BASE_DIR, "data", "processed", "rfm_segments.csv"))
   forecast  = pd.read_csv(os.path.join(BASE_DIR, "data", "processed", "demand_forecast.csv"))
   anomalies = pd.read_csv(os.path.join(BASE_DIR, "data", "processed", "seller_anomalies.csv"))
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
    st.title("Module 4 — AI Report & Email Generator")

    # Load report content safely
    try:
        with open(os.path.join(BASE_DIR, "outputs", "weekly_report.txt")) as f:
            summary_text = f.read()
    except:
        summary_text = ""

    try:
        with open(os.path.join(BASE_DIR, "outputs", "supplier_email.txt")) as f:
            supplier_text = f.read()
    except:
        supplier_text = ""

    # Display reports in UI layout columns
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Weekly Ops Report")
        if summary_text:
            st.text_area("", summary_text, height=400)
        else:
            st.warning("Run Module 4 notebook first.")

    with col2:
        st.markdown("### Supplier Email")
        if supplier_text:
            st.text_area("", supplier_text, height=400)
        else:
            st.warning("Run Module 4 notebook first.")

    # Email generation section
    st.markdown("---")
    st.markdown("### 📧 Send Report via Email")

    recipient_input = st.text_input(
        "Enter recipient email(s)",
        placeholder="e.g. manager@company.com, team@company.com",
        help="Separate multiple emails with a comma"
    )

    if st.button("📧 Send Weekly Report via Email"):
        if not recipient_input.strip():
            st.warning("Please enter at least one email address.")
        elif not summary_text:
            st.warning("No report found. Run Module 4 notebook first.")
        else:
            try:
                import smtplib
                from email.mime.text import MIMEText
                from email.mime.multipart import MIMEMultipart
                from dotenv import load_dotenv
                
                # Ensure local .env file variables are read into memory frame
                load_dotenv()

                # Sync configuration variables
                # 1. Your active dashboard link
                streamlit_dashboard_url = "https://your-app-name.streamlit.app"
                
                # 2. THE BULLETPROOF SLICE: Extract only the middle body content safely
                # This finds where "Dear" starts and where "Best regards," starts
                try:
                    start_idx = supplier_text.find("Dear")
                    end_idx = supplier_text.find("Best regards,")
                    
                    if start_idx != -1 and end_idx != -1:
                        # Slice out just the content text between those two markers
                        supplier_body_only = supplier_text[start_idx:end_idx].strip()
                    else:
                        # Fallback if the AI changed the greeting words completely
                        supplier_body_only = supplier_text
                except Exception:
                    supplier_body_only = supplier_text

                # 3. Construct your structured email output payload frame
                payload = f"""
======================================================================
                 OUTBOUND SUPPLIER REORDER DRAFT
======================================================================
{supplier_body_only}


======================================================================
                  INTERNAL EXECUTIVE SUMMARY 
======================================================================
{summary_text}

======================================================================
📊 LIVE SMARTOPS DASHBOARD CALL-TO-ACTION
======================================================================
To inspect real-time anomaly trends or update inventory parameters:
👉 CLICK HERE TO OPEN DASHBOARD: {streamlit_dashboard_url}
======================================================================

======================================================================
Best regards,
Vinay
E-commerce Operations Team
======================================================================
"""
                recipients = [r.strip() for r in recipient_input.split(",")]

                for recipient in recipients:
                    msg = MIMEMultipart()
                    msg['From']    = os.getenv("SENDER_EMAIL")
                    msg['To']      = recipient
                    msg['Subject'] = "SmartOps Weekly Operations Report"
                    msg.attach(MIMEText(payload, 'plain'))

                    with smtplib.SMTP("smtp-relay.brevo.com", 587) as server:
                        server.starttls()
                        server.login(
                            os.getenv("BREVO_LOGIN"),
                            os.getenv("BREVO_SMTP_KEY")
                        )
                        server.sendmail(
                            os.getenv("SENDER_EMAIL"),
                            recipient,
                            msg.as_string()
                        )

                st.success(f"✅ Report successfully sent to: {recipient_input}")

            except Exception as e:
                st.error(f"Failed to execute automated pipeline relay: {e}")