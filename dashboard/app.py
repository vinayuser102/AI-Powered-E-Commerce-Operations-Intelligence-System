import os
import re
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from dotenv import load_dotenv

# Works both locally (dashboard/ subfolder) and on Streamlit Cloud (repo root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "Data", "Processed")
API_BASE_URL = os.getenv("SMARTOPS_API_URL", "http://127.0.0.1:8000").rstrip("/")
load_dotenv(os.path.join(BASE_DIR, ".env"))


def get_secret(key):
    return st.secrets.get(key, os.getenv(key, ""))

st.set_page_config(page_title="SmartOps", layout="wide")

# Sidebar
st.sidebar.title("SmartOps")
st.sidebar.markdown("AI-Powered E-Commerce Operations Intelligence")
page = st.sidebar.selectbox("Navigate", [
    "Overview",
    "Demand Forecast",
    "Customer Segments",
    "Anomaly Detection",
    "AI Reports",
    "🤖 AI Assistant"
])

# Load data
@st.cache_data
def load_data():
    required_files = ["master_data.csv", "rfm_segments.csv", "demand_forecast.csv", "seller_anomalies.csv"]
    missing_files = [name for name in required_files if not os.path.isfile(os.path.join(DATA_DIR, name))]
    if missing_files:
        raise FileNotFoundError(
            f"Missing processed data files in {DATA_DIR}: {', '.join(missing_files)}. "
            "Run the notebooks in order to generate them."
        )
    master    = pd.read_csv(os.path.join(DATA_DIR, "master_data.csv"))
    rfm       = pd.read_csv(os.path.join(DATA_DIR, "rfm_segments.csv"))
    forecast  = pd.read_csv(os.path.join(DATA_DIR, "demand_forecast.csv"))
    anomalies = pd.read_csv(os.path.join(DATA_DIR, "seller_anomalies.csv"))
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
                    line={'dash': 'dash', 'color': 'lightblue'})
    fig.add_scatter(x=forecast['ds'], y=forecast['yhat_lower'],
                    mode='lines', name='Lower Bound',
                    line={'dash': 'dash', 'color': 'lightblue'})
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
            Customers   = ('customer_id','count'),
            Avg_Recency = ('Recency','mean'),
            Avg_Monetary= ('Monetary','mean')
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

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Weekly Ops Report")
        try:
            with open(os.path.join(BASE_DIR, "outputs", "weekly_report.txt")) as f:
                report_content = f.read()
            st.text_area("", report_content, height=400)
        except FileNotFoundError:
            st.warning("Run Module 4 notebook first.")
            report_content = ""

    with col2:
        st.markdown("### Supplier Email")
        try:
            with open(os.path.join(BASE_DIR, "outputs", "supplier_email.txt")) as f:
                supplier_content = f.read()
            st.text_area("", supplier_content, height=400)
        except FileNotFoundError:
            st.warning("Run Module 4 notebook first.")
            supplier_content = ""

    # Email section
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
        else:
            try:
                smtp_server     = "smtp-relay.brevo.com"
                port            = 587
                login_email     = get_secret("BREVO_LOGIN")
                sender_email    = get_secret("SENDER_EMAIL")
                sender_password = get_secret("BREVO_SMTP_KEY")
                dashboard_url   = get_secret("DASHBOARD_URL")

                if not all([login_email, sender_email, sender_password]):
                    raise ValueError("Email is not configured. Set BREVO_LOGIN, SENDER_EMAIL, and BREVO_SMTP_KEY.")
                recipients = [recipient.strip() for recipient in recipient_input.split(",") if recipient.strip()]
                if any("@" not in recipient for recipient in recipients):
                    raise ValueError("Enter valid comma-separated email addresses.")

                old_closing = "Best regards,\n[Your Name]\nE-commerce Operations Team"
                closing     = "Best regards,\nVinay Sharma\nE-commerce Operations Team"
                supplier_body = supplier_content.replace(old_closing, "").rstrip()

                dashboard_link = (
                    f"\n\n🔗 View Live Dashboard: {dashboard_url}"
                    if dashboard_url else ""
                )

                full_body = (
                    "✉️ PART 1: OUTBOUND SUPPLIER REORDER DRAFT (Procurement Action)\n"
                    + "=" * 60 + "\n\n"
                    + supplier_body
                    + "\n\n"
                    + "=" * 60 + "\n"
                    + "📊 PART 2: INTERNAL EXECUTIVE SUMMARY (Business Health Analytics)\n"
                    + "=" * 60 + "\n\n"
                    + report_content
                    + dashboard_link
                    + "\n\n"
                    + closing
                )

                for recipient in recipients:
                    msg = MIMEMultipart()
                    msg['From']    = sender_email
                    msg['To']      = recipient
                    msg['Subject'] = "SmartOps Weekly Operations Report"
                    msg.attach(MIMEText(full_body, 'plain'))

                    with smtplib.SMTP(smtp_server, port) as server:
                        server.starttls()
                        server.login(login_email, sender_password)
                        server.sendmail(sender_email, recipient, msg.as_string())

                st.success(f"✅ Report sent to: {recipient_input}")

            except (OSError, ValueError, smtplib.SMTPException) as e:
                st.error(f"Failed to send: {e}")
# -----------------------------------------------------------------------------
# PAGE 6: AI ASSISTANT
# -----------------------------------------------------------------------------
elif page == "🤖 AI Assistant":
    st.header("🤖 SmartOps Autonomous AI Assistant")
    st.caption("Ask questions about operations policies (RAG) or calculate customer churn risk in real-time (FastAPI + XGBoost).")

    # 1. Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello! I am your SmartOps Ops Assistant. Ask me about company shipping rules, return policies, or provide customer metrics to evaluate churn risk!"
            }
        ]

    # 2. Render previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 3. Chat Input & Intent Routing
    if prompt := st.chat_input("Ex: 'What is our refund policy?' or 'Predict churn for total_orders=12'"):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Analyzing intent and routing request to FastAPI microservice..."):
                try:
                    # ROUTE A: Machine Learning Churn Risk Inference (FastAPI + XGBoost)
                    if any(keyword in prompt.lower() for keyword in ["churn", "predict", "risk"]):
                        # 1. Default feature values
                        recency = 120.0
                        frequency = 5.0
                        monetary_value = 450.0
                        refund_rate = 0.0

                        # 2. Extract dynamic numbers from chat prompt using Regex
                        recency_match = re.search(r'recency\s*=\s*(\d+\.?\d*)', prompt, re.IGNORECASE)
                        freq_match = re.search(r'frequency\s*=\s*(\d+\.?\d*)', prompt, re.IGNORECASE)
                        monetary_match = re.search(r'monetary\s*=\s*(\d+\.?\d*)', prompt, re.IGNORECASE)
                        refund_match = re.search(r'refund_rate\s*=\s*(\d+\.?\d*)', prompt, re.IGNORECASE)

                        if recency_match:
                            recency = float(recency_match.group(1))
                        if freq_match:
                            frequency = float(freq_match.group(1))
                        if monetary_match:
                            monetary_value = float(monetary_match.group(1))
                        if refund_match:
                            refund_rate = float(refund_match.group(1))

                        # 3. Construct dynamic payload for FastAPI
                        payload = {
                            "recency": recency,
                            "frequency": frequency,
                            "monetary_value": monetary_value,
                            "refund_rate": refund_rate
                        }

                        response = requests.post(f"{API_BASE_URL}/predict/CUST_INTERACTIVE", json=payload, timeout=5)

                        if response.status_code == 200:
                            data = response.json()
                            reply = (
                                f"*📊 ML Risk Inference Result (FastAPI + XGBoost Layer)*\n\n"
                                f"* *Customer ID:** '{data.get('customer_id', 'CUST_999')}'\n"
                                f"* *Evaluated Risk Level:** **{data.get('risk_level', 'High')}**\n"
                                f"* *Model Verdict:** Evaluated churn probability based on purchase recency and total spending metrics."
                            )
                        else:
                            reply = "⚠️ *API Warning:* Communicated with FastAPI, but received a non-200 status code."

                    # ROUTE B: Grounded Document Q&A (FastAPI + ChromaDB RAG Engine)
                    else:
                        payload = {"query": prompt}
                        response = requests.post(f"{API_BASE_URL}/api/v1/query", json=payload, timeout=10)

                        if response.status_code == 200:
                            data = response.json()
                            reply = f"*🧠 Grounded Operations Knowledge (ChromaDB Vector RAG):*\n\n{data.get('answer', data)}"
                        else:
                            reply = "⚠️ *API Warning:* Could not retrieve answer from the RAG endpoint."

                except requests.exceptions.ConnectionError:
                    reply = (
                        f"❌ *Backend Connection Error:* Unable to reach FastAPI at {API_BASE_URL}.\n\n"
                        "Make sure your FastAPI server is running in another terminal window using:\n"
                        "bash\nuvicorn api.app:app --reload\n"
                    )
                except requests.exceptions.RequestException as e:
                    reply = f"❌ *API Request Error:* {e!s}"

            st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
