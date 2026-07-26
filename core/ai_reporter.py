import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import pandas as pd
from groq import Groq
from dotenv import load_dotenv

class AIReporterEngine:
    def __init__(self, output_dir: str = "outputs"):
        load_dotenv()
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        groq_key = os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=groq_key) if groq_key else None

    def generate_reports(self, rfm_path: str, forecast_path: str, anomalies_path: str):
        """Loads processed metrics and uses Groq LLM to generate the executive report and supplier email."""
        rfm = pd.read_csv(rfm_path)
        forecast = pd.read_csv(forecast_path)
        anomalies = pd.read_csv(anomalies_path)
        anomalies = anomalies[anomalies['anomaly'] == -1]

        # Summary statistics
        total_customers = len(rfm)
        champions = (rfm['Segment'] == 'Champions').sum()
        at_risk = (rfm['Segment'] == 'At Risk').sum()
        churned = rfm['Churned'].sum()
        anomaly_count = len(anomalies)
        worst_delay = anomalies['avg_delay'].max() if not anomalies.empty else 0
        next_30_day_avg = forecast.tail(30)['yhat'].mean()

        report_prompt = f"""
You are an Operations Intelligence AI for an e-commerce company.
Generate a professional weekly operations report based on this data:

CUSTOMER ANALYTICS:
- Total customers: {total_customers:,}
- Champions (best customers): {champions:,}
- At Risk customers: {at_risk:,}
- Churned customers: {churned:,}

DEMAND FORECAST:
- Expected orders next 30 days: {next_30_day_avg:.0f} per day

ANOMALY DETECTION:
- Anomalous sellers flagged: {anomaly_count:,}
- Worst delivery delay detected: {worst_delay:.1f} days

Write a 3-paragraph executive summary with:
1. Overall business health
2. Key risks and alerts
3. Recommended actions for operations team
"""

        email_prompt = f"""
You are an AI assistant for an e-commerce operations team.
Write a professional supplier reorder email based on this forecast:

- Product category: bed_bath_table (highest demand category)
- Current avg daily orders: {next_30_day_avg:.0f}
- Forecast shows demand will increase next 30 days
- Current stock risk: Medium

Write a concise, professional reorder email to the supplier.
Include: subject line, greeting, order details, urgency, closing.
"""

        # Generate Executive Report
        report_res = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": report_prompt}]
        )
        summary_text = report_res.choices[0].message.content

        # Generate Supplier Email
        email_res = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": email_prompt}]
        )
        supplier_text = email_res.choices[0].message.content

        # Export text outputs
        with open(os.path.join(self.output_dir, 'weekly_report.txt'), 'w', encoding='utf-8') as f:
            f.write("=== SMARTOPS WEEKLY OPS REPORT ===\n\n" + summary_text)

        with open(os.path.join(self.output_dir, 'supplier_email.txt'), 'w', encoding='utf-8') as f:
            f.write("=== AI GENERATED SUPPLIER EMAIL ===\n\n" + supplier_text)

        return summary_text, supplier_text

    def send_report_email(self, summary_text: str, supplier_text: str, recipient_email: str, dashboard_url: str = "https://your-app.streamlit.app"):
        """Formats the unified payload and dispatches via Brevo SMTP."""
        smtp_server = "smtp-relay.brevo.com"
        port = 587
        login_email = os.getenv("BREVO_LOGIN")
        sender_email = os.getenv("SENDER_EMAIL")
        sender_password = os.getenv("BREVO_SMTP_KEY")

        clean_supplier = supplier_text.replace("Subject: Urgent Reorder Request for Bed, Bath, and Table Products\n\n", "")
        supplier_body_only = clean_supplier.partition("Best regards,")[0].strip()

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
👉 CLICK HERE TO OPEN DASHBOARD: {dashboard_url}
======================================================================

======================================================================
Best regards,
Vinay
E-commerce Operations Team
======================================================================
"""

        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = "SmartOps Weekly Operations Report"
        msg.attach(MIMEText(payload, 'plain'))

        with smtplib.SMTP(smtp_server, port) as server:
            server.starttls()
            server.login(login_email, sender_password)
            server.sendmail(sender_email, recipient_email, msg.as_string())

        print(f"✅ Rearranged email sequence dispatched successfully to {recipient_email}!")