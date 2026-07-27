# 🤖 AI-Powered E-Commerce Operations Intelligence System (SmartOps AI)

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Service%20Layer-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-FF6F61?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-LLM%20AI-00A67E?style=for-the-badge)
![Prophet](https://img.shields.io/badge/Prophet-Forecasting-0467DF?style=for-the-badge)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![GitHub](https://img.shields.io/badge/Status-Complete-10B981?style=for-the-badge)

**An enterprise-grade Operations Intelligence & REST API Platform for modern e-commerce.**  
Built on 99,441 real orders from the Olist Brazilian E-Commerce dataset.

[📊 View Dashboard](#-streamlit-dashboard) • [🚀 Quick Start](#-quick-start) • [📁 Project Structure](#-project-structure) • [📈 Key Findings](#-key-findings--results)

</div>

---

## 📌 1. Why This System? (The Real-World Problem)

Most e-commerce businesses drown in fragmented data streams but starve for fast, actionable decisions:

1. **Unstructured Operational Knowledge:** Shipping rules, return policies, seller compliance guides, and support logs are scattered across text files. Operations teams waste hours manually searching through documents to resolve issues.
2. **Isolated Machine Learning Insights:** Predictive models (demand forecasting, customer churn, seller fraud) stay trapped in offline scripts without standardized APIs for downstream platforms to access them in real time.
3. **Operational Inefficiencies:** Manual log analysis slows down anomaly response times, leading to shipping delays, unexpected stockouts, and poor customer retention.

**SmartOps AI** solves this by unifying **Predictive Analytics Models**, a **RAG Intelligence Core**, and an asynchronous **FastAPI Service Layer** into a single microservice platform.

---

## 🏗️ 2. System Architecture

```text
                               ┌─────────────────────────────────────────┐
                               │           Client Applications           │
                               │  (Streamlit Dashboard / External Web)   │
                               └────────────────────┬────────────────────┘
                                                    │
                                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                             FastAPI Microservice Layer (`api/`)                                 │
│  • Pydantic Input/Output Validation   • Asynchronous Uvicorn Server   • Interactive Swagger UI  │
│  • ML Inference Route (`/predict/churn`)                          • RAG Endpoint (`/api/v1/query`)│
└───────────────────────────────┬─────────────────────────────────┬───────────────────────────────┘
                                │                                 │
                                ▼                                 ▼
┌───────────────────────────────────────────────┐ ┌───────────────────────────────────────────────┐
│      Predictive Analytics ML Engine           │ │        RAG Intelligence Core                  │
│  • Demand Forecasting (Prophet)               │ │        (`rag_qa_engine/`)                      │
│  • Customer Churn & RFM (Gradient Boosting)   │ │  • ChromaDB Local Vector Database             │
│  • Seller Anomaly Detection (Isolation Forest)│ │  • Semantic Document Retrieval                │
│  • Automated AI Reporting (Groq + Brevo SMTP) │ │  • Grounded Llama-3 LLM Generation            │
└───────────────────────────────────────────────┘ └───────────────────────────────────────────────┘
```

---

## 📦 Modules

### Module 1 — Demand Forecasting Engine
> *How much stock do we need for the next 90 days?*

- Algorithm: **Facebook Prophet** (time-series forecasting)
- Input: Daily order counts aggregated from 99,441 orders
- Output: 90-day forecast with upper/lower confidence bounds
- Key finding: Black Friday spike detected in November 2017

### Module 2 — Customer Behavior Analytics
> *Which customers are loyal, which are leaving, which are lost?*

- Framework: **RFM Analysis** (Recency, Frequency, Monetary)
- Algorithm: **Gradient Boosting Classifier** for churn prediction
- Input: 96,478 unique customer purchase histories
- Output: 5 customer segments + individual churn probability scores

### Module 3 — Operations Anomaly Detector
> *Which sellers are behaving unusually and why?*

- Algorithm: **Isolation Forest** (unsupervised ML)
- Input: Seller-level aggregates across 4 operational features (avg delay, order volume, avg price, total revenue)
- Output: 149 anomalous sellers flagged from 2,970 total
- No labeled data required — fully unsupervised

### Module 4 — AI Report & Email Generator
> *What should operations management know this week?*

- Integration: **Groq API** + **llama-3.3-70b-versatile**
- Output 1: Weekly executive operations summary report
- Output 2: Supplier reorder email (triggered by forecast)
- Fully automated — zero manual writing required
- **Email Dispatch: Automated via Brevo SMTP** — report
  sent directly to recipient email automatically

### Module 5 — RAG Vector Intelligence & REST API Layer
> *How can web applications search operational policies and get real-time ML risk scores?*

- Vector Database: ChromaDB local persistent store (`rag_qa_engine/chroma_storage/`)
- LLM Engine: Groq API (`llama-3.3-70b-versatile`) with context-bound system prompts
- Microservice Layer: FastAPI application (`api/app.py`) providing Pydantic-validated REST endpoints (`/health`, `/predict/churn`, `/api/v1/query`)

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Total Orders Analyzed | 99,441 |
| Total Revenue Tracked | R$20,470,727 |
| Delivery Rate | 97.1% |
| Avg Delivery (vs estimated) | 12 days early |
| Anomalous Sellers Detected | 149 / 2,970 |
| At-Risk Customers Identified | 39,822 (41.3%) |
| Champion Customers | 7,781 (8.1%) |
| Forecast Horizon | 90 days |
| Forecast MAE | 81.48 orders/day |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.12, asyncio |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost |
| Time Series | Facebook Prophet |
| Visualization | Plotly, Matplotlib, Seaborn |
| Dashboard | Streamlit |
| Version Control | Git + GitHub |
| CI/CD | GitHub Actions (ruff + pytest) |
| Dataset | Olist Brazilian E-Commerce (Kaggle) |
| LLM & AI Integration | Groq API (`llama-3.3-70b-versatile`) |
| API & Backend | FastAPI, Uvicorn, Pydantic, OpenAPI / Swagger |
| Vector DB & Search | ChromaDB (Local Persistent Embedding Storage) |
| Email Dispatch | Brevo SMTP Relay |

---

## 📁 Project Structure

```text
Smart Ops/
│
├── api/                               # FastAPI Web Service Layer
│   ├── app.py                         # REST API routes & app entrypoint
│   ├── schemas.py                     # Pydantic data validation schemas
│   ├── test_client.py                 # Manual smoke-test client
│   ├── train_model.py                 # XGBoost demo model training (synthetic data)
│   └── xgboost_churn_model.pkl        # Serialized ML model binary
│
├── ml_engine/                         # Core ML Pipeline Modules
│   ├── forecast_pipeline.py           # Prophet demand forecasting engine
│   ├── churn_pipeline.py              # Gradient Boosting churn model (real data)
│   └── anomaly_detector.py            # Isolation Forest anomaly detection
│
├── core/                              # AI Reporting Engine
│   └── ai_reporter.py                 # Groq LLM report generation + Brevo email dispatch
│
├── rag_qa_engine/                     # RAG & Vector Intelligence Core
│   ├── chroma_storage/                # Local persistent vector store
│   ├── ingest.py                      # Text chunking & ChromaDB vector ingestion
│   ├── knowledge_base.txt             # Domain knowledge base & policy corpus
│   └── query_engine.py                # Semantic search & Groq LLM pipeline
│
├── Data/
│   ├── raw/                           # 9 Olist CSV files
│   └── Processed/                     # Cleaned outputs
│       ├── master_data.csv
│       ├── demand_forecast.csv
│       ├── rfm_segments.csv
│       └── seller_anomalies.csv
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_module1_demand.ipynb
│   ├── 03_module2_customer.ipynb
│   ├── 04_module3_anomaly.ipynb
│   └── 05_module4_ai_report.ipynb
│
├── dashboard/
│   └── app.py                         # Streamlit multi-page application
│
├── tests/
│   ├── test_api.py                    # FastAPI endpoint tests
│   └── test_ingest.py                 # RAG ingestion tests
│
├── outputs/                           # Generated charts and reports
│   ├── monthly_orders.png
│   ├── monthly_revenue.png
│   ├── top_categories.png
│   ├── order_status.png
│   ├── delivery_delay.png
│   ├── review_scores.png
│   ├── demand_forecast.png
│   ├── customer_segments.png
│   ├── anomaly_detection.png
│   ├── weekly_report.txt
│   └── supplier_email.txt
│
├── .github/workflows/ci.yml          # GitHub Actions CI pipeline
├── Dockerfile                         # API service container
├── docker-compose.yml                 # Multi-service orchestration
├── requirements.txt                   # Python dependencies
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/vinayuser102/AI-Powered-E-Commerce-Operations-Intelligence-System.git
cd AI-Powered-E-Commerce-Operations-Intelligence-System
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure secrets

Create a `.env` file in the project root. Never add API keys to notebooks, source files, or Git.

```env
GROQ_API_KEY=your_groq_api_key
BREVO_SMTP_KEY=your_brevo_smtp_key
SENDER_EMAIL=your_sender_email
BREVO_LOGIN=your_brevo_login
```

> `GROQ_API_KEY` is required for LLM-powered reporting (Module 4) and RAG Q&A (Module 5).
> `BREVO_SMTP_KEY`, `SENDER_EMAIL`, and `BREVO_LOGIN` are required for automated email dispatch.

For Streamlit Cloud, copy `.streamlit/secrets.toml.example` into its Secrets settings and replace the placeholders there.

### 3. Download the dataset

Download the [Olist Brazilian E-Commerce Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) from Kaggle and place all 9 CSV files in `Data/raw/`.

### 4. Run the notebooks (in order)
```bash
jupyter notebook
```
Run notebooks in sequence: 01 → 02 → 03 → 04 → 05

### 5. Ingest the policy knowledge base and launch the API

```bash
python rag_qa_engine/ingest.py
uvicorn api.app:app --reload
```

The API provides `GET /health`, `POST /predict/{customer_id}`, and `POST /api/v1/query`. RAG answers include source provenance and return a safe fallback when no policy context is available. Interactive documentation is available at `http://127.0.0.1:8000/docs`.

### 6. Launch the Streamlit interactive dashboard
```bash
cd dashboard
python -m streamlit run app.py
```

### 7. Run tests and lint checks

```bash
pytest
ruff check api core ml_engine rag_qa_engine dashboard tests
```

### 8. Run with Docker (optional)

```bash
docker compose up --build
```

This starts the API on port 8000 and the dashboard on port 8501. Add all required keys to `.env` before starting.

---

## 🖥️ Dashboard

The Streamlit dashboard has 6 pages:

| Page | Content |
|------|---------|
| Overview | KPI cards + Monthly order volume chart |
| Demand Forecast | Interactive 90-day Prophet forecast |
| Customer Segments | RFM pie chart + segment table + churn metric |
| Anomaly Detection | Scatter plot with flagged sellers highlighted |
| AI Reports | Generated weekly report + supplier email |
| 🤖 AI Assistant | Chat interface for RAG Q&A and real-time churn risk scoring |

---

## 📋 Dataset Reference

| File | Records |
|------|---------|
| olist_orders_dataset.csv | 99,441 |
| olist_order_items_dataset.csv | 112,650 |
| olist_customers_dataset.csv | 99,441 |
| olist_products_dataset.csv | 32,951 |
| olist_order_reviews_dataset.csv | 99,224 |
| olist_order_payments_dataset.csv | 103,886 |
| olist_sellers_dataset.csv | 3,095 |

---

## 🔑 Key Findings

### Analytics & Predictive Insights:
- **97.1% delivery rate** — Olist maintains strong fulfillment performance
- **41.3% of customers are At Risk** — major retention opportunity identified
- **Only 8.1% are Champions** — loyalty program urgently needed
- **149 sellers flagged as anomalous** — worst flagged seller has 35-day avg delivery delay
- **Black Friday pattern confirmed** — November 2017 shows clear demand spike
- **R$20.4M revenue** tracked across 25 months of operations

### API & RAG Microservice Engineering Results

1. **Context-Bounded RAG Generation:**
   By strictly bounding system prompts with retrieved context chunks from internal documentation (`knowledge_base.txt`), the RAG engine ensures that generated responses are grounded in verified company policies and guidelines. When no relevant context is found, the system returns a safe fallback instead of generating ungrounded text.

2. **Production-Grade Payload Reliability:**
   Introducing **Pydantic schema validation** at the FastAPI layer eliminated runtime data type errors and malformed payload crashes on incoming HTTP requests.

3. **Decoupled & Scalable Microservice Architecture:**
   Separating the codebase into modular components (`api/` for web routes, `rag_qa_engine/` for vector intelligence, `ml_engine/` for model pipelines, and `core/` for report generation) allows ML models, vector databases, and REST endpoints to be developed, tested, or containerized independently without breaking downstream services.

4. **Enterprise Repository Security & Hygiene:**
   Implementing strict environment separation via `.env` files and `.gitignore` rules prevented sensitive credentials (like `GROQ_API_KEY`) and heavy binary directories (`.venv/`, `chroma_storage/`) from leaking into public source control.

---

**Vinay Sharma**  
Data Analyst | Operations Analytics | AI/ML
GitHub: [@vinayuser102](https://github.com/vinayuser102)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<div align="center">

⭐ Star this repo if you found it useful

</div>
