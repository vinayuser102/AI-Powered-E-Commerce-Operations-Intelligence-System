"""FastAPI service for SmartOps churn scoring and grounded policy Q&A."""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import chromadb
import joblib
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from groq import Groq

from .schemas import CustomerFeatures, PredictionResponse, RagQuery, RagResponse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

MODEL_PATH = Path(os.getenv("CHURN_MODEL_PATH", PROJECT_ROOT / "api" / "xgboost_churn_model.pkl"))
CHROMA_PATH = Path(os.getenv("CHROMA_PATH", PROJECT_ROOT / "chroma_storage"))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "smartops_policies")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
logger = logging.getLogger(__name__)

chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
ml_models: dict[str, object | None] = {"churn": None}


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load the optional churn model once, then release it on shutdown."""
    if MODEL_PATH.is_file() and MODEL_PATH.stat().st_size > 0:
        try:
            ml_models["churn"] = joblib.load(MODEL_PATH)
            logger.info("Loaded churn model from %s", MODEL_PATH)
        except Exception:
            logger.exception("Could not load churn model; using deterministic fallback")
    else:
        logger.warning("No churn model at %s; using deterministic fallback", MODEL_PATH)

    yield
    ml_models.clear()
    logger.info("Released churn model from memory")


app = FastAPI(
    title="SmartOps API",
    description="Customer churn scoring and grounded operations-policy Q&A.",
    version="1.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check():
    """Return the service status and whether a trained model is loaded."""
    return {
        "status": "healthy",
        "service": "SmartOps API",
        "model_loaded": ml_models.get("churn") is not None,
    }


@app.post("/predict/{customer_id}", response_model=PredictionResponse)
async def predict_churn(customer_id: str, features: CustomerFeatures):
    """Score a validated customer feature vector for churn risk."""
    try:
        input_matrix = [[features.recency, features.frequency, features.monetary_value, features.refund_rate]]
        model = ml_models.get("churn")
        if model is not None:
            probability = float(model.predict_proba(input_matrix)[0][1])
        else:
            probability = 0.85 if features.recency > 90 or features.refund_rate > 0.15 else 0.25

        return PredictionResponse(
            customer_id=customer_id,
            churn_probability=probability,
            risk_level="HIGH" if probability > 0.70 else "LOW",
        )
    except Exception as exc:
        logger.exception("Churn inference failed for customer %s", customer_id)
        raise HTTPException(status_code=500, detail="Inference engine failure") from exc


@app.post("/api/v1/query", response_model=RagResponse)
async def query_rag(payload: RagQuery):
    """Answer only from retrieved policy chunks and return their provenance."""
    try:
        results = collection.query(
            query_texts=[payload.query],
            n_results=payload.max_results,
            include=["documents", "metadatas", "distances"],
        )
        documents = results.get("documents", [[]])[0] or []
        metadatas = results.get("metadatas", [[]])[0] or []

        if not documents:
            return RagResponse(
                answer="Policy details are unavailable.", status="no_context", sources=[]
            )

        sources = [
            {
                "source": (metadata or {}).get("source", "knowledge_base.txt"),
                "chunk": (metadata or {}).get("chunk", index),
            }
            for index, metadata in enumerate(metadatas)
        ]
        context = "\n---\n".join(documents)
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise HTTPException(status_code=503, detail="GROQ_API_KEY is not configured")

        completion = Groq(api_key=api_key).chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are the SmartOps Assistant. Answer only from the supplied policy context. "
                        "If the context does not contain the answer, respond exactly: Policy details are unavailable."
                    ),
                },
                {"role": "user", "content": f"Policy context:\n{context}\n\nQuestion: {payload.query}"},
            ],
            model=GROQ_MODEL,
            temperature=0,
            max_completion_tokens=300,
        )
        answer = completion.choices[0].message.content or "Policy details are unavailable."
        return RagResponse(answer=answer, status="success", sources=sources)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("RAG query failed")
        raise HTTPException(status_code=500, detail="RAG engine failure") from exc
