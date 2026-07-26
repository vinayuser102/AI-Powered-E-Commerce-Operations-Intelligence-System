from typing import Literal

from pydantic import BaseModel, Field


class CustomerFeatures(BaseModel):
    recency: int = Field(..., description="Days since last purchase", ge=0)
    frequency: int = Field(..., description="Total number of completed orders", ge=1)
    monetary_value: float = Field(..., description="Total lifetime revenue generated", ge=0.0)
    refund_rate: float = Field(0.0, description="Customer-level operational refund rate", ge=0.0, le=1.0)

class PredictionResponse(BaseModel):
    customer_id: str
    churn_probability: float
    risk_level: str


class RagQuery(BaseModel):
    query: str = Field(..., min_length=1, max_length=2_000, description="Operations-policy question")
    max_results: int = Field(2, ge=1, le=5, description="Maximum policy chunks to retrieve")


class RagSource(BaseModel):
    source: str
    chunk: int


class RagResponse(BaseModel):
    answer: str
    status: Literal["success", "no_context"]
    sources: list[RagSource]
