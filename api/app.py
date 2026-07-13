from fastapi import FastAPI, HTTPException
# We import the strict data validations we just built in step 6
from schemas import CustomerFeatures, PredictionResponse

# Initializing the core web application engine
app = FastAPI(
    title="NexusRisk API Engine",
    description="Production endpoint serving live customer risk scoring profiles.",
    version="1.0.0"
)

# 1. System Health Check Router (GET Method)
@app.get("/health")
async def health_check():
    """
    Returns the operational status of the core engine infrastructure.
    """
    return {"status": "healthy", "service": "NexusRisk Core Engine"}

# 2. ML Inference Processing Router (POST Method)
@app.post("/predict/{customer_id}", response_model=PredictionResponse)
async def predict_churn(customer_id: str, features: CustomerFeatures):
    try:
        # Placeholder computation logic to verify our data routing works perfectly
        mock_probability = 0.81
        risk = "HIGH" if mock_probability > 0.70 else "LOW"
        
        return PredictionResponse(
            customer_id=customer_id,
            churn_probability=mock_probability,
            risk_level=risk
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
    

import os
import joblib
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
# Importing the strict validation schemas we built earlier
from schemas import CustomerFeatures, PredictionResponse

# Global dictionary memory space to safely anchor our model weights in RAM
ml_models = {}

# 1. The Production Lifespan Manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # This path matches the empty file we created on your V drive
    model_path = "xgboost_churn_model.pkl"
    
    # Check if the file actually contains trained data weights
    if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
        try:
            ml_models["churn"] = joblib.load(model_path)
            print("🚀 Success: Trained XGBoost model loaded into RAM.")
        except Exception as e:
            print(f"⚠️ Error loading model file: {e}. Falling back to dummy framework.")
            ml_models["churn"] = None
    else:
        print("ℹ️ Info: Empty model file placeholder detected. Operating in simulation mode.")
        ml_models["churn"] = None
        
    yield
    # Everything after 'yield' runs when you hit CTRL+C to close the server
    ml_models.clear()
    print("🛑 Clean Up: Model memory cleared from RAM.")

# Passing our lifespan engine directly into the application instance
app = FastAPI(
    title="NexusRisk API Engine",
    description="Production endpoint serving live customer risk scoring profiles.",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "NexusRisk Core Engine"}

@app.post("/predict/{customer_id}", response_model=PredictionResponse)
async def predict_churn(customer_id: str, features: CustomerFeatures):
    try:
        # Step A: Transform the incoming validated JSON data into a 2D Array format that ML models expect
        input_matrix = [
            [features.recency, features.frequency, features.monetary_value, features.refund_rate]
        ]
        
        # Step B: If the model is loaded in memory, compute actual inference probability
        if ml_models["churn"] is not None:
            # predict_proba returns an array of probabilities: [[prob_class_0, prob_class_1]]
            # We extract the second value [0][1] which represents the probability of Churn (Class 1)
            prob = float(ml_models["churn"].predict_proba(input_matrix)[0][1])
        else:
            # Fallback placeholder calculation logic if the model file is still empty
            prob = 0.81
            
        risk = "HIGH" if prob > 0.70 else "LOW"
        
        return PredictionResponse(
            customer_id=customer_id,
            churn_probability=prob,
            risk_level=risk
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference Engine Failure: {str(e)}")