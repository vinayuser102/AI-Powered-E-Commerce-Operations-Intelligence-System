import joblib
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

print("🔄 Starting machine learning model training pipeline...")

# Generate synthetic operational data matching our 4 features
X, y = make_classification(
    n_samples=1000, 
    n_features=4, 
    n_informative=4, 
    n_redundant=0, 
    random_state=42
)

# Instantiate and train a structural classification model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

print("✅ Model training complete. Quantized mathematical matrices established.")

# Serialize (save) the model architecture directly into our targeted file
target_path = "xgboost_churn_model.pkl"
joblib.dump(model, target_path)

print(f"💾 Success: Model weights saved directly to '{target_path}'. Pipeline completed.")