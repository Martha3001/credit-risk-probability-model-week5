import os
import sys
from src.api.pydantic_models import CustomerFeatures, PredictionResponse
from fastapi import FastAPI
import mlflow
import pandas as pd

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

app = FastAPI(title="Credit Risk Scoring API")

# Set MLflow tracking URI to notebooks/mlruns
mlflow.set_tracking_uri("file:////app/notebooks/mlruns")

# Load model from MLflow Model Registry
MODEL_NAME = "CreditRiskRandomForest"
MODEL_VERSION = 3
model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{MODEL_NAME}/{MODEL_VERSION}"
    )


@app.get("/")
def root():
    return {"message": "Credit Risk Prediction API is live."}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: CustomerFeatures):
    # Convert incoming features to DataFrame
    data = pd.DataFrame([features.dict()])
    # MLflow PyFuncModel uses .predict, which may return probability or class
    prediction = model.predict(data)
    # If prediction is probability, use it;
    # if class, set risk_proba to 1 if high risk
    if hasattr(prediction, '__iter__') and len(prediction) > 0:
        risk_proba = float(prediction[0])
    else:
        risk_proba = float(prediction)

    return PredictionResponse(risk_probability=risk_proba)
