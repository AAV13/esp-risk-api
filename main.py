import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os

#App Initialization
app = FastAPI(
    title="ESP Failure Analytics Project ",
    description="An API to assess ESP health, detecting clear faults and incipient degradation.",
    version="1.0.0"
)

#1. Loading Model and Encoder on Startup
#The models are loaded once once the application starts.
try:
    model_path = os.path.join(os.path.dirname(__file__), 'final_stacking_model.pkl')
    encoder_path = os.path.join(os.path.dirname(__file__), 'final_label_encoder.pkl')
    model = joblib.load(model_path)
    label_encoder = joblib.load(encoder_path)
except FileNotFoundError:
    raise RuntimeError("Model or encoder files not found. Ensure they are in the same directory.")

#2. Define the Input Data Model using Pydantic
#This ensures that the incoming data has the correct structure and types.
class ESPData(BaseModel):
    features: list[float]

    class Config:
        schema_extra = {
            "example": {
                "features": [-1.0, 0.2, 0.3, 0.1, 0.8, 0.4, 0.3]
            }
        }

#3. The Core Risk Assessment Logic
def get_risk_assessment(data: np.ndarray, model, encoder):
    probabilities = model.predict_proba(data)[0]
    prob_per_class = {label: float(prob) for label, prob in zip(encoder.classes_, probabilities)}

    #Level 1: RED ALERT (High-Confidence Fault)
    for label, prob in prob_per_class.items():
        if label != 'Normal' and prob > 0.85:
            return {
                "alert_level": "RED ALERT: HIGH-RISK FAULT",
                "action": "ESP at critical condition - initiate corrective maintenance.",
                "reason": f"Model has high confidence ({prob:.0%}) of a clear '{label}' fault.",
                "probabilities": prob_per_class
            }

    #Level 2: YELLOW ALERT (Incipient Fault / Degradation)
    if prob_per_class['Normal'] < 0.90:
        developing_fault = max(((l, p) for l, p in prob_per_class.items() if l != 'Normal'), key=lambda item: item[1])[0]
        return {
            "alert_level": "YELLOW ALERT: INCIPIENT FAULT WARNING",
            "action": "Flag for expert review. Potential for future failure.",
            "reason": f"ESP is operating within normal parameters, but the model detects a developing signature similar to '{developing_fault}'. This is an early warning.",
            "probabilities": prob_per_class
        }

    #Level 3: GREEN ALERT (Healthy Operation)
    return {
        "alert_level": "GREEN: HEALTHY",
        "action": "Safe for continued deployment.",
        "reason": f"Model has high confidence ({prob_per_class['Normal']:.0%}) of healthy operation.",
        "probabilities": prob_per_class
    }

#4. API Endpoints
@app.get("/", tags=["Health Check"])
def read_root():
    #Root endpoint providing a simple status message
    return {"status": "API is running. Go to /docs for interactive documentation."}

@app.post("/predict", tags=["Prediction"])
def predict_maintenance_alert(esp_data: ESPData):
    """
    Accepts ESP feature data and returns a prognostic risk assessment.
    """
    if len(esp_data.features) != 7:
        raise HTTPException(status_code=400, detail=f"Expected 7 features, but received {len(esp_data.features)}")

    data_to_predict = np.array(esp_data.features).reshape(1, -1)
    alert = get_risk_assessment(data_to_predict, model, label_encoder)
    return alert