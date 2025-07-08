import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from fastapi.middleware.cors import CORSMiddleware

#App Initialization
app = FastAPI(
    title="ESP Failure Analytics Project",
    description="An API to assess ESP health, detecting clear faults and incipient degradation using a hybrid ML and rule-based system.",
    version="1.1.0" #Version updated to reflect new logic
)

#CORS Middleware
#This allows your frontend to communicate with this API
origins = [
    "http://localhost",
    "http://localhost:5500",
    "http://127.0.0.1",
    "http://127.0.0.1:5500",
    "null",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#Load Models and Configuration on Startup
try:
    # Load ML model and label encoder
    model_path = os.path.join(os.path.dirname(__file__), 'final_stacking_model.pkl')
    encoder_path = os.path.join(os.path.dirname(__file__), 'final_label_encoder.pkl')
    model = joblib.load(model_path)
    label_encoder = joblib.load(encoder_path)

except FileNotFoundError:
    raise RuntimeError("Model or encoder files not found. Ensure they are in the same directory.")

#Data-Driven Drift Thresholds
#Based on the boxplot analysis of Healthy vs. Ambiguous Normals during analysis/feature selection phase
DRIFT_THRESHOLDS = {
    #Feature Index: (Feature Name, Threshold, 'upper' or 'lower' bound)
    0: ("median(8,13)", 0.003, 'upper'),
    1: ("rms(98,102)", 0.18, 'upper'),
    2: ("median(98,102)", 0.002, 'upper'),
    #3: ("peak1x", ...), #No threshold set - analysis showed it's a weak indicator
    4: ("peak2x", 0.012, 'upper'),
    5: ("a", -0.001, 'lower'), #Lower bound: values *below* this are a concern
    6: ("b", -6.0, 'upper')  #Upper bound: values *above* this are a concern
}

#Pydantic Input Model
class ESPData(BaseModel):
    features: list[float]

    class Config:
        schema_extra = {
            "example": {
                "features": [-1.0, 0.2, 0.3, 0.1, 0.8, 0.4, 0.3]
            }
        }

#Core Risk Assessment Logic
def get_risk_assessment(data: np.ndarray, model, encoder):
    probabilities = model.predict_proba(data)[0]
    prob_per_class = {label: float(prob) for label, prob in zip(encoder.classes_, probabilities)}

    #Check for feature drift based on predefined thresholds
    drift_flags = []
    for i, val in enumerate(data[0]):
        if i in DRIFT_THRESHOLDS:
            feature_name, threshold, bound_type = DRIFT_THRESHOLDS[i]
            
            is_drifting = False
            if bound_type == 'upper' and val > threshold:
                is_drifting = True
            elif bound_type == 'lower' and val < threshold:
                is_drifting = True
            
            if is_drifting:
                drift_flags.append(f"{feature_name} (value: {val:.3f})")

    # --- Tier 1: RED ALERT (High-Confidence Fault) ---
    for label, prob in prob_per_class.items():
        if label != 'Normal' and prob > 0.85:
            return {
                "alert_level": "RED ALERT: HIGH-RISK FAULT",
                "action": "ESP at critical condition - initiate corrective maintenance.",
                "reason": f"Model has high confidence ({prob:.0%}) of a clear '{label}' fault.",
                "probabilities": prob_per_class
            }

    # --- Tier 2: YELLOW ALERT (Incipient Fault or Feature Drift) ---
    if prob_per_class['Normal'] < 0.90 or len(drift_flags) > 0:
        developing_fault = max(((l, p) for l, p in prob_per_class.items() if l != 'Normal'), key=lambda item: item[1])[0]
        
        reason_parts = []
        if prob_per_class['Normal'] < 0.90:
            reason_parts.append(f"Model predicts 'Normal' operation, but confidence is low. Possible developing fault: '{developing_fault}'.")
        if len(drift_flags) > 0:
            reason_parts.append(f"Feature drift detected in: {', '.join(drift_flags)}.")
            
        return {
            "alert_level": "YELLOW ALERT: INCIPIENT FAULT WARNING",
            "action": "Flag for expert review. Potential for future failure.",
            "reason": " ".join(reason_parts),
            "probabilities": prob_per_class
        }

    # --- Tier 3: GREEN ALERT (Healthy Operation) ---
    return {
        "alert_level": "GREEN: HEALTHY",
        "action": "Safe for continued deployment.",
        "reason": f"Model has high confidence ({prob_per_class['Normal']:.0%}) of healthy operation.",
        "probabilities": prob_per_class
    }

#API Endpoints
@app.get("/", tags=["Health Check"])
def read_root():
    """Root endpoint for basic health check."""
    return {"status": "API is running. Go to /docs for interactive documentation."}

@app.post("/predict", tags=["Prediction"])
#Accepts ESP feature data and returns a prognostic risk assessment.
def predict_maintenance_alert(esp_data: ESPData):
    if len(esp_data.features) != 7:
        raise HTTPException(status_code=400, detail=f"Expected 7 features, but received {len(esp_data.features)}")

    data_to_predict = np.array(esp_data.features).reshape(1, -1)
    alert = get_risk_assessment(data_to_predict, model, label_encoder)
    return alert
