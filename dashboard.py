import streamlit as st
import numpy as np
import joblib
import os
import pandas as pd

#Page Configuration

st.set_page_config(
    page_title="ESP Failure Analytics",
    page_icon="⚡",
    layout="wide"
)

#Cached Model Loading
#Using st.cache_resource to load the models only once

@st.cache_resource
def load_models():
    #Loading the ML model and label encoder
    try:
        model_path = os.path.join(os.path.dirname(__file__), 'final_stacking_model.pkl')
        encoder_path = os.path.join(os.path.dirname(__file__), 'final_label_encoder.pkl')
        model = joblib.load(model_path)
        label_encoder = joblib.load(encoder_path)
        return model, label_encoder
    except FileNotFoundError:
        #Displaying an error in the app if files are not found
        st.error("Model or encoder files not found. Ensure they are in the same directory as dashboard.py.")
        return None, None

model, label_encoder = load_models()

#Logic from the FastAPI App from main.py

DRIFT_THRESHOLDS = {
    0: ("median(8,13)", 0.003, 'upper'),
    1: ("rms(98,102)", 0.18, 'upper'),
    2: ("median(98,102)", 0.002, 'upper'),
    4: ("peak2x", 0.012, 'upper'),
    5: ("a", -0.001, 'lower'),
    6: ("b", -6.0, 'upper')
}

def get_risk_assessment(data: np.ndarray, model, encoder):
    probabilities = model.predict_proba(data)[0]
    prob_per_class = {label: float(prob) for label, prob in zip(encoder.classes_, probabilities)}

    drift_flags = []
    for i, val in enumerate(data[0]):
        if i in DRIFT_THRESHOLDS:
            feature_name, threshold, bound_type = DRIFT_THRESHOLDS[i]
            is_drifting = (bound_type == 'upper' and val > threshold) or \
                          (bound_type == 'lower' and val < threshold)
            if is_drifting:
                drift_flags.append(f"{feature_name} (value: {val:.3f})")

    # --- Tier 1: RED ALERT ---
    for label, prob in prob_per_class.items():
        if label != 'Normal' and prob > 0.85:
            return {
                "alert_level": "RED ALERT: HIGH-RISK FAULT",
                "action": f"ESP at critical condition. High-confidence fault found: '{label}'.",
                "reason": f"Model has high confidence ({prob:.0%}) of a clear '{label}' fault.",
                "probabilities": prob_per_class
            }

    #To identify the top developing fault and its probability
    top_fault_label, top_fault_prob = max(
        ((label, prob) for label, prob in prob_per_class.items() if label != 'Normal'),
        key=lambda item: item[1]
    )

    # --- Tier 2: YELLOW ALERT (Developing Fault > 17% or Feature Drift) ---
    if top_fault_prob > 0.17 or len(drift_flags) > 0:
        reason_parts = []
        if top_fault_prob > 0.17:
            reason_parts.append(f"A potential '{top_fault_label}' fault is developing with {top_fault_prob:.1%} probability.")
        if len(drift_flags) > 0:
            reason_parts.append(f"Feature drift detected in: {', '.join(drift_flags)}.")
            
        return {
            "alert_level": "YELLOW ALERT: INCIPIENT FAULT WARNING",
            "action": "Flag for expert review. Potential for future failure.",
            "reason": " ".join(reason_parts),
            "probabilities": prob_per_class
        }

    # --- Tier 3: GREEN ALERT ---
    return {
        "alert_level": "GREEN: HEALTHY",
        "action": "Safe for continued deployment.",
        "reason": f"Model has high confidence ({prob_per_class['Normal']:.0%}) of healthy operation.",
        "probabilities": prob_per_class
    }

#Streamlit UI

st.title("ESP Failure Analytics Dashboard")
st.write(
    "An interactive tool to assess Electric Submersible Pump (ESP) health. "
    "Input the seven key feature values in the sidebar to get a real-time risk assessment."
)

st.sidebar.header("Input ESP Features")

#Define feature names for labels
feature_names = [
    "median(8,13)", "rms(98,102)", "median(98,102)",
    "peak1x", "peak2x", "a", "b"
]

#Using a dict to hold the input values
inputs = {}

#Creating nos. inputs in the sidebar
for i, name in enumerate(feature_names):
    # Providing a default value based on your previous example
    default_val = [-1.0, 0.2, 0.3, 0.1, 0.8, 0.4, 0.3][i]
    inputs[name] = st.sidebar.number_input(f"Feature {i+1}: {name}", value=default_val, format="%.3f")

#Assess button
assess_button = st.sidebar.button("Assess ESP Health", type="primary")

#Main Panel for Results

if assess_button and model:
    #Collecting features from the input widgets
    features_list = [inputs[name] for name in feature_names]
    data_to_predict = np.array(features_list).reshape(1, -1)

    #Getting the assessment
    alert = get_risk_assessment(data_to_predict, model, label_encoder)
    alert_level = alert["alert_level"]

    #Display the results in a color-coded metric box
    st.header("Risk Assessment Result")
    
    if "RED ALERT" in alert_level:
        st.error(f"## {alert_level}", icon="🚨")
    elif "YELLOW ALERT" in alert_level:
        st.warning(f"## {alert_level}", icon="⚠️")
    else: 
        st.success(f"## {alert_level}", icon="✅")

    #Displaying the action and reason
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Recommended Action")
        st.write(alert['action'])
    with col2:
        st.subheader("Reasoning")
        st.write(alert['reason'])

    #Display the probabilities in an expandable section
    with st.expander("Show Detailed Probabilities"):
        probs_df = pd.DataFrame.from_dict(
            alert['probabilities'],
            orient='index',
            columns=['Probability']
        ).sort_values(by='Probability', ascending=False)
        
        st.bar_chart(probs_df)
else:
    #Initial instruction message
    st.info("Please input feature values in the sidebar and click 'Assess ESP Health'.")
