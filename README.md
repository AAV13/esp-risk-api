# ESP Failure Analytics & Predictive Maintenance Dashboard

A predictive maintenance tool to assess the health of offshore Electric Submersible Pumps (ESPs). This project uses a hybrid machine learning and rule-based system to detect both clear faults and subtle signs of degradation, providing a real-time risk assessment through an interactive web dashboard and a REST API.

---

## 🚀 Key Features

-   **Hybrid Intelligence:** Combines a Scikit-learn classification model with rule-based thresholds for robust fault detection.
-   **Three-Tier Alert System:** Classifies ESP health into `RED ALERT` (High-Risk Fault), `YELLOW ALERT` (Incipient Warning), and `GREEN` (Healthy).
-   **Incipient Fault Detection:** Identifies subtle "feature drift" to flag potential issues before they become critical failures.
-   **Interactive Dashboard:** A user-friendly Streamlit dashboard allows for real-time data input and visual feedback.
-   **REST API:** A FastAPI backend provides programmatic access to the prediction logic.

---

## 📊 Dashboard Preview

The interactive dashboard provides an intuitive interface for real-time risk assessment. Users can input sensor feature values and immediately receive a color-coded health status, analysis, and recommended actions.

<img width="948" height="458" alt="image" src="https://github.com/user-attachments/assets/b365bedd-47ef-43f7-8ea8-b6631cfa36c0" />

---

## 🛠️ Technologies Used

-   **Backend:** FastAPI, Uvicorn
-   **Frontend:** Streamlit
-   **Machine Learning:** Scikit-learn, Joblib, NumPy, Pandas
-   **Language:** Python

---

## ⚙️ Installation & Setup

Follow these steps to set up and run the project locally.

### 1. Clone the Repository

```bash
git clone [https://github.com/your-username/esp-risk-api.git](https://github.com/your-username/esp-risk-api.git)
cd esp-risk-api

2. Create and Activate a Virtual Environment
It's highly recommended to use a virtual environment to manage project dependencies.

# Create a venv
python -m venv venv

# Activate it
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

3. Install Dependencies
The requirements.txt file contains all the necessary packages.

pip install -r requirements.txt

Note: Ensure your final_stacking_model.pkl and final_label_encoder.pkl files are present in the root directory.

▶️ How to Run
You can run the API and the Dashboard independently.

Running the Streamlit Dashboard (Recommended)
To launch the interactive web dashboard, run the following command in your terminal:

streamlit run dashboard.py

A new tab will open in your browser at http://localhost:8501.

Running the FastAPI Backend
To start the API server, run:

uvicorn main:app --reload

The API will be accessible at http://127.0.0.1:8000, with interactive documentation available at http://127.0.0.1:8000/docs.

🔌 API Usage
The primary endpoint for predictions is /predict.

URL: /predict

Method: POST

Body (JSON):

{
  "features": [
    -1.0,
    0.2,
    0.3,
    0.1,
    0.8,
    0.4,
    0.3
  ]
}

Example curl request:

curl -X POST "[http://127.0.0.1:8000/predict](http://127.0.0.1:8000/predict)" \
-H "Content-Type: application/json" \
-d '{"features": [-1.0, 0.2, 0.3, 0.1, 0.8, 0.4, 0.3]}'

📁 Project Structure
esp-risk-api/
├── .gitignore
├── dashboard.py                # Streamlit dashboard application
├── final_label_encoder.pkl     # Saved label encoder model
├── final_stacking_model.pkl    # Saved stacking classifier model
├── main.py                     # FastAPI application
├── README.md                   # This file
└── requirements.txt            # Project dependencies
