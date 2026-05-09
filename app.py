"""
app.py
======
Flask backend for Hospital Readmission Prediction.

Run:
    python app.py
    → http://127.0.0.1:5000
"""

import os
import json
import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Load model bundle on startup
# ─────────────────────────────────────────────────────────────────────────────

MODEL_PATH = "readmission_model.joblib"
METRICS_PATH = "metrics.json"

bundle = None
metrics = {}


def load_bundle():
    global bundle, metrics

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at '{MODEL_PATH}'. Run: python train_model.py"
        )

    bundle = joblib.load(MODEL_PATH)

    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            metrics = json.load(f)

    print("[app] Model bundle loaded ✓")


# Load model immediately when app starts
load_bundle()

# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", metrics=metrics)


@app.route("/predict", methods=["POST"])
def predict():
    """Accept JSON or form POST → return prediction JSON."""
    try:

        # Support both JSON API calls and HTML form submissions
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()

        # ─────────────────────────────────────────────────────────────────────
        # Parse Inputs
        # ─────────────────────────────────────────────────────────────────────

        age = float(data["age"])
        gender = str(data["gender"])
        diagnosis_code = str(data["diagnosis_code"])
        length_of_stay = float(data["length_of_stay"])
        num_previous = float(data["num_previous_admissions"])

        num_medications = float(data.get("num_medications", 7))
        num_lab_procedures = float(data.get("num_lab_procedures", 43))
        a1c_result = str(data.get("a1c_result", "None"))

        # ─────────────────────────────────────────────────────────────────────
        # Encode categorical variables
        # ─────────────────────────────────────────────────────────────────────

        enc = bundle["encoders"]

        def safe_encode(encoder, value):
            """Return encoded value; unknown labels → 0."""
            classes = list(encoder.classes_)
            return classes.index(value) if value in classes else 0

        gender_enc = safe_encode(enc["Gender"], gender)
        diag_enc = safe_encode(enc["Diagnosis_Code"], diagnosis_code)
        a1c_enc = safe_encode(enc["A1C_Result"], a1c_result)

        # ─────────────────────────────────────────────────────────────────────
        # Prepare feature array
        # ─────────────────────────────────────────────────────────────────────

        features = np.array([[
            age,
            gender_enc,
            diag_enc,
            length_of_stay,
            num_previous,
            num_medications,
            num_lab_procedures,
            a1c_enc,
        ]])

        # ─────────────────────────────────────────────────────────────────────
        # Scale Features
        # ─────────────────────────────────────────────────────────────────────

        features_scaled = bundle["scaler"].transform(features)

        # ─────────────────────────────────────────────────────────────────────
        # Predict
        # ─────────────────────────────────────────────────────────────────────

        prob = bundle["model"].predict_proba(features_scaled)[0][1]

        label = int(bundle["model"].predict(features_scaled)[0])

        risk_text = "High Risk" if label == 1 else "Low Risk"

        risk_class = "high" if label == 1 else "low"

        # ─────────────────────────────────────────────────────────────────────
        # Return Response
        # ─────────────────────────────────────────────────────────────────────

        return jsonify({
            "success": True,
            "label": label,
            "risk": risk_text,
            "risk_class": risk_class,
            "probability": round(float(prob) * 100, 1),
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400


@app.route("/metrics")
def get_metrics():
    return jsonify(metrics)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
