"""
HeartLens - Flask REST API
Serves precomputed patient data, SHAP values, UMAP embeddings,
and handles LLM narrative generation + what-if predictions.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
import shap
from flask import Flask, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# ── Load precomputed data ─────────────────────────────────────────────────────

print("Loading precomputed data...")
with open(os.path.join(DATA_DIR, "patients.json")) as f:
    PATIENTS = json.load(f)

with open(os.path.join(DATA_DIR, "stats.json")) as f:
    STATS = json.load(f)

with open(os.path.join(DATA_DIR, "feature_names.json")) as f:
    FEATURE_NAMES = json.load(f)

MODEL = joblib.load(os.path.join(DATA_DIR, "xgboost_model.joblib"))
SCALER = joblib.load(os.path.join(DATA_DIR, "scaler.joblib"))
EXPLAINER = shap.TreeExplainer(MODEL)

X_ENCODED = pd.read_csv(os.path.join(DATA_DIR, "X_encoded.csv"))

PATIENTS_BY_ID = {p["id"]: p for p in PATIENTS}

CONTINUOUS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
RAW_FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs",
                "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]

FEATURE_LABELS = {
    "age": "Age", "sex": "Sex", "cp": "Chest Pain Type",
    "trestbps": "Resting Blood Pressure", "chol": "Serum Cholesterol",
    "fbs": "Fasting Blood Sugar", "restecg": "Resting ECG",
    "thalach": "Max Heart Rate", "exang": "Exercise Angina",
    "oldpeak": "ST Depression (Oldpeak)", "slope": "ST Slope",
    "ca": "Major Vessels (Ca)", "thal": "Thalassemia"
}

print(f"Loaded {len(PATIENTS)} patients, model ready.")


# ── Helper: encode raw features into model input ─────────────────────────────

def encode_features(raw_dict):
    """Convert raw feature dict to encoded DataFrame matching training format."""
    df = pd.DataFrame([raw_dict])

    # Ensure correct types
    for col in CONTINUOUS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in CATEGORICAL:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(int)

    onehot_cols = ["cp", "restecg", "slope", "thal"]
    X = pd.get_dummies(df[RAW_FEATURES], columns=onehot_cols, drop_first=False)

    # Align columns with training data
    for col in FEATURE_NAMES:
        if col not in X.columns:
            X[col] = 0
    X = X[FEATURE_NAMES]

    # Scale continuous
    X[CONTINUOUS] = SCALER.transform(X[CONTINUOUS])
    return X


# ── API Routes ────────────────────────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "patients": len(PATIENTS)})


@app.route("/api/stats", methods=["GET"])
def get_stats():
    return jsonify(STATS)


@app.route("/api/patients", methods=["GET"])
def get_patients():
    """Return all patients. Supports ?site= and ?riskMin=&riskMax= filters."""
    result = PATIENTS

    site = request.args.get("site")
    if site:
        result = [p for p in result if p["site"] == site]

    risk_min = request.args.get("riskMin", type=float)
    risk_max = request.args.get("riskMax", type=float)
    if risk_min is not None:
        result = [p for p in result if p["riskProb"] >= risk_min]
    if risk_max is not None:
        result = [p for p in result if p["riskProb"] <= risk_max]

    return jsonify(result)


@app.route("/api/patients/<int:patient_id>", methods=["GET"])
def get_patient(patient_id):
    """Return a single patient by ID."""
    patient = PATIENTS_BY_ID.get(patient_id)
    if not patient:
        return jsonify({"error": "Patient not found"}), 404
    return jsonify(patient)


@app.route("/api/shap/global", methods=["GET"])
def get_global_shap():
    """Return global feature importance (mean |SHAP|) aggregated to original features."""
    importance = {}
    for f in RAW_FEATURES:
        vals = [abs(p["shapValues"][f]) for p in PATIENTS]
        importance[f] = round(sum(vals) / len(vals), 4)

    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    return jsonify([
        {"feature": f, "label": FEATURE_LABELS[f], "meanAbsShap": v}
        for f, v in sorted_importance
    ])


@app.route("/api/shap/patient/<int:patient_id>", methods=["GET"])
def get_patient_shap(patient_id):
    """Return SHAP values for a specific patient."""
    patient = PATIENTS_BY_ID.get(patient_id)
    if not patient:
        return jsonify({"error": "Patient not found"}), 404

    shap_sorted = sorted(
        patient["shapValues"].items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    return jsonify({
        "patientId": patient_id,
        "baseValue": patient["baseValue"],
        "prediction": patient["riskProb"],
        "shapValues": [
            {"feature": f, "label": FEATURE_LABELS[f], "value": v}
            for f, v in shap_sorted
        ]
    })


@app.route("/api/predict", methods=["POST"])
def predict_whatif():
    """What-if prediction: recompute SHAP for modified feature values."""
    data = request.get_json()
    if not data or "features" not in data:
        return jsonify({"error": "Missing 'features' in request body"}), 400

    raw = data["features"]
    try:
        X = encode_features(raw)
        prob = float(MODEL.predict_proba(X)[:, 1][0])
        shap_vals = EXPLAINER.shap_values(X)[0]
        base = float(EXPLAINER.expected_value)

        # Map encoded SHAP back to original features
        shap_original = {}
        for col in RAW_FEATURES:
            matching = [j for j, fn in enumerate(FEATURE_NAMES)
                        if fn == col or fn.startswith(col + "_")]
            shap_original[col] = round(sum(float(shap_vals[j]) for j in matching), 4)

        shap_sorted = sorted(shap_original.items(), key=lambda x: abs(x[1]), reverse=True)

        return jsonify({
            "riskProb": round(prob, 4),
            "baseValue": round(base, 4),
            "shapValues": [
                {"feature": f, "label": FEATURE_LABELS[f], "value": v}
                for f, v in shap_sorted
            ]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/narrative", methods=["POST"])
def generate_narrative():
    """
    Generate LLM clinical narrative for a patient.
    Uses Claude API if key is configured, otherwise returns template narrative.
    """
    data = request.get_json()
    patient_id = data.get("patientId")

    if patient_id is not None:
        patient = PATIENTS_BY_ID.get(patient_id)
        if not patient:
            return jsonify({"error": "Patient not found"}), 404
    else:
        patient = data

    narrative = _build_narrative(patient)
    return jsonify({"narrative": narrative, "source": _get_llm_source()})


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Conversational query endpoint.
    Accepts {message, patientId?, history?[]} and returns LLM response.
    """
    data = request.get_json()
    message = data.get("message", "")
    patient_id = data.get("patientId")
    history = data.get("history", [])

    patient = PATIENTS_BY_ID.get(patient_id) if patient_id is not None else None

    response = _handle_chat(message, patient, history)
    return jsonify({"response": response, "source": _get_llm_source()})


# ── LLM Integration ──────────────────────────────────────────────────────────

def _get_llm_source():
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if api_key and api_key != "your_key_here":
        return "claude"
    return "template"


def _build_narrative(patient):
    """Build narrative using Claude API or fallback template."""
    api_key = os.getenv("ANTHROPIC_API_KEY", "")

    if api_key and api_key != "your_key_here":
        return _claude_narrative(patient, api_key)
    return _template_narrative(patient)


def _claude_narrative(patient, api_key):
    """Generate narrative via Claude API."""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        features = patient["features"]
        shap_vals = patient["shapValues"]
        risk = patient["riskProb"]

        top_shap = sorted(shap_vals.items(), key=lambda x: abs(x[1]), reverse=True)[:5]

        prompt = f"""You are a clinical AI assistant explaining heart disease risk predictions to clinicians.

A patient has been assessed by an XGBoost classifier trained on the UCI Heart Disease dataset.
Generate a clear, plain-English clinical narrative explaining the prediction.

Patient Profile:
- Age: {features['age']}, Sex: {'Male' if features['sex'] == 1 else 'Female'}
- Chest Pain Type: {int(features['cp'])} (0=typical angina, 1=atypical, 2=non-anginal, 3=asymptomatic)
- Resting BP: {features['trestbps']} mmHg, Cholesterol: {features['chol']} mg/dl
- Fasting Blood Sugar >120: {'Yes' if features['fbs'] == 1 else 'No'}
- Resting ECG: {int(features['restecg'])} (0=normal, 1=ST-T abnormality, 2=LV hypertrophy)
- Max Heart Rate: {features['thalach']} bpm
- Exercise Angina: {'Yes' if features['exang'] == 1 else 'No'}
- ST Depression (oldpeak): {features['oldpeak']}
- ST Slope: {int(features['slope'])} (0=upsloping, 1=flat, 2=downsloping)
- Major Vessels (Ca): {int(features['ca'])}
- Thalassemia: {int(features['thal'])} (1=normal, 2=fixed defect, 3=reversible defect)

Predicted Risk Probability: {risk:.2f} ({'High' if risk >= 0.7 else 'Moderate' if risk >= 0.4 else 'Low'} Risk)
Base Value: {patient['baseValue']:.4f}

Top SHAP Feature Contributions:
{chr(10).join(f'  - {FEATURE_LABELS[f]}: {v:+.4f}' for f, v in top_shap)}

Guidelines:
- Write 3-4 sentences in plain clinical English
- Reference specific feature values and their SHAP contributions
- Mention both risk drivers and protective factors
- Do NOT speculate beyond what the data shows
- Flag if any data seems unusual"""

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text

    except Exception as e:
        print(f"Claude API error: {e}")
        return _template_narrative(patient)


def _template_narrative(patient):
    """Fallback template-based narrative when no LLM API is available."""
    f = patient["features"]
    s = patient["shapValues"]
    risk = patient["riskProb"]

    risk_label = "high" if risk >= 0.7 else "moderate" if risk >= 0.4 else "low"
    sex_text = "male" if f["sex"] == 1 else "female"

    top_shap = sorted(s.items(), key=lambda x: abs(x[1]), reverse=True)[:4]
    drivers = [(k, v) for k, v in top_shap if v > 0]
    protectors = [(k, v) for k, v in top_shap if v < 0]

    narrative = (
        f"This {int(f['age'])}-year-old {sex_text} patient was classified as "
        f"{risk_label}-risk (predicted probability: {risk:.2f})."
    )

    if drivers:
        driver_text = ", ".join(
            f"{FEATURE_LABELS[k]} (contributing {v:+.3f} to risk)" for k, v in drivers
        )
        narrative += f" The primary risk drivers were {driver_text}."

    if protectors:
        protector_text = ", ".join(
            f"{FEATURE_LABELS[k]} ({v:+.3f})" for k, v in protectors
        )
        narrative += f" Protective factors included {protector_text}."

    narrative += (
        f" The patient's resting blood pressure was {int(f['trestbps'])} mmHg, "
        f"serum cholesterol {int(f['chol'])} mg/dl, "
        f"and maximum heart rate achieved was {int(f['thalach'])} bpm."
    )

    return narrative


def _handle_chat(message, patient, history):
    """Handle chat query with Claude API or fallback."""
    api_key = os.getenv("ANTHROPIC_API_KEY", "")

    if api_key and api_key != "your_key_here":
        return _claude_chat(message, patient, history, api_key)
    return _template_chat(message, patient)


def _claude_chat(message, patient, history, api_key):
    """Chat via Claude API with full context."""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        system_prompt = """You are HeartLens, a clinical AI assistant for a heart disease visual analytics dashboard.
You help clinicians and researchers understand heart disease predictions from an XGBoost model trained on the UCI Heart Disease dataset (920 patients from 4 hospital sites).

You have access to SHAP-based feature attributions for each patient. Answer questions clearly and concisely.
Reference specific data values when available. Do not speculate beyond the data. Keep responses under 150 words."""

        if patient:
            system_prompt += f"""

Currently selected patient #{patient['id']}:
- Site: {patient['site']}
- Risk: {patient['riskProb']:.2f}
- Features: {json.dumps(patient['features'])}
- Top SHAP: {json.dumps(dict(sorted(patient['shapValues'].items(), key=lambda x: abs(x[1]), reverse=True)[:5]))}"""

        # Build dataset summary
        system_prompt += f"""

Dataset summary: {len(PATIENTS)} patients across Cleveland ({sum(1 for p in PATIENTS if p['site']=='Cleveland')}), Hungary ({sum(1 for p in PATIENTS if p['site']=='Hungary')}), Switzerland ({sum(1 for p in PATIENTS if p['site']=='Switzerland')}), VA Long Beach ({sum(1 for p in PATIENTS if p['site']=='VA Long Beach')}).
High risk (>0.7): {sum(1 for p in PATIENTS if p['riskProb']>=0.7)}, Moderate (0.4-0.7): {sum(1 for p in PATIENTS if 0.4<=p['riskProb']<0.7)}, Low (<0.4): {sum(1 for p in PATIENTS if p['riskProb']<0.4)}."""

        messages = []
        for h in history[-6:]:  # Keep last 6 messages for context
            messages.append({"role": h["role"], "content": h["text"]})
        messages.append({"role": "user", "content": message})

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=250,
            system=system_prompt,
            messages=messages,
        )
        return response.content[0].text

    except Exception as e:
        print(f"Claude chat error: {e}")
        return _template_chat(message, patient)


def _template_chat(message, patient):
    """Fallback template-based chat responses."""
    q = message.lower()

    if patient and ("why" in q or "explain" in q or "risk" in q):
        top = sorted(patient["shapValues"].items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        drivers = ", ".join(f"**{FEATURE_LABELS[f]}** (SHAP: {v:+.4f})" for f, v in top)
        risk_pct = patient["riskProb"] * 100
        return (
            f"Patient #{patient['id']} has a predicted risk of **{risk_pct:.0f}%**. "
            f"The top contributing features are: {drivers}."
        )

    if "highest" in q or "most at risk" in q:
        top3 = sorted(PATIENTS, key=lambda p: p["riskProb"], reverse=True)[:3]
        lines = [f"{i+1}. **Patient #{p['id']}** ({p['site']}) — {p['riskProb']*100:.0f}% risk"
                 for i, p in enumerate(top3)]
        return "The highest-risk patients are:\n" + "\n".join(lines)

    if "how many" in q or "count" in q:
        high = sum(1 for p in PATIENTS if p["riskProb"] >= 0.7)
        mod = sum(1 for p in PATIENTS if 0.4 <= p["riskProb"] < 0.7)
        low = sum(1 for p in PATIENTS if p["riskProb"] < 0.4)
        return (
            f"Across all **{len(PATIENTS)} patients**:\n"
            f"- **High risk** (>70%): {high}\n"
            f"- **Moderate** (40-70%): {mod}\n"
            f"- **Low risk** (<40%): {low}"
        )

    if "compare" in q or "site" in q or "hospital" in q:
        sites = {}
        for p in PATIENTS:
            sites.setdefault(p["site"], []).append(p["riskProb"])
        lines = [f"- **{s}**: {len(r)} patients, avg risk {sum(r)/len(r)*100:.1f}%"
                 for s, r in sites.items()]
        return "Breakdown by hospital site:\n" + "\n".join(lines)

    if "feature" in q or "important" in q or "shap" in q:
        importance = {}
        for f in RAW_FEATURES:
            importance[f] = sum(abs(p["shapValues"][f]) for p in PATIENTS) / len(PATIENTS)
        top5 = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
        lines = [f"{i+1}. **{FEATURE_LABELS[f]}** (mean |SHAP|: {v:.4f})"
                 for i, (f, v) in enumerate(top5)]
        return "Most impactful features:\n" + "\n".join(lines)

    ctx = f" Currently viewing **Patient #{patient['id']}**." if patient else ""
    return (
        "I can help explore the heart disease prediction data. Try:\n"
        "- \"Why is this patient high risk?\"\n"
        "- \"Which patients are most at risk?\"\n"
        "- \"Compare hospital sites\"\n"
        "- \"What are the most important features?\"\n"
        "- \"How many patients are high risk?\"" + ctx
    )


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.getenv("FLASK_PORT", 5001))
    print(f"\nStarting HeartLens API on port {port}...")
    app.run(debug=True, port=port, host="0.0.0.0")
