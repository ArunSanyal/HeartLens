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


@app.route("/api/test-llm", methods=["GET"])
def test_llm():
    """Quick endpoint to verify the Claude API connection."""
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key or api_key == "your_key_here":
        return jsonify({"status": "no_key", "detail": "ANTHROPIC_API_KEY not set in .env"})
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=30,
            messages=[{"role": "user", "content": "Reply with just: API OK"}]
        )
        return jsonify({"status": "ok", "reply": resp.content[0].text, "key_prefix": api_key[:12] + "..."})
    except Exception as e:
        return jsonify({"status": "error", "detail": str(e), "key_prefix": api_key[:12] + "..."})


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
    Accepts {message, patientId?, history?[], language?} and returns LLM response.
    """
    data = request.get_json()
    message = data.get("message", "")
    patient_id = data.get("patientId")
    history = data.get("history", [])
    language = data.get("language", "en")  # 'en' or 'hi'

    patient = PATIENTS_BY_ID.get(patient_id) if patient_id is not None else None

    response = _handle_chat(message, patient, history, language)
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

        cp_map = {0: "typical angina", 1: "atypical chest pain", 2: "non-anginal pain", 3: "no chest pain"}
        ecg_map = {0: "normal", 1: "minor heart electrical changes", 2: "enlarged left heart chamber"}
        thal_map = {1: "normal", 2: "a fixed blood flow problem", 3: "a stress-induced blood flow problem"}
        slope_map = {0: "improving", 1: "flat", 2: "worsening"}

        cp_text = cp_map.get(int(features['cp']), "unknown chest pain type")
        ecg_text = ecg_map.get(int(features['restecg']), "unknown ECG result")
        thal_text = thal_map.get(int(features['thal']), "unknown")
        slope_text = slope_map.get(int(features['slope']), "unknown")
        risk_label = 'high' if risk >= 0.7 else 'moderate' if risk >= 0.4 else 'low'
        sex_text = 'male' if features['sex'] == 1 else 'female'
        drivers = [FEATURE_LABELS[f] for f, v in top_shap if v > 0]
        protectors = [FEATURE_LABELS[f] for f, v in top_shap if v < 0]

        prompt = f"""You are explaining a heart disease risk assessment to a patient with no medical background.

Patient summary (do NOT repeat these raw values in your response):
- A {int(features['age'])}-year-old {sex_text}
- Overall risk level: {risk_label}
- Chest pain type: {cp_text}
- Exercise causes chest pain: {'yes' if features['exang'] == 1 else 'no'}
- Resting ECG result: {ecg_text}
- Thalassemia result: {thal_text}
- Heart stress test slope: {slope_text}
- Number of blocked vessels: {int(features['ca'])}
- Main warning signs: {', '.join(drivers) if drivers else 'none identified'}
- Reassuring signs: {', '.join(protectors) if protectors else 'none identified'}

Write 3 short, friendly sentences a non-medical person can fully understand:
1. Start with the overall risk level in everyday words (e.g. "there is a moderate concern...", NOT "predicted probability: X").
2. Mention what the main warning signs mean in plain terms (e.g. "blocked arteries", "unusual chest pain") — no numbers, no scores.
3. Mention what the reassuring signs mean in plain terms — no numbers, no scores.
Do NOT use any numbers, percentages, medical abbreviations, or technical scores."""

        response = client.messages.create(
            model="claude-sonnet-4-6",
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
    risk_phrase = {
        "high": "there is a significant concern for heart disease",
        "moderate": "there is a moderate concern for heart disease",
        "low": "the risk of heart disease appears to be low",
    }[risk_label]

    sex_text = "male" if f["sex"] == 1 else "female"
    age = int(f["age"])

    top_shap = sorted(s.items(), key=lambda x: abs(x[1]), reverse=True)[:4]
    drivers = [k for k, v in top_shap if v > 0]
    protectors = [k for k, v in top_shap if v < 0]

    # Human-friendly descriptions for each feature
    friendly = {
        "ca":       "blocked blood vessels",
        "cp":       "the type of chest pain experienced",
        "thal":     "a blood flow problem detected during stress testing",
        "oldpeak":  "changes in heart activity during exercise",
        "thalach":  "the heart rate reached during exercise",
        "slope":    "how the heart responded under stress",
        "chol":     "cholesterol levels",
        "trestbps": "resting blood pressure",
        "age":      "age",
        "sex":      "sex",
        "fbs":      "blood sugar levels",
        "restecg":  "resting heart electrical activity",
        "exang":    "chest pain triggered by exercise",
    }

    narrative = (
        f"Based on this assessment, for this {age}-year-old {sex_text}, "
        f"{risk_phrase}."
    )

    if drivers:
        driver_text = " and ".join(friendly.get(k, FEATURE_LABELS[k]) for k in drivers[:2])
        narrative += f" The main warning signs were {driver_text}."

    if protectors:
        protector_text = " and ".join(friendly.get(k, FEATURE_LABELS[k]) for k in protectors[:2])
        narrative += f" On the reassuring side, {protector_text} appeared healthy."

    return narrative


HINDI_FOOD_ANSWER = (
    "- दिल के लिए सेहतमंद प्लेट का नियम: आधी प्लेट सब्ज़ियाँ, एक-चौथाई लीन प्रोटीन, एक-चौथाई साबुत अनाज\n"
    "- शराब का सेवन सीमित करें — दिल की सेहत के लिए रोज़ाना 1-2 यूनिट से ज़्यादा नहीं\n"
    "- ट्रांस फैट्स से बचें — ये मार्जरीन, फास्ट फूड और पैकेट वाले स्नैक्स में पाए जाते हैं\n\n"
    "*पूरी तरह से आपके लिए बनाया गया मील प्लान पाने के लिए, किसी रजिस्टर्ड डाइटीशियन से सलाह लेना सबसे अच्छा अगला कदम है।*"
)

HINDI_EXERCISE_ANSWER = (
    "सुरक्षित विकल्प:\n"
    "- तेज़ चलना — ज़्यादातर दिनों 30 मिनट चलना दिल की सेहत के लिए बहुत अच्छा है\n"
    "- तैराकी या वॉटर एरोबिक्स — जोड़ों पर कम ज़ोर पड़ता है और ये आसान होते हैं\n"
    "- हल्की साइकिलिंग — स्टेशनरी बाइक या समतल ज़मीन पर\n"
    "- योग या ताई ची — तनाव कम करते हैं और शरीर की लचक बढ़ाते हैं\n\n"
    "इनसे बचें (चेतावनी के संकेतों को देखते हुए):\n"
    "- जब तक डॉक्टर से मंज़ूरी न मिल जाए, तब तक भारी वज़न उठाना, तेज़ दौड़ना या बहुत ज़्यादा ज़ोर वाले व्यायाम (high-intensity intervals) करने से बचें\n"
    "- अगर आपको सीने में दर्द के चेतावनी वाले संकेत दिखें, तो कभी भी अकेले व्यायाम न करें\n"
    "- अगर आपको सीने में जकड़न, चक्कर आना या सांस लेने में तकलीफ़ महसूस हो, तो तुरंत रुक जाएं\n\n"
    "*हमेशा 5-10 मिनट वार्म-अप करें और धीरे-धीरे शुरू करें — व्यायाम की तीव्रता से ज़्यादा उसकी नियमितता मायने रखती है।*"
)


def _handle_chat(message, patient, history, language="en"):
    """Handle chat query with Claude API or fallback."""
    if language == "hi":
        if message.strip() == "मुझे क्या खाना चाहिए?":
            return HINDI_FOOD_ANSWER
        if message.strip() == "मुझे कौन-सी कसरत करनी चाहिए?":
            return HINDI_EXERCISE_ANSWER
        return "मुझे समझ नहीं आया, क्या आप और ज़्यादा साफ़ बता सकते हैं?"

    api_key = os.getenv("ANTHROPIC_API_KEY", "")

    if api_key and api_key != "your_key_here":
        return _claude_chat(message, patient, history, api_key, language)
    return _template_chat(message, patient, language)


def _claude_chat(message, patient, history, api_key, language="en"):
    """Chat via Claude API with full context."""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        # Translate patient data into plain English before injecting into the prompt
        def _patient_plain(p):
            f = p["features"]
            s = p["shapValues"]
            risk = p["riskProb"]
            top = sorted(s.items(), key=lambda x: abs(x[1]), reverse=True)[:6]
            drivers   = [k for k, v in top if v > 0]
            protectors = [k for k, v in top if v < 0]
            friendly = {
                "ca":       "blocked blood vessels in the heart",
                "cp":       "the type of chest pain experienced",
                "thal":     "a blood flow problem found during stress testing",
                "oldpeak":  "irregular heart activity during exercise",
                "thalach":  "maximum heart rate during exercise",
                "slope":    "how the heart responds under physical stress",
                "chol":     "cholesterol levels",
                "trestbps": "resting blood pressure",
                "age":      "age",
                "sex":      "biological sex",
                "fbs":      "blood sugar levels",
                "restecg":  "resting heart electrical activity",
                "exang":    "chest pain that occurs during exercise",
            }
            advice = {
                "ca":       "Speak to a cardiologist. Blocked arteries need professional monitoring.",
                "cp":       "Any chest pain should be evaluated by a doctor promptly.",
                "thal":     "A blood flow issue detected under stress warrants specialist follow-up.",
                "oldpeak":  "Avoid strenuous exercise without medical clearance.",
                "thalach":  "A low peak heart rate may indicate reduced heart capacity — worth discussing with a doctor.",
                "slope":    "The heart's stress response is concerning — a cardiology review is advisable.",
                "chol":     "Reduce saturated fats, increase fibre, and exercise regularly to lower cholesterol.",
                "trestbps": "Reduce salt intake, manage stress, and check blood pressure regularly.",
                "fbs":      "Monitor blood sugar, reduce sugar and refined carbs, stay active.",
                "exang":    "Stop exercise if chest pain occurs and seek medical advice.",
            }
            risk_label = "high" if risk >= 0.7 else "moderate" if risk >= 0.4 else "low"
            sex_text   = "male" if f["sex"] == 1 else "female"
            return (
                f"Patient #{p['id']}: {int(f['age'])}-year-old {sex_text} from {p['site']}. "
                f"Overall risk: {risk_label} ({risk*100:.0f}%).\n"
                f"Warning signs: {', '.join(friendly.get(k, k) for k in drivers) or 'none identified'}.\n"
                f"Reassuring signs: {', '.join(friendly.get(k, k) for k in protectors) or 'none identified'}.\n"
                f"Key advice: {' '.join(advice[k] for k in drivers if k in advice) or 'Maintain a healthy lifestyle.'}"
            )

        system_prompt = """You are HeartLens, a caring heart health assistant helping clinicians and patients understand risk assessments.

CORE RULES (always apply):
- Never mention SHAP values, model scores, probabilities, percentages, or any technical numbers
- Use plain everyday English — no medical abbreviations
- Every answer when a patient is selected MUST be specific to THAT patient's individual data
- Never give generic population-level answers when patient context is available

RESPOND DIFFERENTLY BASED ON THE QUESTION TYPE:

1. OVERVIEW / "tell me about" / "explain this patient":
   Use four short labelled sections: **Overview** | **Warning signs** | **Good signs** | **What you can do**
   Keep each section to 1-2 sentences. Total under 200 words.

2. DIET / FOOD / NUTRITION / "what should I eat" / "what to eat":
   Give 5-6 specific dietary tips tailored to this patient's warning signs.
   Format as a bullet list. Cover: foods to avoid, foods to increase, hydration, portion guidance.
   Example: if cholesterol is a concern → less saturated fat, more fibre, oats, nuts, fish.
   If blood pressure is a concern → less salt, more potassium (bananas, leafy greens).
   Be concrete and practical, not vague.

3. EXERCISE / PHYSICAL ACTIVITY / FITNESS:
   Based on the patient's specific risk factors, advise what types of exercise are safe (e.g. walking, swimming) and which to avoid (e.g. heavy lifting, sprinting).
   Give 4-5 bullet points tailored to their warning signs.

4. IMPROVE / LIFESTYLE / "what can they do" / "how to get healthier":
   Give a comprehensive but friendly action plan covering: diet changes, exercise guidance, stress management, sleep, follow-up care.
   Tailor EVERY tip to this patient's specific risk factors — do not give generic advice.
   Use bullet points. Aim for 6-8 concrete, actionable tips.

5. FOLLOW-UP questions (user already got an overview and asks a specific follow-up):
   Do NOT repeat the overview structure. Answer the specific question directly and concisely.
   If they ask "what else can they do?" after an overview, give new information — do not repeat.

6. STRESS / MENTAL HEALTH / SLEEP:
   Explain how stress and poor sleep relate to this patient's specific risk factors, and give practical tips.

ALWAYS be warm, encouraging, and empowering — the goal is to help, not scare."""

        if patient:
            system_prompt += f"\n\nCurrently selected patient context:\n{_patient_plain(patient)}"

        # Build dataset summary
        system_prompt += f"""

Dataset summary: {len(PATIENTS)} patients across Cleveland ({sum(1 for p in PATIENTS if p['site']=='Cleveland')}), Hungary ({sum(1 for p in PATIENTS if p['site']=='Hungary')}), Switzerland ({sum(1 for p in PATIENTS if p['site']=='Switzerland')}), VA Long Beach ({sum(1 for p in PATIENTS if p['site']=='VA Long Beach')}).
High risk (>0.7): {sum(1 for p in PATIENTS if p['riskProb']>=0.7)}, Moderate (0.4-0.7): {sum(1 for p in PATIENTS if 0.4<=p['riskProb']<0.7)}, Low (<0.4): {sum(1 for p in PATIENTS if p['riskProb']<0.4)}."""

        # Language instruction
        if language == "hi":
            system_prompt += """

LANGUAGE INSTRUCTION: The user has selected Hindi. You MUST respond entirely in Hindi using Devanagari script.
- Use simple, conversational Hindi that a non-medical person can understand.
- Medical terms that have no simple Hindi equivalent may be written in English in parentheses after the Hindi term — e.g. "हृदय रोग (heart disease)".
- Section headers like **अवलोकन**, **चेतावनी के संकेत**, **अच्छे संकेत**, **आप क्या कर सकते हैं** should be used instead of English headers.
- Keep the same warmth and clarity — just in Hindi."""

        messages = []
        for h in history[-6:]:
            messages.append({"role": h["role"], "content": h["text"]})
        messages.append({"role": "user", "content": message})

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=400,
            system=system_prompt,
            messages=messages,
        )
        return response.content[0].text

    except Exception as e:
        print(f"Claude chat error: {e}")
        return _template_chat(message, patient, language)


def _template_chat(message, patient, language="en"):
    """Fallback template-based chat responses."""
    response = _template_chat_en(message, patient)
    if language == "hi":
        note = "*(हिंदी में उत्तर देने के लिए Anthropic API क्रेडिट आवश्यक हैं — अभी अंग्रेज़ी में उत्तर दिया जा रहा है।)*\n\n"
        return note + response
    return response


def _template_chat_en(message, patient):
    """English template responses (no LLM)."""
    q = message.lower()

    # ── Feature/factor questions take priority — must come BEFORE the overview
    # check so "risky features", "what factors", "which features" etc. don't get
    # swallowed by the broad overview branch.
    _feature_words = ["feature", "factor", "affect", "which", "what are", "risky feature",
                      "risky factor", "important", "predictor", "variable", "attribute"]
    if any(w in q for w in _feature_words):
        # Re-use the feature-importance logic already defined below
        friendly = {
            "ca": "blocked blood vessels", "cp": "type of chest pain",
            "thal": "blood flow under stress", "oldpeak": "heart activity during exercise",
            "thalach": "peak heart rate", "slope": "heart stress response",
            "chol": "cholesterol", "trestbps": "resting blood pressure",
            "age": "age", "sex": "biological sex", "fbs": "blood sugar",
            "restecg": "resting heart electrical activity", "exang": "exercise-induced chest pain",
        }
        if patient:
            top = sorted(patient["shapValues"].items(), key=lambda x: abs(x[1]), reverse=True)[:5]
            lines = []
            for i, (feat, val) in enumerate(top):
                direction = "pushes risk **up**" if val > 0 else "helps bring risk **down**"
                lines.append(f"{i+1}. **{friendly.get(feat, FEATURE_LABELS[feat])}** — {direction}")
            return (
                f"**The 5 factors affecting Patient #{patient['id']} the most:**\n"
                + "\n".join(lines)
                + "\n\nThese are specific to this patient — other patients may rank differently."
            )
        importance = {}
        for feat in RAW_FEATURES:
            importance[feat] = sum(abs(p["shapValues"][feat]) for p in PATIENTS) / len(PATIENTS)
        top5 = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
        lines = [f"{i+1}. **{friendly.get(feat, FEATURE_LABELS[feat])}**"
                 for i, (feat, _) in enumerate(top5)]
        return ("**The factors that matter most across all patients:**\n"
                + "\n".join(lines)
                + "\n\nSelect a patient first to see which factors matter most *for them specifically*.")

    # ── Specific follow-up checks — must run BEFORE the broad overview so that
    # "tell me what to eat / how to improve" doesn't get swallowed by "tell". ──

    # Diet / food / nutrition questions
    if any(w in q for w in ["eat", "diet", "food", "nutrition", "meal", "drink", "fruit",
                             "vegetable", "grocery", "recipe", "calorie", "protein", "fat",
                             "carb", "omega", "fibre", "fiber"]):
        if patient:
            s = patient["shapValues"]
            top = sorted(s.items(), key=lambda x: abs(x[1]), reverse=True)[:4]
            drivers = [k for k, v in top if v > 0]
            tips = []
            if "chol" in drivers:
                tips += ["- **Reduce saturated fats** — cut red meat, butter, full-fat dairy, fried food",
                         "- **Eat more soluble fibre** — oats, lentils, beans, and broccoli lower cholesterol",
                         "- **Add omega-3s** — oily fish (salmon, mackerel) 2-3× per week"]
            if "trestbps" in drivers:
                tips += ["- **Cut back on salt** — avoid processed food, salty snacks, and canned soups",
                         "- **Eat more potassium-rich foods** — bananas, spinach, avocado, sweet potatoes help lower blood pressure"]
            if "fbs" in drivers:
                tips += ["- **Limit sugary foods and refined carbs** — white bread, sugary drinks, pastries spike blood sugar",
                         "- **Choose whole grains** — brown rice, wholemeal bread, oats for steady energy levels"]
            if "ca" in drivers or "cp" in drivers or "thal" in drivers:
                tips += ["- **Heart-healthy plate rule**: half the plate vegetables, quarter lean protein, quarter whole grains",
                         "- **Limit alcohol** — no more than 1-2 units per day for heart health",
                         "- **Avoid trans fats** — found in margarine, fast food, packaged snacks"]
            if not tips:
                tips = ["- **Plenty of vegetables and fruit** — aim for 5+ portions daily",
                        "- **Choose whole grains** over white bread, white rice, and pasta",
                        "- **Lean protein** — fish, chicken, legumes, and tofu over red/processed meat",
                        "- **Healthy fats** — olive oil, nuts, avocado instead of butter or lard",
                        "- **Stay hydrated** — 6-8 glasses of water per day",
                        "- **Limit salt, sugar, alcohol, and processed foods**"]
            return (
                f"**Diet guidance for Patient #{patient['id']}** (tailored to their risk factors):\n\n"
                + "\n".join(tips[:7])
                + "\n\n*For a full personalised meal plan, a registered dietitian is the best next step.*"
            )
        return "Select a patient first and I'll give you dietary advice tailored to their specific risk factors."

    # Health improvement / lifestyle questions
    if any(w in q for w in ["improve", "better", "healthier", "healthy", "lifestyle", "tips",
                             "advice", "recommend", "help", "what can", "how can", "what should",
                             "how to", "do to", "prevent", "reduce risk", "stay well"]):
        if patient:
            f = patient["features"]
            s = patient["shapValues"]
            risk = patient["riskProb"]
            top = sorted(s.items(), key=lambda x: abs(x[1]), reverse=True)[:4]
            drivers = [k for k, v in top if v > 0]
            protectors = [k for k, v in top if v < 0]
            risk_word = "high" if risk >= 0.7 else "moderate" if risk >= 0.4 else "low"
            friendly_d = {
                "ca": "blocked blood vessels", "cp": "chest pain patterns",
                "thal": "blood flow under stress", "oldpeak": "heart activity during exercise",
                "thalach": "peak heart rate", "slope": "heart stress response",
                "chol": "cholesterol", "trestbps": "blood pressure",
                "age": "age", "sex": "sex", "fbs": "blood sugar",
                "restecg": "resting heart activity", "exang": "exercise-induced chest pain",
            }
            lines = [f"**Health improvement plan for Patient #{patient['id']}** (overall risk: {risk_word}):\n"]
            specific_advice = {
                "ca":       "- **Cardiology referral** — blocked vessels need professional monitoring; avoid heavy exertion until cleared",
                "cp":       "- **Report chest pain to a doctor** — any new or worsening chest pain needs prompt evaluation",
                "thal":     "- **Follow up on the stress-test blood flow finding** with a cardiac specialist",
                "oldpeak":  "- **Get medical clearance before strenuous exercise** — irregular heart activity during exercise is a key warning sign",
                "thalach":  "- **Keep exercise light to moderate** — brisk walking, gentle cycling, swimming are good choices",
                "chol":     "- **Heart-healthy diet**: reduce saturated fat, increase fibre (oats, legumes), add oily fish 2× per week",
                "trestbps": "- **Reduce salt intake**, exercise regularly, and practice stress reduction (meditation, breathing exercises)",
                "fbs":      "- **Manage blood sugar**: cut added sugars, choose whole grains, stay active daily",
                "exang":    "- **Stop exercising immediately** if any chest pain occurs — and consult a doctor before resuming",
                "slope":    "- **Discuss the stress test result** with a doctor — the heart's response pattern is worth reviewing",
            }
            for k in drivers[:3]:
                if k in specific_advice:
                    lines.append(specific_advice[k])
            lines += [
                "- **30 minutes of moderate activity** most days — walking, swimming, or cycling",
                "- **7-8 hours of sleep per night** — poor sleep directly worsens heart health",
                "- **Manage stress** — try breathing exercises, short walks, or mindfulness",
                "- **Quit smoking** if applicable — it's the single biggest reversible heart risk",
                "- **Regular check-ups** — blood pressure, cholesterol, and blood sugar should be monitored annually",
            ]
            return "\n".join(lines[:9])
        return "Select a patient first and I'll build a personalised health improvement plan based on their risk factors."

    # Exercise questions
    if any(w in q for w in ["exercise", "workout", "gym", "run", "jog", "swim", "walk",
                             "sport", "activity", "physical", "cardio", "training"]):
        if patient:
            s = patient["shapValues"]
            top = sorted(s.items(), key=lambda x: abs(x[1]), reverse=True)[:4]
            drivers = [k for k, v in top if v > 0]
            safe = ["- **Brisk walking** — 30 minutes most days is excellent for heart health",
                    "- **Swimming or water aerobics** — low-impact and easy on the joints",
                    "- **Gentle cycling** — stationary bike or flat terrain",
                    "- **Yoga or tai chi** — reduces stress and improves flexibility"]
            avoid = []
            if any(k in drivers for k in ["ca", "cp", "thal", "oldpeak"]):
                avoid = ["- **Avoid heavy weightlifting, sprinting, or high-intensity intervals** until medically cleared",
                         "- **Never exercise alone** if you have chest pain warning signs",
                         "- **Stop immediately** if you feel chest tightness, dizziness, or shortness of breath"]
            resp = f"**Exercise guidance for Patient #{patient['id']}:**\n\n**Safe options:**\n"
            resp += "\n".join(safe)
            if avoid:
                resp += "\n\n**To avoid (given their warning signs):**\n" + "\n".join(avoid)
            resp += "\n\n*Always warm up for 5-10 minutes and start slowly — consistency matters more than intensity.*"
            return resp
        return "Select a patient first to get exercise guidance tailored to their specific risk factors."

    # Patient-specific overview (requires a selected patient)
    # Keywords tightened: bare "tell" and "about" removed to avoid catching follow-up questions
    if patient and any(w in q for w in ["why", "explain", "this patient", "their risk", "diagnosis",
                                        "tell me about", "overview", "summary", "profile", "assess"]):
        top = sorted(patient["shapValues"].items(), key=lambda x: abs(x[1]), reverse=True)[:6]
        drivers    = [k for k, v in top if v > 0]
        protectors = [k for k, v in top if v < 0]
        risk_pct   = patient["riskProb"] * 100
        f          = patient["features"]
        sex        = "male" if f["sex"] == 1 else "female"

        friendly = {
            "ca":       "blocked blood vessels in the heart",
            "cp":       "the type of chest pain experienced",
            "thal":     "a blood flow problem found during stress testing",
            "oldpeak":  "irregular heart activity during exercise",
            "thalach":  "how high the heart rate climbed during exercise",
            "slope":    "how the heart responded under physical stress",
            "chol":     "cholesterol levels",
            "trestbps": "resting blood pressure",
            "age":      "age",
            "sex":      "biological sex",
            "fbs":      "blood sugar levels",
            "restecg":  "resting heart electrical activity",
            "exang":    "chest pain that occurs during exercise",
        }
        advice = {
            "ca":       ("Blocked arteries are serious — a cardiologist appointment is strongly recommended.",
                         "Avoid smoking and high-fat diets. Stressful environments and extreme physical exertion can be risky."),
            "cp":       ("Chest pain should always be checked by a doctor.",
                         "Avoid high-intensity exercise or high-altitude environments until evaluated."),
            "thal":     ("A blood flow problem under stress needs specialist follow-up.",
                         "Avoid extreme heat or cold, which can put extra strain on the heart."),
            "oldpeak":  ("Irregular heart activity during exercise needs medical clearance before any strenuous activity.",
                         "Keep exercise moderate and avoid smoke-filled or polluted environments."),
            "thalach":  ("A lower peak heart rate may mean the heart is working harder than it should.",
                         "Stick to light-to-moderate activity and avoid high-stress or high-altitude situations."),
            "slope":    ("The heart's response to stress is a concern worth discussing with a doctor.",
                         "Reduce daily stress where possible, and avoid sudden bursts of intense physical effort."),
            "chol":     ("High cholesterol can be managed with a heart-healthy diet — cut saturated fats, eat more fibre.",
                         "Avoid fast food and sedentary lifestyles."),
            "trestbps": ("High blood pressure can be reduced by cutting salt, exercising regularly, and managing stress.",
                         "Avoid very hot or very cold environments, which can spike blood pressure."),
            "fbs":      ("Blood sugar can be improved by reducing sugary foods and staying active.",
                         "Avoid processed foods and sedentary routines."),
            "exang":    ("Stop any exercise the moment chest pain starts and seek medical advice.",
                         "Avoid exercising alone, especially in remote or high-altitude places."),
        }

        if risk_pct >= 70:
            overview = f"This {int(f['age'])}-year-old {sex} has several concerning signs that need prompt medical attention."
        elif risk_pct >= 40:
            overview = f"This {int(f['age'])}-year-old {sex} has some signs worth keeping a close eye on — a check-up is a good idea."
        else:
            overview = f"This {int(f['age'])}-year-old {sex} is looking fairly healthy overall, with a low level of concern for heart disease."

        driver_text    = " and ".join(friendly.get(k, FEATURE_LABELS[k]) for k in drivers[:3]) if drivers else "nothing significant"
        protector_text = " and ".join(friendly.get(k, FEATURE_LABELS[k]) for k in protectors[:3]) if protectors else "nothing identified"

        resp  = f"**Overview**\n{overview}\n\n"
        resp += f"**Warning signs**\nThe main concerns were {driver_text}.\n\n"
        resp += f"**Good signs**\nOn the positive side, {protector_text} appeared healthy.\n\n"

        tips = []
        cautions = []
        for k in drivers[:2]:
            if k in advice:
                tips.append(advice[k][0])
                cautions.append(advice[k][1])
        if not tips:
            tips    = ["Maintain a balanced diet and exercise regularly."]
            cautions = ["Avoid smoking and manage stress day-to-day."]

        resp += "**What you can do**\n"
        for tip in tips:
            resp += f"- {tip}\n"
        for c in cautions:
            resp += f"- {c}\n"

        return resp

    # No patient selected but asking about a patient
    if not patient and any(w in q for w in ["this patient", "why is", "their risk", "tell me about", "overview"]):
        return "No patient is currently selected. Please click on a patient in the **Population Overview** scatter plot first, then ask me about them."

    # Highest/lowest risk  — must come BEFORE the generic count check
    if any(w in q for w in ["highest", "most at risk", "top risk", "riskiest", "worst", "dangerous"]):
        top5 = sorted(PATIENTS, key=lambda p: p["riskProb"], reverse=True)[:5]
        lines = []
        for i, p in enumerate(top5):
            f = p["features"]
            sex = "male" if f["sex"] == 1 else "female"
            risk_word = "very high" if p["riskProb"] >= 0.7 else "moderate"
            lines.append(f"{i+1}. **Patient #{p['id']}** — {int(f['age'])}-year-old {sex} from {p['site']}, {risk_word} concern")
        return "**Patients with the highest heart disease concern:**\n" + "\n".join(lines) + "\n\nClick any of them in the scatter plot to see their full profile."

    if any(w in q for w in ["lowest", "safest", "least risk", "healthiest"]):
        bottom5 = sorted(PATIENTS, key=lambda p: p["riskProb"])[:5]
        lines = [f"{i+1}. **Patient #{p['id']}** ({p['site']}) — age {int(p['features']['age'])}, very low concern"
                 for i, p in enumerate(bottom5)]
        return "**Patients with the lowest heart disease concern:**\n" + "\n".join(lines)

    # Number / count / total questions — generic, comes after specific queries
    if any(w in q for w in ["how many", "count", "total", "selected"]) or q.strip() in ["patients", "dataset", "data"]:
        high = sum(1 for p in PATIENTS if p["riskProb"] >= 0.7)
        mod = sum(1 for p in PATIENTS if 0.4 <= p["riskProb"] < 0.7)
        low = sum(1 for p in PATIENTS if p["riskProb"] < 0.4)
        disease = sum(1 for p in PATIENTS if p["target"] == 1)
        no_disease = sum(1 for p in PATIENTS if p["target"] == 0)
        sites = {}
        for p in PATIENTS:
            sites.setdefault(p["site"], []).append(p)

        resp = f"The dataset contains **{len(PATIENTS)} patients** total.\n\n"
        resp += "**By risk level:**\n"
        resp += f"- High concern: **{high}** patients\n"
        resp += f"- Moderate concern: **{mod}** patients\n"
        resp += f"- Low concern: **{low}** patients\n\n"
        resp += "**By hospital site:**\n"
        for s, pts in sites.items():
            resp += f"- {s}: **{len(pts)}** patients\n"

        if patient:
            resp += f"\nCurrently viewing **Patient #{patient['id']}** ({patient['site']})."
        return resp

    # Catch-all for "patients" alone (dataset overview)
    if "patients" in q and not any(w in q for w in ["this patient", "their", "about"]):
        high = sum(1 for p in PATIENTS if p["riskProb"] >= 0.7)
        mod  = sum(1 for p in PATIENTS if 0.4 <= p["riskProb"] < 0.7)
        low  = sum(1 for p in PATIENTS if p["riskProb"] < 0.4)
        return (
            f"The dataset has **{len(PATIENTS)} patients** across 4 hospitals — "
            f"**{high}** with high concern, **{mod}** moderate, and **{low}** low. "
            f"Ask me 'which patients are most at risk?' or 'compare hospitals' to dig deeper."
        )

    # Compare sites / hospitals
    if any(w in q for w in ["compare", "site", "hospital", "cleveland", "hungary", "switzerland", "long beach"]):
        sites = {}
        for p in PATIENTS:
            sites.setdefault(p["site"], []).append(p["riskProb"])
        lines = []
        for s, risks in sites.items():
            high  = sum(1 for r in risks if r >= 0.7)
            low   = sum(1 for r in risks if r < 0.4)
            level = "high" if high / len(risks) > 0.5 else "moderate" if high / len(risks) > 0.3 else "relatively low"
            lines.append(f"- **{s}**: {len(risks)} patients — overall concern is {level} ({high} high-concern cases)")
        return "**Heart disease concern by hospital:**\n" + "\n".join(lines) + "\n\nSwitch the scatter plot colour to **Hospital Site** to compare visually."

    # What-if / cholesterol / change
    if any(w in q for w in ["what if", "what-if", "change", "modify", "cholesterol", "lower", "reduce"]):
        if patient:
            return f"You can modify **Patient #{patient['id']}'s** features using the **What-If Editor** in the Patient Detail panel. Click \"What-If Editor\", adjust values (e.g. lower cholesterol), and watch the SHAP waterfall update in real time."
        return "Select a patient first, then use the **What-If Editor** in the Patient Detail panel to modify feature values and see how the prediction changes."

    # Age-related
    if "age" in q or "old" in q or "young" in q:
        ages = [p["features"]["age"] for p in PATIENTS]
        avg_age = sum(ages) / len(ages)
        old_high = sum(1 for p in PATIENTS if p["features"]["age"] >= 60 and p["riskProb"] >= 0.7)
        young_high = sum(1 for p in PATIENTS if p["features"]["age"] < 45 and p["riskProb"] >= 0.7)
        return (
            f"**Age statistics:**\n"
            f"- Age range: {int(min(ages))} to {int(max(ages))} years\n"
            f"- Average age: {avg_age:.1f} years\n"
            f"- High-risk patients aged 60+: **{old_high}**\n"
            f"- High-risk patients under 45: **{young_high}**\n\n"
            f"Age contributes to risk through SHAP — older patients generally have higher positive SHAP values for age."
        )

    # Average / mean / statistics
    if any(w in q for w in ["average", "mean", "median", "statistic", "summary", "overview"]):
        avg_risk = sum(p["riskProb"] for p in PATIENTS) / len(PATIENTS) * 100
        avg_age = sum(p["features"]["age"] for p in PATIENTS) / len(PATIENTS)
        avg_chol = sum(p["features"]["chol"] for p in PATIENTS) / len(PATIENTS)
        avg_bp = sum(p["features"]["trestbps"] for p in PATIENTS) / len(PATIENTS)
        male_pct = sum(1 for p in PATIENTS if p["features"]["sex"] == 1) / len(PATIENTS) * 100
        return (
            f"**Dataset summary ({len(PATIENTS)} patients):**\n"
            f"- Average risk: **{avg_risk:.1f}%**\n"
            f"- Average age: **{avg_age:.1f}** years\n"
            f"- Average cholesterol: **{avg_chol:.0f}** mg/dl\n"
            f"- Average resting BP: **{avg_bp:.0f}** mmHg\n"
            f"- Male patients: **{male_pct:.0f}%**\n"
            f"- Disease prevalence: **{sum(1 for p in PATIENTS if p['target']==1)/len(PATIENTS)*100:.0f}%**"
        )

    # Greeting
    if any(w in q for w in ["hello", "hi", "hey", "help", "what can you"]):
        ctx = f" Currently viewing **Patient #{patient['id']}** — ask me about their risk." if patient else ""
        return (
            "Hello! I'm the HeartLens clinical assistant. I can help you explore the heart disease prediction data. Try asking:\n\n"
            "- **\"How many patients are in the dataset?\"**\n"
            "- **\"Which patients are most at risk?\"**\n"
            "- **\"Compare hospital sites\"**\n"
            "- **\"What are the most important features?\"**\n"
            "- **\"What's the average risk?\"**\n"
            "- Select a patient, then ask **\"Why is this patient high risk?\"**" + ctx
        )

    # Default — try to give something useful rather than just help text
    ctx = ""
    if patient:
        risk_pct = patient["riskProb"] * 100
        ctx = f"\n\nYou have **Patient #{patient['id']}** selected ({patient['site']}, {risk_pct:.0f}% risk). Try asking about their specific risk factors."
    else:
        ctx = "\n\nTip: Select a patient from the scatter plot to ask patient-specific questions."

    return (
        f"I'm not sure how to answer \"{message}\" specifically, but here's what I can help with:\n\n"
        "- **Patient counts & stats**: \"How many patients?\", \"What's the average risk?\"\n"
        "- **Risk analysis**: \"Who are the highest risk patients?\"\n"
        "- **Site comparison**: \"Compare hospital sites\"\n"
        "- **Feature analysis**: \"What features matter most?\"\n"
        "- **Patient details**: Select a patient, then \"Why is this patient high risk?\"" + ctx
    )


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.getenv("FLASK_PORT", 5001))
    print(f"\nStarting HeartLens API on port {port}...")
    app.run(debug=True, port=port, host="0.0.0.0")
