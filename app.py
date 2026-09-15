
from flask import Flask, render_template, request, jsonify
import sqlite3, json, os, math
from datetime import datetime, date

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "health_journal.db")
MODEL_DIR = os.path.join(BASE_DIR, "models")

app = Flask(__name__)

FEATURES = [
    "Age", "BMI", "SystolicBP", "DiastolicBP", "HeartRate",
    "Temperature", "SleepHours", "WaterLitres", "ActivityMinutes",
    "StressLevel", "SmokingStatus"
]

def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS journal (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entry_date TEXT NOT NULL,
            entry_time TEXT NOT NULL,
            age REAL,
            weight REAL,
            height REAL,
            bmi REAL,
            systolic REAL,
            diastolic REAL,
            heart_rate REAL,
            temperature REAL,
            sleep REAL,
            water REAL,
            activity REAL,
            stress INTEGER,
            smoking TEXT,
            symptoms TEXT,
            notes TEXT,
            about_me TEXT,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def num(form, key):
    value = form.get(key, "").strip()
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None

def bmi_category(bmi):
    if bmi is None:
        return "Not available"
    if bmi < 18.5: return "Underweight range"
    if bmi < 25: return "Normal range"
    if bmi < 30: return "Overweight range"
    return "Obesity range"

def xgb_risks(data):
    """Use optional XGBoost demonstration models if they exist."""
    try:
        import joblib
        import numpy as np
        names=["diabetes","hypertension","heart","respiratory"]
        paths=[os.path.join(MODEL_DIR,f"{n}_xgboost.joblib") for n in names]
        feature_path=os.path.join(MODEL_DIR,"feature_columns.joblib")
        if not all(os.path.exists(x) for x in paths+[feature_path]):
            return None
        smoking = 1 if data.get("smoking") == "Yes" else 0
        values=[data.get("age"),data.get("bmi"),data.get("systolic"),data.get("diastolic"),
                data.get("heart_rate"),data.get("temperature"),data.get("sleep"),
                data.get("water"),data.get("activity"),data.get("stress"),smoking]
        # Missing optional values are represented by 0 only for the synthetic demonstration models.
        # The UI still tells the user which information was not provided.
        X=np.array([[0 if v is None else float(v) for v in values]])
        labels=["Diabetes","Hypertension","Heart disease","Respiratory condition"]
        result={}
        for label,path in zip(labels,paths):
            model=joblib.load(path)
            result[label]=round(float(model.predict_proba(X)[0,1])*100)
        return result
    except Exception:
        return None

def risk_from_inputs(data):
    """Educational demonstration only. Not a clinical prediction model."""
    age = data.get("age")
    bmi = data.get("bmi")
    sbp = data.get("systolic")
    dbp = data.get("diastolic")
    hr = data.get("heart_rate")
    sleep = data.get("sleep")
    activity = data.get("activity")
    stress = data.get("stress")
    smoking = data.get("smoking")
    symptoms = set(data.get("symptoms", []))

    def pct(base, factors):
        score = base + sum(f for f in factors if f is not None)
        return max(1, min(95, round(score)))

    diabetes = pct(10, [
        12 if bmi is not None and bmi >= 30 else 5 if bmi is not None and bmi >= 25 else 0,
        6 if age is not None and age >= 45 else 0,
        4 if "Excessive thirst" in symptoms or "Frequent urination" in symptoms else 0
    ])
    hypertension = pct(8, [
        30 if sbp is not None and sbp >= 140 else 18 if sbp is not None and sbp >= 130 else 0,
        15 if dbp is not None and dbp >= 90 else 8 if dbp is not None and dbp >= 80 else 0,
        6 if age is not None and age >= 45 else 0,
        5 if smoking == "Yes" else 0
    ])
    heart = pct(6, [
        18 if sbp is not None and sbp >= 140 else 8 if sbp is not None and sbp >= 130 else 0,
        8 if hr is not None and (hr < 50 or hr > 100) else 0,
        7 if smoking == "Yes" else 0,
        5 if age is not None and age >= 50 else 0,
        4 if "Chest discomfort" in symptoms or "Palpitations" in symptoms else 0
    ])
    respiratory = pct(5, [
        20 if smoking == "Yes" else 0,
        12 if "Shortness of breath" in symptoms else 0,
        8 if "Cough" in symptoms else 0,
        6 if "Wheezing" in symptoms else 0
    ])
    return {
        "Diabetes": diabetes,
        "Hypertension": hypertension,
        "Heart disease": heart,
        "Respiratory condition": respiratory
    }

def contributing_factors(data):
    out=[]
    bmi=data.get("bmi"); sbp=data.get("systolic"); dbp=data.get("diastolic")
    sleep=data.get("sleep"); activity=data.get("activity")
    symptoms=data.get("symptoms",[])
    if bmi is not None and bmi >= 25: out.append("BMI is in an above-reference range.")
    if sbp is not None and sbp >= 130: out.append("Systolic blood pressure is elevated relative to common reference ranges.")
    if dbp is not None and dbp >= 80: out.append("Diastolic blood pressure is elevated relative to common reference ranges.")
    if sleep is not None and sleep < 7: out.append("Sleep duration is below 7 hours.")
    if activity is not None and activity < 20: out.append("Reported activity is relatively low.")
    if symptoms: out.append("Reported symptoms were included in the educational screening estimate.")
    return out or ["There were no obvious factors in the entered information that changed the demonstration estimate."]

@app.route("/")
def index():
    init_db()
    return render_template("index.html")

@app.post("/predict")
def predict():
    data = {
        "age": num(request.form, "age"),
        "weight": num(request.form, "weight"),
        "height": num(request.form, "height"),
        "systolic": num(request.form, "systolic"),
        "diastolic": num(request.form, "diastolic"),
        "heart_rate": num(request.form, "heart_rate"),
        "temperature": num(request.form, "temperature"),
        "sleep": num(request.form, "sleep"),
        "water": num(request.form, "water"),
        "activity": num(request.form, "activity"),
        "stress": int(num(request.form, "stress") or 0) if request.form.get("stress","") else None,
        "smoking": request.form.get("smoking",""),
        "symptoms": request.form.getlist("symptoms"),
    }
    if data["weight"] and data["height"] and data["height"] > 0:
        data["bmi"] = round(data["weight"] / ((data["height"]/100)**2), 1)
    else:
        data["bmi"] = None

    model_risks = xgb_risks(data)
    risks = model_risks or risk_from_inputs(data)
    status = "XGBoost demonstration model — trained on synthetic educational data; not clinically validated." if model_risks else "Educational demonstration estimate — no trained XGBoost model files found."
    return jsonify({
        "bmi": data["bmi"],
        "bmi_category": bmi_category(data["bmi"]),
        "risks": risks,
        "factors": contributing_factors(data),
        "model_status": status
    })

@app.get("/api/journal")
def get_journal():
    conn=db()
    rows=conn.execute("SELECT * FROM journal ORDER BY entry_date DESC, entry_time DESC").fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

@app.post("/api/journal")
def save_journal():
    payload=request.get_json(silent=True) or {}
    conn=db()
    conn.execute("""
        INSERT INTO journal
        (entry_date,entry_time,age,weight,height,bmi,systolic,diastolic,heart_rate,
         temperature,sleep,water,activity,stress,smoking,symptoms,notes,about_me,created_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        payload.get("date") or date.today().isoformat(),
        payload.get("time") or datetime.now().strftime("%H:%M"),
        payload.get("age"), payload.get("weight"), payload.get("height"),
        payload.get("bmi"), payload.get("systolic"), payload.get("diastolic"),
        payload.get("heart_rate"), payload.get("temperature"), payload.get("sleep"),
        payload.get("water"), payload.get("activity"), payload.get("stress"),
        payload.get("smoking"), json.dumps(payload.get("symptoms",[])),
        payload.get("notes",""), payload.get("about_me",""), datetime.now().isoformat()
    ))
    conn.commit(); conn.close()
    return jsonify({"ok":True})

@app.delete("/api/journal/<int:entry_id>")
def delete_journal(entry_id):
    conn=db(); conn.execute("DELETE FROM journal WHERE id=?", (entry_id,)); conn.commit(); conn.close()
    return jsonify({"ok":True})

@app.get("/api/summary")
def summary():
    conn=db()
    rows=conn.execute("SELECT * FROM journal ORDER BY entry_date DESC, entry_time DESC").fetchall()
    conn.close()
    if not rows:
        return jsonify({"days":0,"streak":0,"latest":None,"first_date":None})
    unique=sorted({r["entry_date"] for r in rows}, reverse=True)
    streak=0
    from datetime import timedelta
    cursor=date.today()
    dates=set(unique)
    if cursor.isoformat() not in dates and (cursor-timedelta(days=1)).isoformat() in dates:
        cursor -= timedelta(days=1)
    while cursor.isoformat() in dates:
        streak+=1; cursor-=timedelta(days=1)
    latest=dict(rows[0])
    latest["symptoms"]=json.loads(latest.get("symptoms") or "[]")
    return jsonify({"days":len(unique),"streak":streak,"latest":latest,"first_date":min(unique)})

@app.post("/api/chat")
def chat():
    """Context-aware, friendly health assistant.

    This is intentionally local/offline: it does not pretend to be a medical LLM.
    It combines the user's latest journal record, history and the current message
    to produce useful explanations while keeping safety boundaries.
    """
    payload = request.get_json(silent=True) or {}
    raw_msg = (payload.get("message") or "").strip()
    msg = raw_msg.lower()
    history = payload.get("history") or []
    latest = payload.get("latest") or {}
    entries = payload.get("entries") or []

    if not raw_msg:
        return jsonify({"reply": "Tell me what you're noticing or ask me about your health entries. 😊"})

    # Decode symptoms from SQLite JSON strings when necessary.
    def symptoms_of(record):
        value = record.get("symptoms", []) if isinstance(record, dict) else []
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except Exception:
                value = [x.strip() for x in value.split(",") if x.strip()]
        return value or []

    latest_symptoms = symptoms_of(latest)
    latest_date = latest.get("entry_date") or ""

    # Emergency-first safety handling.
    emergency_terms = [
        "severe chest pain", "crushing chest pain", "can't breathe", "cannot breathe",
        "difficulty breathing", "severe difficulty breathing", "fainting", "passed out",
        "seizure", "confusion", "blue lips", "unconscious", "heavy bleeding"
    ]
    if any(term in msg for term in emergency_terms):
        return jsonify({"reply": "⚠️ Some symptoms you mentioned can be urgent. Please seek urgent medical care now or contact local emergency services. Don't rely on this app to decide whether an emergency is safe to wait out. If you want, I can also help you prepare a short summary of your recent readings for a clinician."})

    # Friendly acknowledgements.
    if any(term in msg for term in ["thank", "thanks", "great job", "good job", "you're good", "you are good"]):
        return jsonify({"reply": "You're welcome! 😊 Keep going at your own pace. Consistent tracking is more useful than trying to make every entry perfect."})

    # Context helpers.
    def fmt(v, suffix=""):
        if v in (None, "", "Not provided"):
            return "not provided"
        return f"{v}{suffix}"

    def latest_snapshot():
        if not latest:
            return "I don't have a saved journal entry yet."
        parts = []
        if latest.get("age") not in (None, ""): parts.append(f"age {latest['age']}")
        if latest.get("weight") not in (None, ""): parts.append(f"weight {latest['weight']} kg")
        if latest.get("height") not in (None, ""): parts.append(f"height {latest['height']} cm")
        if latest.get("bmi") not in (None, ""): parts.append(f"BMI {latest['bmi']}")
        if latest.get("systolic") not in (None, "") and latest.get("diastolic") not in (None, ""):
            parts.append(f"BP {latest['systolic']}/{latest['diastolic']}")
        if latest.get("heart_rate") not in (None, ""): parts.append(f"heart rate {latest['heart_rate']} bpm")
        if latest.get("temperature") not in (None, ""): parts.append(f"temperature {latest['temperature']}")
        if latest.get("sleep") not in (None, ""): parts.append(f"sleep {latest['sleep']} hours")
        if latest.get("water") not in (None, ""): parts.append(f"water {latest['water']} L")
        if latest.get("activity") not in (None, ""): parts.append(f"activity {latest['activity']} minutes")
        if latest.get("stress") not in (None, ""): parts.append(f"stress {latest['stress']}/10")
        if latest_symptoms: parts.append("symptoms: " + ", ".join(latest_symptoms))
        return (f"Your latest saved entry is from {latest_date}: " if latest_date else "Your latest saved entry: ") + (", ".join(parts) if parts else "it contains no filled measurements") + "."

    # Direct questions about the user's latest data.
    if any(term in msg for term in ["what do you know", "my latest", "latest entry", "my data", "my readings", "my health data"]):
        return jsonify({"reply": latest_snapshot() + " If you'd like, ask me about one reading and I'll explain what it generally means. I can't diagnose a condition from these values."})

    if "bmi" in msg or "body mass index" in msg:
        bmi = latest.get("bmi")
        if bmi in (None, ""):
            reply = "I don't have a BMI in your latest entry. BMI can be calculated from weight and height, so enter both values and save the entry."
        else:
            try:
                b = float(bmi)
                if b < 18.5: cat = "below the usual adult reference range"
                elif b < 25: cat = "within the usual adult reference range"
                elif b < 30: cat = "above the usual adult reference range"
                else: cat = "in the obesity range used by common adult BMI classifications"
                reply = f"Your latest BMI is {b:g}, which is {cat}. BMI is a screening measure, not a diagnosis, and it doesn't account for every person's body composition or health situation."
            except Exception:
                reply = f"Your latest recorded BMI is {bmi}. BMI is a screening measure rather than a diagnosis."
        return jsonify({"reply": reply})

    if "blood pressure" in msg or " bp" in f" {msg}" or msg.startswith("bp"):
        sbp, dbp = latest.get("systolic"), latest.get("diastolic")
        if sbp in (None, "") or dbp in (None, ""):
            reply = "Your latest entry doesn't contain both blood-pressure numbers. Enter systolic/diastolic values such as 120/80 if you have a reading."
        else:
            try:
                s, d = float(sbp), float(dbp)
                reply = f"Your latest blood pressure is {s:g}/{d:g} mmHg. A single reading doesn't establish a diagnosis; readings can vary with timing, activity, stress and measurement technique. If readings are repeatedly high or you feel unwell, discuss them with a healthcare professional."
            except Exception:
                reply = f"Your latest recorded BP is {sbp}/{dbp}."
        return jsonify({"reply": reply})

    if "heart rate" in msg or "pulse" in msg:
        hr = latest.get("heart_rate")
        reply = (f"Your latest recorded heart rate is {hr} bpm. Heart rate can vary with activity, stress, sleep, temperature and medications. "
                 "If it is persistently unusual for you or comes with concerning symptoms, seek medical advice." if hr not in (None, "") else
                 "I don't have a latest heart-rate reading. If you have one, save it in the journal and I can explain it in context.")
        return jsonify({"reply": reply})

    if "sleep" in msg:
        sl = latest.get("sleep")
        reply = (f"Your latest recorded sleep is {sl} hours. Sleep needs vary, but adults commonly need around 7–9 hours. Look at your trend rather than one night, and mention persistent sleep problems to a healthcare professional." if sl not in (None, "") else
                 "I don't have a latest sleep value. You can add it to your journal, then ask me about the trend.")
        return jsonify({"reply": reply})

    if "water" in msg or "hydration" in msg:
        wa = latest.get("water")
        reply = (f"Your latest recorded water intake is {wa} L. Hydration needs vary with heat, activity, diet and medical conditions, so there isn't one perfect number for everyone. "
                 "Regular fluids and paying attention to thirst and urine color can be useful general guides unless a clinician has given you different advice." if wa not in (None, "") else
                 "I don't have a latest water-intake value. Add it to your journal if you'd like to track hydration.")
        return jsonify({"reply": reply})

    # Risk/result explanation.
    if any(term in msg for term in ["risk", "prediction", "result", "why did i get", "why is my"]):
        factors = []
        bmi = latest.get("bmi")
        sbp, dbp = latest.get("systolic"), latest.get("diastolic")
        sleep = latest.get("sleep")
        activity = latest.get("activity")
        stress = latest.get("stress")
        if bmi not in (None, ""):
            try:
                if float(bmi) >= 25: factors.append(f"BMI ({bmi}) is above the usual adult reference range")
            except Exception: pass
        try:
            if sbp not in (None, "") and float(sbp) >= 130: factors.append(f"systolic BP ({sbp}) is elevated")
            if dbp not in (None, "") and float(dbp) >= 80: factors.append(f"diastolic BP ({dbp}) is elevated")
        except Exception: pass
        if sleep not in (None, ""):
            try:
                if float(sleep) < 7: factors.append(f"sleep ({sleep} hours) is relatively low")
            except Exception: pass
        if activity not in (None, ""):
            try:
                if float(activity) < 20: factors.append(f"activity ({activity} minutes) is relatively low")
            except Exception: pass
        if stress not in (None, ""):
            try:
                if float(stress) >= 7: factors.append(f"stress ({stress}/10) is high")
            except Exception: pass
        if latest_symptoms: factors.append("reported symptoms were included in the screening estimate")
        if factors:
            reply = "I can see these possible contributors in your latest saved information: " + "; ".join(factors) + ". These are associations used for an educational screening estimate, not proof that you have a disease."
        else:
            reply = "I don't have enough filled-in information to point to a specific contributor. Missing information is not treated as proof that everything is normal. The risk cards are educational estimates, not diagnoses."
        return jsonify({"reply": reply})

    # Symptoms and common health questions.
    if any(term in msg for term in ["headache", "head ache"]):
        return jsonify({"reply": "Headaches can have many causes, including dehydration, lack of sleep, stress, illness and others. Track when it started, how severe it is, where it hurts, sleep, fluids and other symptoms. A sudden severe headache, or a headache with weakness, confusion, fainting, seizures, vision loss or other serious symptoms needs urgent medical assessment."})

    if "fever" in msg or "temperature" in msg:
        temp = latest.get("temperature")
        prefix = f"Your latest recorded temperature is {temp}. " if temp not in (None, "") else ""
        return jsonify({"reply": prefix + "For a fever, rest, drink appropriate fluids and monitor how you're feeling and your temperature. Seek medical care if it is severe, persistent, worsening, or accompanied by breathing difficulty, confusion, fainting, severe dehydration or other concerning symptoms. I can't diagnose the cause or prescribe medication."})

    if "doctor" in msg or "doctor" in msg or "hospital" in msg:
        return jsonify({"reply": "For a doctor visit, it can help to bring: when the symptom started, how often it happens, severity, recent temperature/BP/heart-rate readings, sleep and activity changes, current medicines or supplements, allergies, and your journal history. You can also show the clinician your saved dates and notes from this app."})

    if any(term in msg for term in ["missing", "empty", "not provided", "blank"]):
        return jsonify({"reply": "Blank optional fields stay blank. I don't assume that missing information is normal. If a reading matters to your question, enter it in the journal and save the record; then ask me again."})

    if any(term in msg for term in ["summary", "about me", "tell you about me"]):
        about = latest.get("about_me") or ""
        return jsonify({"reply": ("Your latest journal entry includes this personal note: " + about[:500] + ". " if about else "I don't see a personal note in your latest entry. ") + "You can use the 'Tell us about yourself' box to add context that helps the assistant understand your situation. Please avoid entering passwords or highly sensitive information."})

    # Use recent conversation context if available, so follow-ups feel natural.
    previous_user = ""
    for item in reversed(history):
        if isinstance(item, dict) and item.get("role") == "user":
            previous_user = item.get("content", "")
            break

    if msg in {"yes", "yeah", "yep", "okay", "ok", "sure"} and previous_user:
        return jsonify({"reply": "Absolutely. 😊 Tell me which part you want to look at next—your latest reading, symptoms, journal trend, or the screening estimate."})

    # General conversational fallback with useful next actions.
    return jsonify({"reply": "I can help you understand the information you've entered, your journal history, BMI, blood pressure, heart rate, sleep, hydration, symptoms and the educational screening estimate. 😊 For example, you can ask: 'What are my latest readings?', 'Why is my risk higher?', 'Explain my BMI', 'What information is missing?', or 'What should I tell my doctor?' I can't diagnose you, but I can help you make sense of your records."})

init_db()

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
