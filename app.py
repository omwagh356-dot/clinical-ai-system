import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import smtplib
from email.message import EmailMessage

# --- 1. CORE FUNCTIONS ---

def create_clinical_report(report, reasons, warnings):
    return f"""
    <div style='font-family:Arial; border:2px solid #333; padding:20px; border-radius:10px;'>
        <h1 style='color:#1a73e8; text-align:center;'>Clinical AI Diagnostic Report</h1>
        <hr>
        <p><b>Patient:</b> {report['Name']} | <b>Age:</b> {report['Age']}</p>
        <div style='background:#f0f2f6; padding:15px; border-radius:10px;'>
            <h3>Diagnosis: {report['Disease']}</h3>
            <p><b>Triage: {report['Risk']}</b></p>
            <p><b>Confidence:</b> {report['Prob']}%</p>
        </div>
        <h4>Analysis:</h4>
        <ul>{"".join([f"<li>{r}</li>" for r in reasons])}</ul>
    </div>
    """

def send_to_physician(receiver, report):
    msg = EmailMessage()
    msg['Subject'] = f"🚨 {report['Risk']} Alert - {report['Name']}"
    msg['From'] = st.secrets.get("EMAIL_USER", "")
    msg['To'] = receiver
    msg.set_content(f"Diagnosis: {report['Disease']} | Risk: {report['Risk']}")
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(st.secrets["EMAIL_USER"], st.secrets["EMAIL_PASS"])
            smtp.send_message(msg)
        return True
    except:
        return False

# --- 2. LOAD DATA ---

@st.cache_data
def load_data():
    med_df = pd.read_csv("Medicine_description.csv")
    return med_df

@st.cache_resource
def load_model_assets():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    le = joblib.load("label_encoder.pkl")
    return model, scaler, le

model, scaler, label_encoder = load_model_assets()
med_db = load_data()

# Optional modules
try:
    from drug_module import check_drugs
except:
    check_drugs = None

# --- 3. UI ---

st.set_page_config(page_title="CDSS", layout="wide")
st.title("🛡️ Clinical Decision Support System")

c1, c2 = st.columns([1, 1.2])

with c1:
    name = st.text_input("Patient Name")
    doc_email = st.text_input("Doctor Email")
    age = st.number_input("Age", 1, 120, 30)

    st.subheader("Vitals")
    hr = st.number_input("Heart Rate", 40, 180, 72)
    bp = st.number_input("BP Systolic", 80, 200, 120)
    spo2 = st.number_input("SpO2", 70, 100, 98)
    temp = st.number_input("Temperature", 35.0, 42.0, 37.0)
    gluc = st.number_input("Glucose", 50, 400, 95)

with c2:
    symptoms = st.text_area("Symptoms")
    meds = st.text_area("Current Medications")

# --- 4. PREDICTION ---

if st.button("🚀 Run Diagnosis"):

    # Prepare input
    input_data = pd.DataFrame([{
        "Age": age,
        "HeartRate": hr,
        "BP_Systolic": bp,
        "SpO2": spo2,
        "Temperature": temp,
        "Glucose": gluc
    }])

    # Align columns
    input_data = input_data.reindex(columns=scaler.feature_names_in_, fill_value=0)

    scaled = scaler.transform(input_data)

    # Predict
    pred = model.predict(scaled)
    prob = model.predict_proba(scaled)

    disease = label_encoder.inverse_transform(pred)[0]
    confidence = np.max(prob) * 100

    # Risk Score
    risk_score = 0
    if hr > 120: risk_score += 2
    if spo2 < 90: risk_score += 3
    if temp > 39: risk_score += 3
    if gluc > 250: risk_score += 2

    status = "🔴 CRITICAL" if risk_score >= 5 else "🟢 STABLE"

    st.success(f"{status} : {disease} ({confidence:.2f}%)")

    # --- Analytics ---
    st.subheader("📊 Prediction Distribution")
    df_plot = pd.DataFrame({
        "Disease": label_encoder.classes_,
        "Probability": prob[0]*100
    })
    st.plotly_chart(px.bar(df_plot, x="Probability", y="Disease"))

    # --- Drug Recommendation ---
    st.subheader("💊 Recommended Medicines")
    rel = med_db[med_db['Reason'].str.contains(disease, case=False, na=False)].head(5)

    if not rel.empty:
        for _, row in rel.iterrows():
            st.write(f"**{row['Drug_Name']}** - {row['Description']}")
    else:
        st.warning("No medicine found")

    # --- Drug Interaction ---
    if meds and check_drugs:
        interaction = check_drugs(meds)
        if interaction:
            st.error(f"⚠️ Interaction Warning: {interaction}")

    # --- Email Alert ---
    if status == "🔴 CRITICAL" and doc_email:
        if send_to_physician(doc_email, {"Name": name, "Disease": disease, "Risk": status}):
            st.success("Alert sent to doctor")

    # --- Report ---
    report_data = {
        "Name": name,
        "Age": age,
        "Disease": disease,
        "Prob": round(confidence, 2),
        "Risk": status
    }

    html = create_clinical_report(report_data, ["Vitals analyzed"], [])
    st.download_button("📄 Download Report", html, file_name="report.html")
