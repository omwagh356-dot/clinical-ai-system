import os
import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import smtplib
from email.message import EmailMessage

# --- IMPORTANT: Upload these files to GitHub too! ---
try:
    from drug_module import check_drugs
    from explain import explain_values
except ImportError:
    # Fallback if files aren't uploaded yet
    def check_drugs(a, b, c): return ["Module not found"], ["Upload drug_module.py"]
    def explain_values(a, b, c, d): return ["Analysis pending"]

# --- PAGE CONFIG ---
st.set_page_config(page_title="Clinical AI Portal", page_icon="🏥", layout="wide")

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    model = load_model("model/model.h5")
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    return model, scaler, label_encoder

try:
    model, scaler, label_encoder = load_assets()
except Exception as e:
    st.error("⚠️ Model assets missing! Ensure model.h5, scaler.pkl, and label_encoder.pkl are on GitHub.")

# --- EMAIL FUNCTION ---
def send_to_doctor(receiver_email, data):
    msg = EmailMessage()
    msg['Subject'] = f"Urgent Report: {data['Risk']} Risk - {data['Name']}"
    msg['From'] = st.secrets["EMAIL_USER"]
    msg['To'] = receiver_email
    
    msg.set_content(f"""
    URGENT CLINICAL REPORT
    ----------------------
    Patient: {data['Name']} | Age: {data['Age']} | Gender: {data['Gender']}
    
    AI PREDICTION: {data['Disease']} ({data['Prob']}%)
    RISK LEVEL: {data['Risk']}
    URGENCY ACTION: {data['Urgency']}
    REASONS: {data['Reasons']}
    
    VITAL SIGNS:
    HR: {data['hr']} | SpO2: {data['spo2']}% | Temp: {data['temp']}°C
    BP: {data['bps']}/{data['bpd']} | Glucose: {data['gluc']}
    """)

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(st.secrets["EMAIL_USER"], st.secrets["EMAIL_PASS"])
            smtp.send_message(msg)
        return True
    except:
        return False

# --- UI LAYOUT ---
st.title("🏥 Clinical AI Diagnostic System")

# Create two main columns for the layout
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("👤 Patient Identity")
    name = st.text_input("Full Name")
    doc_email = st.text_input("Doctor's Email")
    g_col, a_col = st.columns(2)
    gender = g_col.selectbox("Gender", ["Male", "Female", "Other"])
    age = a_col.number_input("Age", value=30)

    st.subheader("📉 Clinical Vitals")
    v1, v2, v3 = st.columns(3)
    hr = v1.number_input("Heart Rate", value=72.0)
    bps = v1.number_input("BP Systolic", value=120.0)
    spo2 = v2.number_input("SpO2 %", value=98.0)
    bpd = v2.number_input("BP Diastolic", value=80.0)
    temp = v3.number_input("Temp °C", value=37.0)
    gluc = v3.number_input("Glucose", value=95.0)
    
    # Hidden defaults from your original code
    chol = 190.0
    resp = 16.0

with col_right:
    st.subheader("💊 Safety & History")
    curr_drugs = st.text_area("Current Medications (comma separated)", placeholder="e.g. Aspirin, Metformin")
    curr_diseases = st.text_area("Known Conditions", placeholder="e.g. Diabetes, Asthma")
    curr_allergies = st.text_area("Allergies", placeholder="e.g. Penicillin")
    
    if st.button("Check Drug Safety", use_container_width=True):
        warnings, recs = check_drugs(curr_drugs.split(","), curr_diseases.split(","), curr_allergies.split(","))
        st.warning(f"**Warnings:** {chr(10).join(warnings)}")
        st.success(f"**Recommendations:** {', '.join(recs)}")

# --- ANALYSIS TRIGGER ---
st.divider()
if st.button("RUN FULL DIAGNOSTIC", type="primary", use_container_width=True):
    # 1. Prediction
    inputs = np.array([[age, hr, bps, bpd, spo2, temp, chol, gluc, resp]])
    scaled = scaler.transform(pd.DataFrame(inputs, columns=["age", "heart_rate", "bp_systolic", "bp_diastolic", "spo2", "temp", "cholesterol", "glucose", "respiratory_rate"]))
    pred = model.predict(scaled, verbose=0)
    idx = np.argmax(pred)
    disease = label_encoder.inverse_transform([idx])[0]
    prob = pred[0][idx] * 100

    # 2. Triage & Explanation
    risk = "Low"
    urgency = "No immediate action required"
    if disease.lower() != "normal" and prob > 60:
        risk = "Medium"
        urgency = "Consultation recommended"
    
    if spo2 < 90 or temp > 39.5 or temp < 35.1 or bps >= 180 or gluc < 50:
        risk = "High"
        urgency = "IMMEDIATE DOCTOR VISIT REQUIRED / ER"

    reasons = explain_values(hr, (bps+bpd)/2, spo2, temp)

    # 3. Report & Transfer
    report = {
        "Name": name, "Age": age, "Gender": gender, "Disease": disease, 
        "Risk": risk, "Prob": f"{prob:.2f}", "Urgency": urgency, "Reasons": ", ".join(reasons),
        "hr": hr, "spo2": spo2, "temp": temp, "bps": bps, "bpd": bpd, "gluc": gluc
    }
    
    # 4. Results Display
    st.subheader("📊 Diagnostic Result")
    res_box = st.container(border=True)
    if risk == "High":
        res_box.error(f"**PREDICTION:** {disease} ({prob:.2f}%)")
    else:
        res_box.success(f"**PREDICTION:** {disease} ({prob:.2f}%)")
        
    res_box.write(f"**Risk Level:** {risk} | **Action:** {urgency}")
    res_box.write(f"**AI Reasoning:** {report['Reasons']}")

    # 5. Transfer
    if doc_email:
        with st.spinner("Sending report..."):
            if send_to_doctor(doc_email, report):
                st.success("Report Sent Successfully ✅")
            else:
                st.info("Report Saved Locally (Check output.csv) 💾")
    
    # Save to local CSV on the server
    pd.DataFrame([report]).to_csv("output.csv", mode='a', header=not os.path.exists("output.csv"), index=False)
