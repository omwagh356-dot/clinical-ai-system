import os
import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import smtplib
from email.message import EmailMessage

# --- 1. EXTERNAL LOGIC IMPORTS ---
try:
    from drug_module import check_drugs
    from explain import explain_values
except ImportError:
    st.error("⚠️ Missing 'drug_module.py' or 'explain.py' on GitHub. Please upload them!")

# --- 2. THE CLINICAL KNOWLEDGE BASE ---
SYMPTOM_DRUGS = {
    "chest pain": {
        "rec": "Aspirin (324mg chewable), Nitroglycerin (if prescribed).",
        "safety": "🚨 CRITICAL: Possible Myocardial Infarction. Do not delay transport to ER."
    },
    "cough": {
        "rec": "Dextromethorphan (suppressant) or Guaifenesin (expectorant).",
        "safety": "⚠️ Avoid suppressants if cough is productive (bringing up mucus) and associated with fever."
    },
    "fever": {
        "rec": "Acetaminophen (650mg) or Ibuprofen (400mg).",
        "safety": "✅ Monitor for 'red flags' like stiff neck or confusion."
    },
    "cold": {
        "rec": "Decongestants (Pseudoephedrine) and Vitamin C.",
        "safety": "⚠️ Decongestants can significantly raise Blood Pressure. Use with caution in Hypertension."
    },
    "diarrhea": {
        "rec": "Loperamide (Imodium) and Oral Rehydration Salts (ORS).",
        "safety": "⚠️ Do not use Loperamide if there is blood in stool or high fever (indicates bacterial infection)."
    },
    "headache": {
        "rec": "Ibuprofen (400mg) or Naproxen.",
        "safety": "✅ Seek immediate care if headache is 'the worst of your life' or follows a head injury."
    },
    "nausea": {
        "rec": "Ginger extract or Ondansetron (Zofran - Rx only).",
        "safety": "⚠️ Persistent vomiting leads to Electrolyte Imbalance. Monitor SpO2 and HR."
    },
    "shortness of breath": {
        "rec": "Rescue Inhaler (Albuterol) if prescribed; Oxygen.",
        "safety": "🚨 EMERGENCY: High risk of Respiratory Failure. Monitor SpO2 immediately."
    }
}
# --- 3. THE CLINICAL KNOWLEDGE BASE ---
CLINICAL_DATABASE = {
    "Infection": {
        "icon": "🤒", "severity": "High",
        "symptoms": ["High Fever", "Tachycardia (High HR)", "Chills", "Lethargy"],
        "drugs": ["Acetaminophen (for fever)", "Broad-spectrum Antibiotics (Ceftriaxone)", "IV Saline"],
        "next_steps": "1. Perform Blood Cultures x2. \n2. Order CBC with Differential. \n3. Check Lactic Acid levels.",
        "safety": "Monitor temperature every 30 minutes. Risk of Septic Shock if Blood Pressure drops."
    },
    "Respiratory Failure": {
        "icon": "🫁", "severity": "Critical",
        "symptoms": ["Low SpO2", "Cyanosis", "Rapid shallow breathing"],
        "drugs": ["Supplemental Oxygen", "Albuterol Nebulizer", "Methylprednisolone (Steroid)"],
        "next_steps": "1. Immediate ABG (Arterial Blood Gas). \n2. Portable Chest X-Ray. \n3. Evaluate for BiPAP.",
        "safety": "Do not leave patient unattended. Keep head of bed elevated."
    },
    "Hypertension": {
        "icon": "🩸", "severity": "High",
        "symptoms": ["Severe Headache", "Chest Pain", "Blurred Vision"],
        "drugs": ["Lisinopril", "Amlodipine", "Labetalol (if crisis)"],
        "next_steps": "1. Repeat BP in both arms. \n2. 12-Lead ECG. \n3. Urinalysis.",
        "safety": "Risk of Stroke. Advise patient to avoid sudden movements."
    },
    "Normal": {
        "icon": "✅", "severity": "Stable",
        "symptoms": ["Vitals within physiological limits"],
        "drugs": ["Maintain current regimen", "Multivitamins"],
        "next_steps": "1. Routine follow-up in 6 months. \n2. Continue lifestyle management.",
        "safety": "Cleared for standard activity."
    },
    "Cardiac Emergency": {
        "icon": "💔",
        "severity": "Critical",
        "symptoms": ["Chest Pressure", "Left Arm Pain", "Shortness of Breath", "Cold Sweat"],
        "drugs": ["Aspirin (324mg)", "Nitroglycerin", "Morphine", "High-flow Oxygen"],
        "next_steps": "1. 12-Lead ECG immediately. \n2. Check Troponin I/T levels. \n3. Activate Cath Lab/Cardiac Team.",
        "safety": "Keep patient sitting upright. Minimize all physical movement. Prepare for ACLS."
    }
}

# --- 3. PAGE CONFIG & MODEL LOADING ---
st.set_page_config(page_title="Clinical AI Portal", page_icon="🏥", layout="wide")

@st.cache_resource
def load_assets():
    model = load_model("model/model.h5")
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    return model, scaler, label_encoder

model, scaler, label_encoder = load_assets()

# --- 4. EMAIL SYSTEM ---
def send_to_doctor(receiver_email, report):
    msg = EmailMessage()
    msg['Subject'] = f"🚨 {report['Risk']} Risk Clinical Report - {report['Name']}"
    msg['From'] = st.secrets["EMAIL_USER"]
    msg['To'] = receiver_email
    msg.set_content(f"Patient: {report['Name']}\nDiagnosis: {report['Disease']}\nUrgency: {report['Urgency']}\n\nVitals: {report['vitals']}")
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(st.secrets["EMAIL_USER"], st.secrets["EMAIL_PASS"])
            smtp.send_message(msg)
        return True
    except: return False

# --- 5. UI LAYOUT ---
st.title("🏥 Clinical AI Diagnostic Dashboard")

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("👤 Patient Identity")
    name = st.text_input("Patient Name")
    doc_email = st.text_input("Doctor's Email")
    g_col, a_col = st.columns(2)
    gender = g_col.selectbox("Gender", ["Male", "Female", "Other"])
    age = a_col.number_input("Age", value=30)

    st.subheader("📉 Clinical Vitals")
    v1, v2, v3 = st.columns(3)
    hr = v1.number_input("Heart Rate", value=72.0)
    bps = v1.number_input("BP Systolic", value=120.0)
    resp = v1.number_input("Resp. Rate", value=16.0)
    spo2 = v2.number_input("SpO2 %", value=98.0)
    bpd = v2.number_input("BP Diastolic", value=80.0)
    chol = v2.number_input("Cholesterol", value=190.0)
    temp = v3.number_input("Temp °C", value=37.0)
    gluc = v3.number_input("Glucose", value=95.0)

with col_right:
    st.subheader("💊 Medication Safety")
    curr_drugs = st.text_area("Current Medications (comma separated)")
    curr_diseases = st.text_area("Known Conditions")
    curr_allergies = st.text_area("Allergies")
    
    if st.button("CHECK DRUG INTERACTIONS", use_container_width=True):
        warnings, recs = check_drugs(curr_drugs.split(","), curr_diseases.split(","), curr_allergies.split(","))
        for w in warnings: st.error(w)
        for r in recs: st.success(r)

# --- 6. EXECUTION BLOCK ---
st.divider()
if st.button("RUN FULL DIAGNOSTIC", type="primary", use_container_width=True):
    # ML Prediction
    inputs_raw = [age, hr, bps, bpd, spo2, temp, chol, gluc, resp]
    inputs_df = pd.DataFrame([inputs_raw], columns=scaler.feature_names_in_)
    scaled = scaler.transform(inputs_df)
    pred = model.predict(scaled, verbose=0)
    idx = np.argmax(pred)
    disease = label_encoder.inverse_transform([idx])[0]
    prob = pred[0][idx] * 100

    # Triage Overrides
    risk, urgency = "Low", "Routine follow-up"
    if spo2 < 90 or temp > 39.5 or bps >= 180:
        risk, urgency = "High", "IMMEDIATE ER VISIT REQUIRED"

    # Get Database Info
    info = CLINICAL_DATABASE.get(disease, CLINICAL_DATABASE["Normal"])
    reasons = explain_values(hr, (bps+bpd)/2, spo2, temp)

    # UI Result Display
    st.header(f"{info['icon']} Diagnosis: {disease}")
    tab1, tab2, tab3 = st.tabs(["🎯 Results", "💊 Therapy", "🔬 Next Steps"])
    
    with tab1:
        st.metric("AI Confidence", f"{prob:.2f}%", delta=risk)
        st.write(f"**Urgency:** {urgency}")
        st.write("**Reasoning:**", ", ".join(reasons))
        
    with tab2:
        st.subheader("Pharmacotherapy Recommendations")
        for d in info['drugs']: st.write(f"- {d}")
        
    with tab3:
        st.info(f"**Next Steps:**\n{info['next_steps']}")
        st.warning(f"**Safety:** {info['safety']}")

    # Email Transfer
    if doc_email:
        report = {"Name": name, "Age": age, "Gender": gender, "Disease": disease, "Risk": risk, "Urgency": urgency, "vitals": inputs_raw}
        if send_to_doctor(doc_email, report): st.success("Report Transmitted Successfully! ✅")
