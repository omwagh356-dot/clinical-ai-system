import os
import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import smtplib
from email.message import EmailMessage
import base64

# --- 1. EXTERNAL LOGIC IMPORTS ---
try:
    from drug_module import check_drugs
    from explain import explain_values
except ImportError:
    st.error("⚠️ Missing 'drug_module.py' or 'explain.py' on GitHub. Please upload them!")

# --- 2. KNOWLEDGE BASES ---
SYMPTOM_DRUGS = {
    "chest pain": {"rec": "Aspirin (324mg), Nitroglycerin.", "safety": "🚨 CRITICAL: High risk of Heart Attack. Proceed to ER."},
    "cough": {"rec": "Dextromethorphan or Guaifenesin.", "safety": "⚠️ Avoid suppressants if cough is productive with fever."},
    "fever": {"rec": "Acetaminophen (650mg) or Ibuprofen.", "safety": "✅ Monitor for stiff neck or confusion."},
    "cold": {"rec": "Decongestants and Vitamin C.", "safety": "⚠️ Decongestants raise Blood Pressure. Use caution."},
    "diarrhea": {"rec": "Loperamide and ORS.", "safety": "⚠️ Do not use if stool is bloody or fever is high."},
    "headache": {"rec": "Ibuprofen or Naproxen.", "safety": "✅ Seek care if it is the 'worst headache of your life'."},
    "nausea": {"rec": "Ginger or Ondansetron.", "safety": "⚠️ Persistent vomiting leads to Electrolyte Imbalance."},
    "shortness of breath": {"rec": "Albuterol or Oxygen.", "safety": "🚨 EMERGENCY: High risk of Respiratory Failure."},
    "dizziness": {"rec": "Meclizine or hydration.", "safety": "⚠️ Risk of Stroke if slurred speech is present."},
    "abdominal pain": {"rec": "Antacids.", "safety": "⚠️ Avoid heating pads if pain is sharp in lower right."},
    "allergic reaction": {"rec": "Antihistamine or Epinephrine.", "safety": "🚨 CRITICAL: Use Epi-Pen if throat feels tight."}
}

CLINICAL_DATABASE = {
    "Infection": {"icon": "🤒", "severity": "High", "drugs": ["Acetaminophen", "Ceftriaxone", "IV Saline"], "next_steps": "Blood Cultures, CBC, Lactic Acid check.", "safety": "Monitor for Septic Shock."},
    "Respiratory Failure": {"icon": "🫁", "severity": "Critical", "drugs": ["Oxygen", "Albuterol", "Steroids"], "next_steps": "ABG, Chest X-Ray, Intubation Eval.", "safety": "Keep head of bed elevated."},
    "Hypertension": {"icon": "🩸", "severity": "High", "drugs": ["Lisinopril", "Amlodipine"], "next_steps": "ECG, Urinalysis, BP monitoring.", "safety": "Risk of Stroke. Avoid sudden movement."},
    "Normal": {"icon": "✅", "severity": "Stable", "drugs": ["Maintain regimen"], "next_steps": "Routine 6-month follow-up.", "safety": "Cleared for standard activity."},
    "Cardiac Emergency": {"icon": "💔", "severity": "Critical", "drugs": ["Aspirin", "Nitroglycerin", "Morphine"], "next_steps": "12-Lead ECG, Troponin levels.", "safety": "Minimize all movement. Prepare for ACLS."}
}

# --- 3. PAGE CONFIG & ASSETS ---
st.set_page_config(page_title="Clinical AI Portal", page_icon="🏥", layout="wide")

@st.cache_resource
def load_assets():
    model = load_model("model/model.h5")
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    return model, scaler, label_encoder

model, scaler, label_encoder = load_assets()

# --- 4. CORE FUNCTIONS (PDF & EMAIL) ---
def create_pdf_report(report_data, info, reasons, safety_warnings):
    html_content = f"""
    <div style="font-family: Arial; padding: 20px; border: 2px solid #333;">
        <h1 style="color: #1a73e8; text-align: center;">Clinical AI Diagnostic Report</h1>
        <hr>
        <p><b>Patient:</b> {report_data['Name']} | <b>Age:</b> {report_data['Age']} | <b>Gender:</b> {report_data['Gender']}</p>
        <div style="background: #f8f9fa; padding: 15px; border-radius: 10px;">
            <p><b>Prediction:</b> {report_data['Disease']} ({report_data['Prob']}%)</p>
            <p><b>Urgency:</b> {report_data['Urgency']}</p>
        </div>
        <h3>Reported Symptoms & Conditions</h3>
        <p>{report_data['Symptoms']}</p>
        <h3>Vitals Analysis</h3>
        <ul>{"".join([f"<li>🚩 {r}</li>" for r in reasons])}</ul>
        <h3>Drug Safety Warnings</h3>
        <ul>{"".join([f"<li>⚠️ {w}</li>" for w in safety_warnings]) if safety_warnings else "<li>None flagged</li>"}</ul>
        <h3>Protocol</h3>
        <p><b>Next Steps:</b> {info['next_steps']}</p>
        <p><b>Meds:</b> {", ".join(info['drugs'])}</p>
    </div>
    """
    return html_content

def send_to_doctor(receiver_email, report, reasons, safety_warnings):
    msg = EmailMessage()
    msg['Subject'] = f"🚨 {report['Risk']} Risk Report - {report['Name']}"
    msg['From'] = st.secrets["EMAIL_USER"]
    msg['To'] = receiver_email
    
    body = f"""
    CLINICAL REPORT: {report['Name'],['Age'],['Gender']}
    -------------------------------
    RESULT: {report['Disease']} ({report['Prob']}%)
    URGENCY: {report['Urgency']}
    
    SYMPTOMS/CONDITIONS:
    {report['Symptoms']}

    VITALS ANALYSIS:
    {chr(10).join(['- ' + r for r in reasons])}

    DRUG SAFETY WARNINGS:
    {chr(10).join(['- ' + w for w in safety_warnings]) if safety_warnings else "No interactions flagged."}

    RAW VITALS: {report['vitals']}
    -------------------------------
    Generated via Clinical AI. Requires MD Verification.
    """
    msg.set_content(body)
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
    curr_diseases = st.text_area("Known Conditions / Symptoms")
    curr_allergies = st.text_area("Allergies")
    
    if st.button("CHECK DRUG INTERACTIONS", use_container_width=True):
        warnings_check, recs_check = check_drugs(curr_drugs.split(","), curr_diseases.split(","), curr_allergies.split(","))
        for w in warnings_check: st.error(w)
        for r in recs_check: st.success(r)

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

    # Get Database Info & Drug Checks
    info = CLINICAL_DATABASE.get(disease, CLINICAL_DATABASE["Normal"])
    reasons = explain_values(hr, (bps+bpd)/2, spo2, temp)
    warnings, _ = check_drugs(curr_drugs.split(","), curr_diseases.split(","), curr_allergies.split(","))

    # UI Result Display
    st.header(f"{info['icon']} Diagnosis: {disease}")
    tab1, tab2, tab3 = st.tabs(["🎯 Results", "💊 Therapy", "📄 Export & Email"])
    
    with tab1:
        st.metric("AI Confidence", f"{prob:.2f}%", delta=risk)
        st.write(f"**Urgency:** {urgency}")
        st.write("**Reasoning:**", ", ".join(reasons))
        
    with tab2:
        st.subheader("💊 Symptom-Based Recommendations")
        user_input = curr_diseases.lower()
        found_symptom = False
        for symptom, advice in SYMPTOM_DRUGS.items():
            if symptom in user_input:
                st.markdown(f"### For {symptom.capitalize()}:")
                st.success(f"**Recommended:** {advice['rec']}")
                st.warning(f"**Safety Check:** {advice['safety']}")
                found_symptom = True
                st.divider()
        if not found_symptom:
            st.info("Enter symptoms in the Known Conditions box for specific drug advice.")

        st.subheader(f"Standard Protocol for {disease}")
        for d in info['drugs']: st.write(f"- {d}")
        
    with tab3:
        report_data = {
            "Name": name, "Age": age, "Gender": gender, "Disease": disease, 
            "Prob": f"{prob:.2f}", "Risk": risk, "Urgency": urgency, 
            "Symptoms": curr_diseases, "vitals": inputs_raw
        }
        
        # HTML Report Generation
        html_report = create_pdf_report(report_data, info, reasons, warnings)
        st.download_button("Download Medical Report (HTML)", data=html_report, file_name=f"Report_{name}.html", mime="text/html", use_container_width=True)

        # Email Transfer
        if doc_email:
            with st.spinner("Sending report to doctor..."):
                if send_to_doctor(doc_email, report_data, reasons, warnings):
                    st.success("Report Transmitted Successfully! ✅")
                else:
                    st.error("Email transmission failed. Check your Secrets.")
