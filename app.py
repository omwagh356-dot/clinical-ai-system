import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import smtplib
from email.message import EmailMessage
from tensorflow.keras.models import load_model
import io

# --- 1. CONFIG & ASSETS ---
st.set_page_config(page_title="Clinical AI Portal", layout="wide", page_icon="🏥")

st.markdown("""
    <style>
    .report-container { border: 2px solid #1a73e8; padding: 20px; border-radius: 10px; margin-top: 20px; color: inherit; }
    .drug-card { background-color: rgba(26, 115, 232, 0.1); border-left: 5px solid #1a73e8; padding: 12px; margin-bottom: 10px; border-radius: 5px; }
    .triage-alert { background-color: #fce8e6; border: 1px solid #d93025; color: #d93025; padding: 15px; border-radius: 5px; font-weight: bold; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #1a73e8; color: white; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_ml_assets():
    try:
        model = load_model("model/model.h5")
        scaler = joblib.load("scaler.pkl")
        le = joblib.load("label_encoder.pkl")
        return model, scaler, le
    except: return None, None, None

@st.cache_data
def load_knowledge_bases():
    symptom_file = 'DiseaseAndSymptoms.csv'
    medicine_file = 'Medicine_description (2).xlsx - Sheet1.csv'
    try:
        disease_df = pd.read_csv(symptom_file, encoding='latin1', on_bad_lines='skip')
        medicine_db = pd.read_csv(medicine_file, encoding='latin1', on_bad_lines='skip')
        medicine_db.columns = [str(c).strip().replace('\xa0', '') for c in medicine_db.columns]
        disease_map = {}
        for _, row in disease_df.iterrows():
            d = str(row['Disease']).strip()
            s = [str(val).strip().lower().replace("_", " ") for val in row[1:] if pd.notna(val)]
            if d not in disease_map: disease_map[d] = set(s)
            else: disease_map[d].update(s)
        return disease_map, medicine_db
    except: return {}, pd.DataFrame()

model, scaler, le = load_ml_assets()
disease_map, medicine_db = load_knowledge_bases()

# --- 2. CORE LOGIC ---

def get_medication_recs(detected_condition):
    if medicine_db.empty or not detected_condition or "General" in str(detected_condition): return []
    query = str(detected_condition).lower().strip()
    cols = list(medicine_db.columns)
    mask = medicine_db[cols[1]].astype(str).str.contains(query, case=False, na=False)
    results = medicine_db[mask].head(5)
    return [{'Drug_Name': row[cols[0]], 'Description': row[cols[2]]} for _, row in results.iterrows()]

# --- 3. UI ---
st.title("🏥 Clinical AI: Intelligent Risk & Therapy Engine")
st.caption("M.Sc. Final Project | Dual-Engine Diagnostic Framework")
st.divider()

col_l, col_r = st.columns([1, 1], gap="large")
with col_l:
    st.subheader("👤 Patient Identity & Vitals")
    p_name = st.text_input("Full Name", "Patient")
    doc_email = st.text_input("Doctor Email")
    p_age = st.number_input("Age", 1, 120, 23)
    v1, v2 = st.columns(2)
    hr = v1.number_input("Heart Rate", 72.0)
    spo2 = v2.number_input("SpO2 %", 98.0)
    bps = v1.number_input("BP Systolic", 120.0)
    temp = v2.number_input("Temp °C", 37.0)

with col_r:
    st.subheader("📋 Clinical Presentation")
    s_input = st.text_area("Describe Symptoms")

# --- 4. EXECUTION ---
st.divider()
if st.button("RUN FULL DIAGNOSTIC & GENERATE REPORT", type="primary"):
    if model and scaler:
        # A. Diagnostic Engines
        features = ['age', 'heart_rate', 'bp_systolic', 'bp_diastolic', 'spo2', 'temp', 'cholesterol', 'glucose', 'respiratory_rate']
        raw_v = [p_age, hr, bps, 80.0, spo2, temp, 190.0, 95.0, 16.0]
        scaled = scaler.transform(pd.DataFrame([raw_v], columns=features))
        dl_disease = le.inverse_transform([np.argmax(model.predict(scaled, verbose=0))])[0]
        
        user_text = s_input.lower()
        matched_scores = {d: sum(1 for sym in s_set if sym in user_text) for d, s_set in disease_map.items()}
        res_disease = max(matched_scores, key=matched_scores.get, default="General Assessment") if any(matched_scores.values()) else "General Assessment"
        
        # B. Triage & Next Step Logic
        urgency = "IMMEDIATE ER" if spo2 < 90 or bps > 175 or temp >= 39.5 else "Urgent Consultation" if temp >= 38.0 else "Stable / Routine Follow-up"
        
        # C. Report Display
        st.markdown(f"""<div class="report-container"><h3>Clinical Diagnostic Report</h3><hr>
            <p><b>Vitals Diagnosis:</b> {dl_disease}</p>
            <p><b>Symptom Matching:</b> {res_disease}</p>
            <p><b>Clinical Status:</b> {urgency}</p></div>""", unsafe_allow_html=True)

        # D. ACTIONABLE NEXT STEPS
        st.subheader("🚨 What you should do next:")
        if "IMMEDIATE" in urgency:
            st.markdown('<div class="triage-alert">CRITICAL: Visit the nearest Emergency Room immediately. Oxygen or IV fluids may be required.</div>', unsafe_allow_html=True)
        elif "Urgent" in urgency:
            st.info("Schedule an urgent appointment with your GP today. Monitor vitals every 4 hours.")
        else:
            st.success("Continue monitoring symptoms. If symptoms persist for >3 days, consult a physician.")

        # E. Therapy
        final_cond = res_disease if res_disease != "General Assessment" else dl_disease
        meds = get_medication_recs(final_cond)
        if meds:
            st.subheader(f"💊 Therapy Recommendation: {final_cond}")
            for m in meds: 
                st.markdown(f'<div class="drug-card"><b>{m["Drug_Name"]}</b><br><small>{m["Description"]}</small></div>', unsafe_allow_html=True)

        # F. REPORT DOWNLOAD (RESTORED)
        report_text = f"Clinical AI Report\nPatient: {p_name}\nDiagnosis: {dl_disease}\nSymptom Detection: {res_disease}\nStatus: {urgency}\n\nRecommended Drugs:\n"
        for m in meds: report_text += f"- {m['Drug_Name']}\n"
        
        st.download_button(
            label="📄 Download Diagnostic Report",
            data=report_text,
            file_name=f"Clinical_Report_{p_name}.txt",
            mime="text/plain"
        )
