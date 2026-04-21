import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import smtplib
from email.message import EmailMessage
from tensorflow.keras.models import load_model

# --- 1. CONFIG & ASSETS ---
st.set_page_config(page_title="Clinical AI Portal", layout="wide", page_icon="🏥")

# Professional Styles
st.markdown("""
    <style>
    .report-container { border: 2px solid #1a73e8; padding: 20px; border-radius: 10px; margin-top: 20px; color: inherit; }
    .drug-card { background-color: rgba(26, 115, 232, 0.1); border-left: 5px solid #1a73e8; padding: 12px; margin-bottom: 10px; border-radius: 5px; }
    .triage-box { padding: 15px; border-radius: 5px; margin-top: 10px; font-weight: bold; }
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
    medicine_file = 'Medicine_description.xlsx'
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

model_assets = load_ml_assets()
disease_map, medicine_db = load_knowledge_bases()

# --- 2. LOGIC FUNCTIONS ---

def get_medication_recs(conditions):
    if medicine_db.empty or not conditions: return []
    matches = []
    cols = list(medicine_db.columns)
    for cond in conditions:
        if not cond or "General" in str(cond) or "Normal" in str(cond): continue
        mask = medicine_db[cols[1]].astype(str).str.contains(str(cond), case=False, na=False)
        for _, row in medicine_db[mask].head(3).iterrows():
            matches.append({'name': row[cols[0]], 'desc': row[cols[2]], 'for': cond})
    return matches

# --- 3. UI - HEADER ---
st.title("🏥 Clinical AI: Intelligent Risk & Therapy Engine")
st.subheader("Project: Dual-Engine Disease Diagnostic System")
st.markdown("Onkar Suresh Wagh | M.Sc. Data Science Final year project")
st.divider()

# --- 4. INPUTS ---
col_l, col_r = st.columns([1, 1], gap="large")
with col_l:
    st.subheader("👤 Patient Vitals")
    p_name = st.text_input("Patient Full Name", "Onkar Wagh")
    p_age = st.number_input("Age", 1, 120, 23)
    v1, v2 = st.columns(2)
    hr = v1.number_input("Heart Rate", 72.0)
    spo2 = v2.number_input("SpO2 %", 98.0)
    bps = v1.number_input("BP Systolic", 120.0)
    temp = v2.number_input("Temp °C", 37.0)

with col_r:
    st.subheader("📋 Clinical Presentation")
    s_input = st.text_area("Enter Symptoms (e.g., 'joint stiffness', 'fatigue', 'high fever')")
    doc_email = st.text_input("Doctor Email for Alerts")

# --- 5. EXECUTION ---
st.divider()
if st.button("RUN FULL DIAGNOSTIC & GENERATE REPORT", type="primary"):
    if model_assets[0] is not None:
        # A. Engines
        features = ['age', 'heart_rate', 'bp_systolic', 'bp_diastolic', 'spo2', 'temp', 'cholesterol', 'glucose', 'respiratory_rate']
        raw_v = [p_age, hr, bps, 80.0, spo2, temp, 190.0, 95.0, 16.0]
        scaled = model_assets[1].transform(pd.DataFrame([raw_v], columns=features))
        dl_disease = model_assets[2].inverse_transform([np.argmax(model_assets[0].predict(scaled, verbose=0))])[0]
        
        user_text = s_input.lower()
        matched_scores = {d: sum(1 for sym in s_set if sym in user_text) for d, s_set in disease_map.items()}
        # Added: Priority for direct name mentions
        for d in disease_map:
            if d.lower() in user_text: matched_scores[d] += 5
        res_disease = max(matched_scores, key=matched_scores.get, default="General Assessment") if any(matched_scores.values()) else "General Assessment"
        
        # B. Triage
        urgency = "EMERGENCY" if spo2 < 90 or bps > 180 or temp >= 39.5 else "Stable"
        
        # C. Display Diagnostic Report
        st.markdown(f"""<div class="report-container"><h3>Clinical Diagnostic Report</h3><hr>
            <p><b>Vitals Engine Diagnosis:</b> {dl_disease}</p>
            <p><b>Symptom Engine Detection:</b> {res_disease}</p>
            <p><b>Current Status:</b> <span style="color: {'red' if urgency=='EMERGENCY' else 'green'}">{urgency}</span></p></div>""", unsafe_allow_html=True)

        # D. NEXT STEPS
        st.subheader("🛑 Next Steps & Action Plan")
        if urgency == "EMERGENCY":
            st.error("🚨 IMMEDIATE ACTION REQUIRED: Please visit the nearest Emergency Room.")
        else:
            st.info("💡 Continue monitoring vitals. Consult a general physician if symptoms persist.")

        # E. THERAPY
        found_conds = []
        if dl_disease != "Normal": found_conds.append(dl_disease)
        if res_disease != "General Assessment": found_conds.append(res_disease)
        
        meds = get_medication_recs(found_conds)
        if meds:
            st.subheader("💊 Medication Recommendations")
            for m in meds:
                st.markdown(f'<div class="drug-card"><b>{m["name"]}</b> (for {m["for"]})<br><small>{m["desc"]}</small></div>', unsafe_allow_html=True)

        # F. DOWNLOAD REPORT
        report_data = f"CLINICAL AI REPORT\nDeveloper: Onkar Wagh\nPatient: {p_name}\nVitals Diagnosis: {dl_disease}\nSymptom Detection: {res_disease}\nStatus: {urgency}\n\nSuggested Meds:\n"
        for m in meds: report_data += f"- {m['name']} ({m['for']})\n"
        
        st.download_button(label="📥 Download Report", data=report_data, file_name=f"Report_{p_name}.txt")
    else:
        st.error("Model assets not found.")
