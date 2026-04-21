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

st.markdown("""
    <style>
    .report-container { border: 2px solid #1a73e8; padding: 20px; border-radius: 10px; margin-top: 20px; color: inherit; }
    .drug-card { background-color: rgba(26, 115, 232, 0.1); border-left: 5px solid #1a73e8; padding: 12px; margin-bottom: 10px; border-radius: 5px; }
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

model, scaler, le = load_ml_assets()
disease_map, medicine_db = load_knowledge_bases()

# --- 2. CORE LOGIC ---

def get_medication_recs(condition_list):
    """Fetches drugs for a list of conditions found by either engine."""
    if medicine_db.empty or not condition_list:
        return []
    
    matches = []
    cols = list(medicine_db.columns)
    
    for condition in condition_list:
        if not condition or "General" in str(condition) or "Normal" in str(condition):
            continue
            
        query = str(condition).lower().strip()
        mask = medicine_db[cols[1]].astype(str).str.contains(query, case=False, na=False)
        results = medicine_db[mask].head(3)
        
        for _, row in results.iterrows():
            matches.append({'Drug_Name': row[cols[0]], 'Description': row[cols[2]], 'For': condition})
            
    return matches

# --- 3. UI ---
st.title("🏥 Clinical AI: Intelligent Risk & Therapy Engine")
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
    s_input = st.text_area("Describe Symptoms (e.g., 'muscle pain, joint stiffness')")

# --- 4. EXECUTION ---
st.divider()
if st.button("RUN FULL DIAGNOSTIC", type="primary"):
    if model and scaler:
        # A. Vitals Engine (Predicted by ML)
        features = ['age', 'heart_rate', 'bp_systolic', 'bp_diastolic', 'spo2', 'temp', 'cholesterol', 'glucose', 'respiratory_rate']
        raw_v = [p_age, hr, bps, 80.0, spo2, temp, 190.0, 95.0, 16.0]
        input_df = pd.DataFrame([raw_v], columns=features)
        scaled = scaler.transform(input_df)
        pred = model.predict(scaled, verbose=0)
        dl_disease = le.inverse_transform([np.argmax(pred)])[0]
        prob = np.max(pred) * 100
        
        # B. Symptom Engine (Predicted by Symptoms)
        user_text = s_input.lower()
        matched_scores = {d: sum(1 for sym in s_set if sym in user_text) for d, s_set in disease_map.items()}
        res_disease = max(matched_scores, key=matched_scores.get, default="General Assessment") if any(matched_scores.values()) else "General Assessment"
        
        # C. Triage Status
        urgency = "IMMEDIATE ER" if spo2 < 90 or bps > 175 or temp >= 38.5 else "Stable"

        # D. Display Results
        st.markdown(f"""<div class="report-container"><h2 style='text-align: center;'>Clinical Diagnostic Report</h2><hr>
            <p><b>AI Diagnosis (from Vitals):</b> {dl_disease} ({prob:.2f}%)</p>
            <p><b>Symptom Detection (from Text):</b> {res_disease}</p>
            <p><b>Status:</b> <span style="color: {'red' if urgency != 'Stable' else 'green'}; font-weight: bold;">{urgency}</span></p></div>""", unsafe_allow_html=True)

        # E. Independent Therapy Logic
        # It collects findings from BOTH engines and searches for meds for both
        conditions_found = []
        if dl_disease != "Normal": conditions_found.append(dl_disease)
        if res_disease != "General Assessment": conditions_found.append(res_disease)
        
        meds = get_medication_recs(conditions_found)
        
        if meds:
            st.subheader("💊 Combined Therapy Recommendations")
            for m in meds: 
                st.markdown(f"""<div class="drug-card">
                    <b>{m['Drug_Name']}</b> (Target: {m['For']})<br>
                    <small>{m['Description']}</small>
                </div>""", unsafe_allow_html=True)
        else:
            st.info("No specific medications matched the detected conditions.")
    else:
        st.error("Assets missing.")
