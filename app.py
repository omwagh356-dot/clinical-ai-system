import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import smtplib
from email.message import EmailMessage
from tensorflow.keras.models import load_model

# --- 1. CONFIG & ASSETS LOADING ---
st.set_page_config(page_title="Clinical AI Portal", layout="wide", page_icon="🏥")

# Theme-neutral CSS for Dark/Light mode visibility
st.markdown("""
    <style>
    .report-container { border: 2px solid #1a73e8; padding: 20px; border-radius: 10px; margin-top: 20px; color: inherit; }
    .drug-card { background-color: rgba(26, 115, 232, 0.1); border-left: 5px solid #1a73e8; padding: 10px; margin-bottom: 10px; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_ml_assets():
    """Loads Deep Learning assets."""
    try:
        model = load_model("model/model.h5")
        scaler = joblib.load("scaler.pkl")
        le = joblib.load("label_encoder.pkl")
        return model, scaler, le
    except Exception as e:
        st.warning(f"ML Assets not found: {e}")
        return None, None, None

@st.cache_data
def load_knowledge_bases():
    """Loads CSVs with robust error handling for encoding."""
    symptom_file = 'DiseaseAndSymptoms.csv'
    medicine_file = 'Medicine_description.xlsx'
    
    if not os.path.exists(symptom_file) or not os.path.exists(medicine_file):
        st.error("❌ Critical Error: CSV files missing.")
        return {}, pd.DataFrame()

    try:
        # engine='python' handles malformed rows; encoding='latin1' handles 0xa0 characters
        disease_df = pd.read_csv(symptom_file, encoding='latin1', on_bad_lines='skip', engine='python')
        medicine_db = pd.read_csv(medicine_file, encoding='latin1', on_bad_lines='skip', engine='python')
        medicine_db.columns = medicine_db.columns.str.strip()
        
        disease_map = {}
        for _, row in disease_df.iterrows():
            d = str(row['Disease']).strip()
            s = [str(val).strip().lower().replace("_", " ") for val in row[1:] if pd.notna(val)]
            if d not in disease_map:
                disease_map[d] = set(s)
            else:
                disease_map[d].update(s)
        return disease_map, medicine_db
    except Exception as e:
        st.error(f"Database Loading Error: {e}")
        return {}, pd.DataFrame()

# Initialize
model, scaler, le = load_ml_assets()
disease_map, medicine_db = load_knowledge_bases()

# --- 2. CORE LOGIC FUNCTIONS ---

def get_medication_recs(search_term):
    """Filters medicine database using the predicted disease or condition."""
    if not search_term or medicine_db.empty: return []
    cols = medicine_db.columns
    target_col = 'Reason' 
    if 'Reason' not in cols:
        possible = [c for c in cols if 'reason' in c.lower() or 'disease' in c.lower()]
        if possible: target_col = possible[0]
        else: return []
    
    mask = medicine_db[target_col].str.contains(search_term[:5], case=False, na=False)
    return medicine_db[mask][['Drug_Name', 'Description']].head(5).to_dict('records')

# --- 3. UI LAYOUT ---
st.title("🏥 Intelligent Clinical Risk & Therapy System")
st.caption("Onkar Suresh Wagh - Master of Science Data Science 2026 [cite: 17]")
st.divider()

col_l, col_r = st.columns([1, 1], gap="large")

with col_l:
    st.subheader("👤 Patient Identity & Vitals")
    p_name = st.text_input("Patient Full Name")
    doc_email = st.text_input("Doctor Email for Alerts")
    p_age = st.number_input("Age", 1, 120, 30)
    v1, v2 = st.columns(2)
    hr = v1.number_input("Heart Rate", value=72.0)
    spo2 = v2.number_input("SpO2 %", value=98.0)
    bps = v1.number_input("BP Systolic", value=120.0)
    temp = v2.number_input("Temp °C", value=37.0)

with col_r:
    st.subheader("📋 Known Condition / Symptoms")
    s_input = st.text_area("Enter Symptoms", placeholder="e.g. skin rash, chest pain")

# --- 4. INTEGRATED EXECUTION ---
st.divider()
if st.button("RUN FULL DIAGNOSTIC & RECOMMEND THERAPY", type="primary", use_container_width=True):
    if model and scaler:
        # A. Process Vitals through Deep Learning Engine [cite: 215, 356]
        raw_v = [p_age, hr, bps, 80.0, spo2, temp, 190.0, 95.0, 16.0]
        scaled = scaler.transform([raw_v])
        pred = model.predict(scaled, verbose=0)
        idx = np.argmax(pred)
        dl_disease = le.inverse_transform([idx])[0]
        prob = pred[0][idx] * 100
        
        # B. Process Symptoms through Rule-Based Engine [cite: 405, 463]
        user_s = [s.strip().lower() for s in s_input.split(",")] if s_input else []
        scores = {d: len(set(user_s).intersection(s_set)) for d, s_set in disease_map.items()}
        symptom_disease = max(scores, key=scores.get, default=None) if any(scores.values()) else "Normal"
        
        # C. Triage Urgency
        urgency = "IMMEDIATE ER" if spo2 < 90 or bps > 175 else "Stable"

        # D. Combined Diagnosis Display [cite: 509]
        st.markdown(f"""
            <div class="report-container">
                <h2>Clinical Diagnostic Result</h2>
                <hr>
                <p><b>Deep Learning Prediction (from Vitals):</b> {dl_disease} ({prob:.2f}%)</p>
                <p><b>Symptom Matching Result:</b> {symptom_disease}</p>
                <p><b>Triage Urgency:</b> <span style="color: {'red' if urgency != 'Stable' else 'green'};">{urgency}</span></p>
            </div>
            """, unsafe_allow_html=True)

        # E. Therapy Recommendations based on Vitals-Prediction [cite: 466]
        st.subheader("💊 Vitals-Based Recommended Therapy")
        # We prioritize recommending drugs for the disease predicted from vitals
        vitals_meds = get_medication_recs(dl_disease)
        
        if vitals_meds:
            for m in vitals_meds:
                st.markdown(f"""
                    <div class="drug-card">
                        <b>{m['Drug_Name']}</b><br>
                        <i>Indication: {m['Description']}</i>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("No specific medications found for the predicted vital-sign condition.")
            
    else:
        st.error("Deep Learning Assets Missing.")
