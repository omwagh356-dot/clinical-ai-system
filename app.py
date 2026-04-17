import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
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
    """Loads CSVs and cleans symptom strings for better matching."""
    symptom_file = 'DiseaseAndSymptoms.csv'
    medicine_file = 'Medicine_description.xlsx'
    
    if not os.path.exists(symptom_file) or not os.path.exists(medicine_file):
        st.error("❌ Critical Error: CSV files missing.")
        return {}, pd.DataFrame()

    try:
        disease_df = pd.read_csv(symptom_file, encoding='latin1', on_bad_lines='skip', engine='python')
        medicine_db = pd.read_csv(medicine_file, encoding='latin1', on_bad_lines='skip', engine='python')
        medicine_db.columns = medicine_db.columns.str.strip()
        
        disease_map = {}
        for _, row in disease_df.iterrows():
            d = str(row['Disease']).strip()
            # Clean underscores so 'skin_rash' in CSV matches 'skin rash' in user text
            s = [str(val).strip().lower().replace("_", " ") for val in row[1:] if pd.notna(val)]
            if d not in disease_map:
                disease_map[d] = set(s)
            else:
                disease_map[d].update(s)
        return disease_map, medicine_db
    except Exception as e:
        st.error(f"Database Loading Error: {e}")
        return {}, pd.DataFrame()

model, scaler, le = load_ml_assets()
disease_map, medicine_db = load_knowledge_bases()

# --- 2. CORE LOGIC FUNCTIONS ---

def get_medication_recs(search_term):
    """Filters database for specific drug recommendations."""
    if not search_term or medicine_db.empty: return []
    cols = medicine_db.columns
    target_col = 'Reason' if 'Reason' in cols else 'Disease'
    
    # Use partial matching to bridge slight variations
    mask = medicine_db[target_col].str.contains(search_term[:5], case=False, na=False)
    return medicine_db[mask][['Drug_Name', 'Description']].head(5).to_dict('records')

# --- 3. UI LAYOUT ---
st.title("🏥 Intelligent Clinical Risk & Therapy System")
st.caption("Final Year Project - Onkar Suresh Wagh")
st.divider()

col_l, col_r = st.columns([1, 1], gap="large")

with col_l:
    st.subheader("👤 Patient Identity & Vitals")
    p_name = st.text_input("Patient Full Name")
    p_age = st.number_input("Age", 1, 120, 30)
    v1, v2 = st.columns(2)
    hr = v1.number_input("Heart Rate", value=72.0)
    spo2 = v2.number_input("SpO2 %", value=98.0)
    bps = v1.number_input("BP Systolic", value=120.0)
    temp = v2.number_input("Temp °C", value=37.0)

with col_r:
    st.subheader("📋 Clinical Presentation / Symptoms")
    s_input = st.text_area("Enter Symptoms (Natural Language)", placeholder="e.g. I have a very bad fever and itchy skin rash")

# --- 4. SMART MATCHING EXECUTION BLOCK ---
st.divider()
if st.button("RUN FULL DIAGNOSTIC & RECOMMEND THERAPY", type="primary", use_container_width=True):
    if model and scaler:
        # A. Vitals Processing
        raw_v = [p_age, hr, bps, 80.0, spo2, temp, 190.0, 95.0, 16.0]
        scaled = scaler.transform([raw_v])
        pred = model.predict(scaled, verbose=0)
        idx = np.argmax(pred)
        dl_disease = le.inverse_transform([idx])[0]
        prob = pred[0][idx] * 100
        
        # B. SMART SYMPTOM ENGINE
        user_text = s_input.lower()
        matched_scores = {}
        for disease, symptoms_set in disease_map.items():
            count = 0
            for sym in symptoms_set:
                if sym in user_text: # Checks if known symptom is part of user sentence
                    count += 1
            if count > 0:
                matched_scores[disease] = count
        
        # Pick the disease with the most symptom matches
        res_disease = max(matched_scores, key=matched_scores.get, default="Normal")
        
        # C. Triage Urgency
        urgency = "IMMEDIATE ER" if spo2 < 90 or bps > 175 or temp >= 39.5 else "Stable"

        # D. Display Professional Report
        st.markdown(f"""
            <div class="report-container">
                <h2 style='text-align: center;'>Clinical Diagnostic Report</h2>
                <hr>
                <p><b>AI Vital Analysis:</b> {dl_disease} ({prob:.2f}%)</p>
                <p><b>Smart Symptom Detection:</b> {res_disease}</p>
                <p><b>Triage Urgency:</b> <span style="color: {'red' if urgency != 'Stable' else 'green'}; font-weight: bold;">{urgency}</span></p>
            </div>
            """, unsafe_allow_html=True)

        # E. Therapy Recommendations
        st.subheader("💊 Recommended Therapy")
        # Prioritize therapy for the detected symptom condition or vitals prediction
        search_query = res_disease if res_disease != "Normal" else dl_disease
        meds = get_medication_recs(search_query)
        
        if meds:
            for m in meds:
                st.markdown(f"""
                    <div class="drug-card">
                        <b>{m['Drug_Name']}</b><br>
                        <small><i>{m['Description']}</i></small>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("No specific medications found in the database for the identified condition.")
            
    else:
        st.error("Deep Learning Assets Missing.")
