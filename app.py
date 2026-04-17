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

@st.cache_resource
def load_ml_assets():
    """Loads Deep Learning model and pre-processing scalers."""
    try:
        model = load_model("model/model.h5")
        scaler = joblib.load("scaler.pkl")
        le = joblib.load("label_encoder.pkl")
        return model, scaler, le
    except Exception as e:
        st.warning(f"ML Assets (model.h5/scaler.pkl) not found: {e}")
        return None, None, None

@st.cache_data
def load_knowledge_bases():
    """Loads CSVs with robust error handling for encoding and malformed rows."""
    symptom_file = 'DiseaseAndSymptoms.csv'
    medicine_file = 'Medicine_description.xlsx'
    
    if not os.path.exists(symptom_file) or not os.path.exists(medicine_file):
        st.error("❌ Critical Error: CSV files missing from project directory.")
        return {}, pd.DataFrame()

    try:
        # Using engine='python' and on_bad_lines='skip' to fix Buffer Overflow
        # Using encoding='latin1' to fix the 0xa0 decoding error
        disease_df = pd.read_csv(symptom_file, encoding='latin1', on_bad_lines='skip', engine='python')
        medicine_db = pd.read_csv(medicine_file, encoding='latin1', on_bad_lines='skip', engine='python')
        
        disease_map = {}
        for _, row in disease_df.iterrows():
            d = str(row['Disease']).strip()
            # Clean underscores so 'skin_rash' matches user input 'skin rash'
            s = [str(val).strip().lower().replace("_", " ") for val in row[1:] if pd.notna(val)]
            if d not in disease_map:
                disease_map[d] = set(s)
            else:
                disease_map[d].update(s)
        return disease_map, medicine_db
    except Exception as e:
        st.error(f"Database Loading Error: {e}")
        return {}, pd.DataFrame()

# Global Initialization
model, scaler, le = load_ml_assets()
disease_map, medicine_db = load_knowledge_bases()

# --- 2. CORE LOGIC FUNCTIONS ---

def send_to_doctor(receiver_email, report, drug_list):
    """Sends a detailed clinical alert email."""
    msg = EmailMessage()
    msg['Subject'] = f"🚨 {report['urgency']} Alert: {report['name']}"
    msg['From'] = st.secrets["EMAIL_USER"]
    msg['To'] = receiver_email
    
    med_text = "\n".join([f"- {m['Drug_Name']}: {m['Description'][:100]}..." for m in drug_list])
    
    body = f"""
    CLINICAL DIAGNOSTIC SUMMARY
    ---------------------------
    PATIENT: {report['name']} ({report['age']} years)
    URGENCY: {report['urgency']}
    
    AI PREDICTION:
    - Result: {report['disease']}
    - Confidence: {report['prob']}%
    
    SYMPTOM ANALYSIS:
    - Likely Condition: {report['symptom_disease']}
    
    RECOMMENDED THERAPY:
    {med_text if med_text else "No drugs identified."}
    
    VITALS:
    - HR: {report['vitals'][1]} | SpO2: {report['vitals'][4]}% | Temp: {report['vitals'][5]}°C
    ---------------------------
    Generated via Onkar Wagh Clinical AI System.
    """
    msg.set_content(body)
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(st.secrets["EMAIL_USER"], st.secrets["EMAIL_PASS"])
            smtp.send_message(msg)
        return True
    except: return False

def get_medication_recs(disease_name):
    """Filters medicine database based on the predicted reason."""
    if not disease_name or medicine_db.empty: return []
    mask = medicine_db['Reason'].str.contains(disease_name[:5], case=False, na=False)
    return medicine_db[mask][['Drug_Name', 'Description']].head(5).to_dict('records')

# --- 3. UI LAYOUT ---

st.title("🏥 Intelligent Clinical Risk & Therapy System")
st.divider()

col_l, col_r = st.columns([1, 1])

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
    st.subheader("💊 Therapy Recommendation Engine")
    s_input = st.text_area("Enter Symptoms (comma separated)", placeholder="e.g. itching, skin rash")
    
    # Process symptoms
    user_s = [s.strip().lower() for s in s_input.split(",")] if s_input else []
    scores = {d: len(set(user_s).intersection(s_set)) for d, s_set in disease_map.items()}
    res_disease = max(scores, key=scores.get, default=None) if any(scores.values()) else None
    
    if res_disease:
        st.info(f"**Detected Condition:** {res_disease}")
        med_recs = get_medication_recs(res_disease)
        if med_recs:
            for m in med_recs:
                with st.expander(f"Drug: {m['Drug_Name']}"):
                    st.write(m['Description'])

# --- 4. EXECUTION ---
st.divider()
if st.button("RUN FULL DIAGNOSTIC & NOTIFY DOCTOR", type="primary", use_container_width=True):
    if model and scaler:
        # Prediction Pipeline
        raw_v = [p_age, hr, bps, 80.0, spo2, temp, 190.0, 95.0, 16.0]
        scaled = scaler.transform([raw_v])
        pred = model.predict(scaled, verbose=0)
        idx = np.argmax(pred)
        disease = le.inverse_transform([idx])[0]
        prob = pred[0][idx] * 100
        
        # Triage logic
        urgency = "IMMEDIATE ER" if spo2 < 90 or bps > 175 else "Stable"
        
        report = {
            'name': p_name if p_name else "Unknown",
            'age': p_age,
            'disease': disease,
            'prob': f"{prob:.2f}",
            'urgency': urgency,
            'vitals': raw_v,
            'symptom_disease': res_disease if res_disease else "None detected"
        }
        
        # Results Display
        st.header(f"AI Result: {disease} ({prob:.2f}%)")
        st.write(f"**Urgency:** {urgency}")
        
        # Email Notification
        if doc_email:
            with st.spinner("Sending medical alert..."):
                meds_for_email = get_medication_recs(res_disease) if res_disease else []
                if send_to_doctor(doc_email, report, meds_for_email):
                    st.success(f"Full Report sent to {doc_email}! ✅")
    else:
        st.error("Deep Learning Assets Missing (Check model.h5/scaler.pkl).")
