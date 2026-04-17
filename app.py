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
    """Loads Deep Learning assets: Model, Scaler, and Label Encoder."""
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
    """Loads CSVs using 'latin1' to handle Excel encoding issues."""
    symptom_file = 'DiseaseAndSymptoms.csv'
    medicine_file = 'Medicine_description.xlsx'
    
    if not os.path.exists(symptom_file) or not os.path.exists(medicine_file):
        st.error("❌ CSV files missing from project directory.")
        return {}, pd.DataFrame()

    try:
        # Encoding fix for the 0xa0 error
        disease_df = pd.read_csv(symptom_file, encoding='latin1')
        medicine_db = pd.read_csv(medicine_file, encoding='latin1')
        
        # Build Symptom Map: {Disease: {symptom_set}}
        disease_map = {}
        for _, row in disease_df.iterrows():
            d = str(row['Disease']).strip()
            # Remove underscores to match natural user input
            s = [str(val).strip().lower().replace("_", " ") for val in row[1:] if pd.notna(val)]
            if d not in disease_map:
                disease_map[d] = set(s)
            else:
                disease_map[d].update(s)
        return disease_map, medicine_db
    except Exception as e:
        st.error(f"Database Loading Error: {e}")
        return {}, pd.DataFrame()

# Initialize Global Assets
model, scaler, le = load_ml_assets()
disease_map, medicine_db = load_knowledge_bases()

# --- 2. CORE LOGIC FUNCTIONS ---

def send_to_doctor(receiver_email, report, drug_list):
    """Sends a detailed clinical alert email."""
    msg = EmailMessage()
    msg['Subject'] = f"🚨 {report['urgency']} Alert: {report['name']}"
    msg['From'] = st.secrets["EMAIL_USER"]
    msg['To'] = receiver_email
    
    # Format drug recommendations for the email body
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
    {med_text if med_text else "No specific drugs identified."}
    
    VITALS:
    - HR: {report['vitals'][1]} | SpO2: {report['vitals'][4]}% | BP: {report['vitals'][2]}/80
    ---------------------------
    Generated via Clinical AI Decision Support.
    """
    msg.set_content(body)
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(st.secrets["EMAIL_USER"], st.secrets["EMAIL_PASS"])
            smtp.send_message(msg)
        return True
    except: return False

def get_medication_recs(disease_name):
    """Filters 22,000+ drug records for the specific disease."""
    if not disease_name or medicine_db.empty: return []
    mask = medicine_db['Reason'].str.contains(disease_name[:5], case=False, na=False)
    return medicine_db[mask][['Drug_Name', 'Description']].head(5).to_dict('records')

# --- 3. UI LAYOUT ---

st.title("🏥 Clinical AI: Risk & Therapy System")
st.divider()

col_l, col_r = st.columns([1, 1])

with col_l:
    st.subheader("👤 Patient Profile & Vitals")
    p_name = st.text_input("Full Name")
    doc_email = st.text_input("Doctor's Email (for alerts)")
    p_age = st.number_input("Age", 1, 120, 30)
    
    v1, v2 = st.columns(2)
    hr = v1.number_input("Heart Rate", value=72.0)
    spo2 = v2.number_input("SpO2 (%)", value=98.0)
    bps = v1.number_input("BP Systolic", value=120.0)
    temp = v2.number_input("Temp (°C)", value=37.0)

with col_r:
    st.subheader("💊 Therapy Recommendation Engine")
    s_input = st.text_area("Known Symptoms", placeholder="e.g., skin rash, itching")
    
    # Process symptoms in real-time
    user_s = [s.strip().lower() for s in s_input.split(",")] if s_input else []
    scores = {d: len(set(user_s).intersection(s_set)) for d, s_set in disease_map.items()}
    res_disease = max(scores, key=scores.get, default=None) if any(scores.values()) else None
    
    if res_disease:
        st.info(f"**Potential Diagnosis:** {res_disease}")
        med_recs = get_medication_recs(res_disease)
        if med_recs:
            for m in med_recs:
                with st.expander(f"Drug: {m['Drug_Name']}"):
                    st.write(m['Description'])

# --- 4. FINAL EXECUTION ---
st.divider()
if st.button("RUN FULL DIAGNOSTIC & SEND ALERT", type="primary", use_container_width=True):
    if model and scaler:
        # Prepare Data
        raw_v = [p_age, hr, bps, 80.0, spo2, temp, 190.0, 95.0, 16.0]
        scaled = scaler.transform([raw_v])
        
        # Prediction
        pred = model.predict(scaled, verbose=0)
        idx = np.argmax(pred)
        disease = le.inverse_transform([idx])[0]
        prob = pred[0][idx] * 100
        
        # Triage Logic
        urgency = "IMMEDIATE ER" if spo2 < 90 or bps > 175 else "Stable"
        
        report = {
            'name': p_name if p_name else "Unknown",
            'age': p_age,
            'disease': disease,
            'prob': f"{prob:.2f}",
            'urgency': urgency,
            'vitals': raw_v,
            'symptom_disease': res_disease if res_disease else "None"
        }
        
        # Results
        st.header(f"AI Result: {disease} ({prob:.2f}%)")
        st.write(f"**Triage:** {urgency}")
        
        if doc_email:
            with st.spinner("Sending medical alert..."):
                current_meds = get_medication_recs(res_disease) if res_disease else []
                if send_to_doctor(doc_email, report, current_meds):
                    st.success(f"Full Clinical Report sent to {doc_email}! ✅")
    else:
        st.error("Deep Learning Assets Missing.")
