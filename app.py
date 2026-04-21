import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import re
import smtplib
from email.message import EmailMessage
from tensorflow.keras.models import load_model

# --- 1. CONFIG & UI STYLING ---
st.set_page_config(page_title="Clinical AI Portal", layout="wide", page_icon="🏥")

st.markdown("""
    <style>
    .report-container { border: 2px solid #1a73e8; padding: 20px; border-radius: 10px; margin-top: 20px; color: inherit; }
    .drug-card { background-color: rgba(26, 115, 232, 0.1); border-left: 5px solid #1a73e8; padding: 12px; margin-bottom: 10px; border-radius: 5px; }
    .triage-box { padding: 15px; border-radius: 5px; margin-top: 10px; font-weight: bold; border: 1px solid #ccc; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #1a73e8; color: white; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. ASSET LOADING ---
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
    # AUTO-DETECT FILENAMES to prevent naming errors
    files = os.listdir('.')
    sym_file = next((f for f in files if "DiseaseAndSymptoms" in f and f.endswith('.csv')), 'DiseaseAndSymptoms.csv')
    med_file = next((f for f in files if "Medicine_description" in f and f.endswith('.csv')), 'Medicine_description.csv')
    
    try:
        disease_df = pd.read_csv(sym_file, encoding='latin1')
        medicine_db = pd.read_csv(med_file, encoding='latin1')
        medicine_db.columns = [str(c).strip() for c in medicine_db.columns]
        
        disease_map = {}
        for _, row in disease_df.iterrows():
            d = str(row['Disease']).strip()
            # Clean symptoms: lowercase and replace underscores with spaces
            s = [str(val).strip().lower().replace("_", " ") for val in row[1:] if pd.notna(val)]
            if d not in disease_map: disease_map[d] = set(s)
            else: disease_map[d].update(s)
        return disease_map, medicine_db
    except: return {}, pd.DataFrame()

model_assets = load_ml_assets()
disease_map, medicine_db = load_knowledge_bases()

# --- 3. NLP & UTILITY FUNCTIONS ---

def clean_and_tokenize(text):
    """Filters out stop words and keeps only clinical tokens."""
    stop_words = {'i', 'have', 'having', 'with', 'a', 'the', 'is', 'am', 'are', 'feeling', 'symptoms', 'of', 'and', 'my', 'in', 'some', 'very', 'bad'}
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    tokens = text.split()
    return [w for w in tokens if w not in stop_words]

def get_medication_recs(conditions):
    if medicine_db.empty or not conditions: return []
    matches = []
    cols = list(medicine_db.columns)
    for cond in conditions:
        if not cond or cond in ["Normal", "General Assessment"]: continue
        mask = medicine_db[cols[1]].astype(str).str.contains(str(cond), case=False, na=False)
        for _, row in medicine_db[mask].head(3).iterrows():
            matches.append({'name': row[cols[0]], 'desc': row[cols[2]], 'for': cond})
    return matches

def send_to_doctor(receiver_email, report_data, meds):
    msg = EmailMessage()
    msg['Subject'] = f"🚨 {report_data['urgency']} Alert: {report_data['name']}"
    msg['From'] = st.secrets.get("EMAIL_USER", "alert@clinical-ai.com")
    msg['To'] = receiver_email
    med_text = "\n".join([f"- {m['name']} (for {m['for']})" for m in meds])
    body = f"Patient: {report_data['name']}\nAge: {report_data['age']}\nVitals Diag: {report_data['v_diag']}\nSymptom Diag: {report_data['s_diag']}\nStatus: {report_data['urgency']}\n\nTherapy:\n{med_text}"
    msg.set_content(body)
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(st.secrets["EMAIL_USER"], st.secrets["EMAIL_PASS"])
            smtp.send_message(msg)
            return True
    except: return False

# --- 4. UI HEADER ---
st.title("🏥 Clinical AI: Intelligent Risk & Therapy Engine")
st.markdown("### **M.Sc. Data Science Final Project**")
st.markdown("**Developer:** Onkar Suresh Wagh")
st.divider()

# --- 5. UI INPUTS ---
col_l, col_r = st.columns([1, 1], gap="large")
with col_l:
    st.subheader("👤 Patient Identity & Vitals")
    p_name = st.text_input("Patient Full Name", "Onkar Wagh")
    p_age = st.number_input("Age", 1, 120, 23)
    v1, v2 = st.columns(2)
    hr = v1.number_input("Heart Rate (BPM)", 72.0)
    spo2 = v2.number_input("SpO2 (%)", 98.0)
    bps = v1.number_input("Blood Pressure (Systolic)", 120.0)
    temp = v2.number_input("Temperature (°C)", 37.0)

with col_r:
    st.subheader("📋 Clinical Presentation")
    s_input = st.text_area("Describe Symptoms (e.g., 'I have joint stiffness and pain')")
    doc_email = st.text_input("Doctor Email for Alerts")

# --- 6. EXECUTION ---
st.divider()
if st.button("RUN FULL DIAGNOSTIC", type="primary"):
    if model_assets[0] is not None:
        # A. Vitals Engine
        features = ['age', 'heart_rate', 'bp_systolic', 'bp_diastolic', 'spo2', 'temp', 'cholesterol', 'glucose', 'respiratory_rate']
        raw_v = [p_age, hr, bps, 80.0, spo2, temp, 190.0, 95.0, 16.0]
        input_df = pd.DataFrame([raw_v], columns=features)
        scaled = model_assets[1].transform(input_df)
        dl_disease = model_assets[2].inverse_transform([np.argmax(model_assets[0].predict(scaled, verbose=0))])[0]
        
        # B. Symptom Engine (Enhanced Tokenization)
        tokens = clean_and_tokenize(s_input)
        matched_scores = {}
        
        if tokens:
            for d, s_set in disease_map.items():
                # Direct match for disease name
                score = 15 if d.lower() in " ".join(tokens) else 0
                # Match tokens against cleaned symptom set
                for sym in s_set:
                    # If the symptom name (e.g. 'muscle pain') is in our tokenized input
                    if any(t in sym for t in tokens):
                        score += 1
                matched_scores[d] = score
            
            res_disease = max(matched_scores, key=matched_scores.get) if any(v > 0 for v in matched_scores.values()) else "General Assessment"
        else:
            res_disease = "General Assessment"

        # C. Triage Status
        urgency = "EMERGENCY" if spo2 < 90 or bps > 180 or temp >= 39.0 else "Stable"

        # D. Display Diagnostic Summary
        st.markdown(f"""<div class="report-container">
            <h3 style='text-align: center;'>Clinical Diagnostic Report</h3><hr>
            <p><b>Predicted Disease (Vitals Engine):</b> {dl_disease}</p>
            <p><b>Detected Disease (Symptom Engine):</b> {res_disease}</p>
            <p><b>Urgency Status:</b> <span style="color:{'red' if urgency=='EMERGENCY' else 'green'}">{urgency}</span></p>
            </div>""", unsafe_allow_html=True)

        # E. Actionable Insights
        st.subheader("🛑 Actionable Insights")
        if urgency == "EMERGENCY":
            st.error("🚨 **CRITICAL:** Immediate clinical intervention required. Proceed to ER.")
        else:
            st.info("✅ **STABLE:** Routine follow-up recommended. Monitor symptoms.")

        # F. Medication Therapy
        found_conds = [d for d in [dl_disease, res_disease] if d not in ["Normal", "General Assessment"]]
        meds = get_medication_recs(found_conds)
        if meds:
            st.subheader("💊 Therapy Recommendations")
            for m in meds:
                st.markdown(f'<div class="drug-card"><b>{m["name"]}</b> (Target: {m["for"]})<br><small>{m["desc"]}</small></div>', unsafe_allow_html=True)

        # G. Report Download & Email
        st.divider()
        report_txt = f"Patient: {p_name}\nAge: {p_age}\nVitals: {dl_disease}\nSymptoms: {res_disease}\nStatus: {urgency}"
        c1, c2 = st.columns(2)
        with c1:
            st.download_button("📄 Download Clinical Report", report_txt, file_name=f"ClinicalReport_{p_name}.txt")
        with c2:
            if doc_email:
                with st.spinner("Sending Email Alert..."):
                    report_data = {'name': p_name, 'age': p_age, 'v_diag': dl_disease, 's_diag': res_disease, 'urgency': urgency}
                    if send_to_doctor(doc_email, report_data, meds):
                        st.success(f"Alert successfully sent to {doc_email}!")
    else:
        st.error("Missing ML Model files (model.h5) in the 'model/' directory.")
