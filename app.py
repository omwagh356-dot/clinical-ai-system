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
    # Detects your specific uploaded filenames automatically
    files = os.listdir('.')
    sym_file = next((f for f in files if "DiseaseAndSymptoms" in f), 'DiseaseAndSymptoms.csv')
    med_file = next((f for f in files if "Medicine_description" in f), 'Medicine_description.csv')
    
    try:
        disease_df = pd.read_csv(sym_file, encoding='latin1')
        medicine_db = pd.read_csv(med_file, encoding='latin1')
        medicine_db.columns = [str(c).strip() for c in medicine_db.columns]
        
        disease_map = {}
        for _, row in disease_df.iterrows():
            d = str(row['Disease']).strip()
            # NLP Fix: Convert 'joint_stiffness' to 'joint stiffness'
            s = [str(val).strip().lower().replace("_", " ") for val in row[1:] if pd.notna(val)]
            if d not in disease_map: disease_map[d] = set(s)
            else: disease_map[d].update(s)
        return disease_map, medicine_db
    except: return {}, pd.DataFrame()

model_assets = load_ml_assets()
disease_map, medicine_db = load_knowledge_bases()

# --- 3. CORE LOGIC FUNCTIONS ---

def get_clean_tokens(text):
    """Filters out non-clinical filler words for better matching"""
    stop_words = {'i', 'am', 'have', 'having', 'with', 'a', 'the', 'is', 'feeling', 'symptoms', 'of', 'and', 'my', 'in', 'very', 'bad'}
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return [w for w in text.split() if w not in stop_words]

def send_to_doctor(receiver_email, report, meds):
    """Restored SMTP Email Function"""
    msg = EmailMessage()
    msg['Subject'] = f"🚨 {report['urgency']} Alert: {report['name']}"
    msg['From'] = st.secrets.get("EMAIL_USER", "clinical-ai@portal.com")
    msg['To'] = receiver_email
    
    med_text = "\n".join([f"- {m['name']} (for {m['for']})" for m in meds])
    body = f"Patient: {report['name']}\nVitals Diag: {report['v_diag']} ({report['v_prob']})\nSymptom Diag: {report['s_diag']}\nStatus: {report['urgency']}\n\nTherapy:\n{med_text}"
    msg.set_content(body)
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(st.secrets["EMAIL_USER"], st.secrets["EMAIL_PASS"])
            smtp.send_message(msg); return True
    except: return False

# --- 4. UI HEADER ---
st.title("🏥 Clinical AI: Intelligent Risk & Therapy Engine")
st.markdown("### **M.Sc. Data Science Final Project**")
st.markdown("**Developer:** Onkar Suresh Wagh")
st.divider()

# --- 5. INPUTS ---
col_l, col_r = st.columns([1, 1], gap="large")
with col_l:
    st.subheader("👤 Patient Identity & Vitals")
    p_name = st.text_input("Full Name")
    p_age = st.number_input("Age", min_value=1, max_value=120, value=23)
    
    v1, v2 = st.columns(2)
    
    # Heart Rate (Ensure all values have .0)
    hr = v1.number_input("Heart Rate (BPM)", min_value=30.0, max_value=250.0, value=72.0)
    
    # SpO2
    spo2 = v2.number_input("SpO2 (%)", min_value=50.0, max_value=100.0, value=98.0)
    
    # BP Systolic
    bps = v1.number_input("BP Systolic (mmHg)", min_value=50.0, max_value=250.0, value=120.0)
    
    # Temperature
    temp = v2.number_input("Temperature (°C)", min_value=30.0, max_value=45.0, value=37.0)
with col_r:
    st.subheader("📋 Clinical Presentation")
    s_input = st.text_area("Describe Symptoms (e.g. 'I have joint stiffness')")
    doc_email = st.text_input("Doctor Email for Alerts")

# --- 6. EXECUTION ---
st.divider()
if st.button("RUN FULL DIAGNOSTIC & NOTIFY DOCTOR", type="primary"):
    if model_assets[0] is not None:
        # A. Vitals Engine (ML)
        features = ['age', 'heart_rate', 'bp_systolic', 'bp_diastolic', 'spo2', 'temp', 'cholesterol', 'glucose', 'respiratory_rate']
        raw_v = [p_age, hr, bps, 80.0, spo2, temp, 190.0, 95.0, 16.0]
        scaled = model_assets[1].transform(pd.DataFrame([raw_v], columns=features))
        
        preds = model_assets[0].predict(scaled, verbose=0)
        idx = np.argmax(preds)
        dl_disease = model_assets[2].inverse_transform([idx])[0]
        prob = preds[0][idx] * 100  # RESTORED PERCENTAGE
        
        # B. Symptom Engine (NLP)
        tokens = get_clean_tokens(s_input)
        matched_scores = {d: (15 if d.lower() in " ".join(tokens) else 0) + 
                          sum(1 for token in tokens for sym in s_set if token in sym) 
                          for d, s_set in disease_map.items()}
        res_disease = max(matched_scores, key=matched_scores.get) if any(v > 0 for v in matched_scores.values()) else "General Assessment"

        # C. Triage
        urgency = "EMERGENCY" if spo2 < 90 or bps > 180 or temp >= 39.0 else "Stable"

        # D. REPORT DISPLAY
        st.markdown(f"""<div class="report-container">
            <h3 style='text-align: center;'>Clinical Diagnostic Report</h3><hr>
            <p><b>Vitals Prediction:</b> {dl_disease} ({prob:.2f}%)</p>
            <p><b>Symptom Detection:</b> {res_disease}</p>
            <p><b>Current Status:</b> <span style="color:{'red' if urgency=='EMERGENCY' else 'green'}">{urgency}</span></p>
            </div>""", unsafe_allow_html=True)

        # E. ACTIONABLE INSIGHTS (NEXT STEPS)
        st.subheader("🛑 Clinical Action Plan")
        if urgency == "EMERGENCY":
            st.error("🚨 CRITICAL: Immediate intervention required. Visit ER.")
        else:
            st.info("✅ STABLE: Monitor vitals and follow up with a GP if symptoms persist.")

        # F. DUAL-ENGINE DRUG RECOMMENDATIONS
        found = [d for d in [dl_disease, res_disease] if d not in ["Normal", "General Assessment"]]
        meds = []
        if found:
            st.subheader("💊 Combined Therapy Recommendations")
            cols = list(medicine_db.columns)
            for cond in found:
                mask = medicine_db[cols[1]].astype(str).str.contains(str(cond), case=False, na=False)
                for _, row in medicine_db[mask].head(2).iterrows():
                    m = {'name': row[cols[0]], 'desc': row[cols[2]], 'for': cond}
                    meds.append(m)
                    st.markdown(f'<div class="drug-card"><b>{m["name"]}</b> (for {m["for"]})<br><small>{m["desc"]}</small></div>', unsafe_allow_html=True)

        # G. EMAIL & DOWNLOAD RESTORED
        st.divider()
        report_txt = f"Patient: {p_name}\nVitals: {dl_disease} ({prob:.2f}%)\nSymptoms: {res_disease}\nStatus: {urgency}"
        c1, c2 = st.columns(2)
        with c1:
            st.download_button("📄 Download Report", report_txt, file_name=f"Report_{p_name}.txt")
        with c2:
            if doc_email:
                with st.spinner("Sending Email..."):
                    rep = {'name': p_name, 'v_diag': dl_disease, 'v_prob': f"{prob:.2f}%", 's_diag': res_disease, 'urgency': urgency}
                    if send_to_doctor(doc_email, rep, meds):
                        st.success(f"Clinical Alert sent to {doc_email}!")
    else:
        st.error("Model assets not found.")
