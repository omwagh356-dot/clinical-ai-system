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

# Theme-neutral CSS for professional "Clinical Vibe"
st.markdown("""
    <style>
    .report-container { border: 2px solid #1a73e8; padding: 20px; border-radius: 10px; margin-top: 20px; color: inherit; }
    .drug-card { background-color: rgba(26, 115, 232, 0.1); border-left: 5px solid #1a73e8; padding: 12px; margin-bottom: 10px; border-radius: 5px; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #1a73e8; color: white; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_ml_assets():
    """Loads Deep Learning model and pre-processing scalers."""
    try:
        model = load_model("model/model.h5")
        scaler = joblib.load("scaler.pkl")
        le = joblib.load("label_encoder.pkl")
        return model, scaler, le
    except Exception as e:
        st.warning(f"ML Assets (model.h5) not found: {e}")
        return None, None, None

@st.cache_data
def load_knowledge_bases():
    """Loads CSV/Excel with robust error handling for encoding and hidden characters."""
    symptom_file = 'DiseaseAndSymptoms.csv'
    medicine_file = 'Medicine_description.xlsx'
    
    if not os.path.exists(symptom_file) or not os.path.exists(medicine_file):
        st.error(f"❌ Critical Error: Ensure '{medicine_file}' and '{symptom_file}' are in the project folder.")
        return {}, pd.DataFrame()

    try:
        # Load Symptoms
        disease_df = pd.read_csv(symptom_file, encoding='latin1', on_bad_lines='skip', engine='python')
        # Load Medicines directly from Excel
        medicine_db = pd.read_excel(medicine_file)
        
        # Aggressive Header Cleaning
        medicine_db.columns = [str(c).strip().replace('\xa0', '') for c in medicine_db.columns]
        
        disease_map = {}
        for _, row in disease_df.iterrows():
            d = str(row['Disease']).strip()
            s = [str(val).strip().lower().replace("_", " ") for val in row[1:] if pd.notna(val)]
            if d not in disease_map: disease_map[d] = set(s)
            else: disease_map[d].update(s)
        return disease_map, medicine_db
    except Exception as e:
        st.error(f"File Loading Error: {e}")
        return {}, pd.DataFrame()

model, scaler, le = load_ml_assets()
disease_map, medicine_db = load_knowledge_bases()

# --- 2. CORE SEARCH LOGIC ---

def get_medication_recs(user_input_text, detected_disease):
    """STRICT DISEASE-FIRST SEARCH: Prioritizes the detected disease to prevent irrelevant results."""
    if medicine_db.empty: return []
    
    primary_disease = detected_disease.lower().strip()
    cols = list(medicine_db.columns)
    
    # Standard mapping: Col 0: Name, Col 1: Reason, Col 2: Description
    name_col = cols[0]
    reason_col = cols[1]
    desc_col = cols[2] if len(cols) > 2 else cols[-1]

    # PRIORITY 1: Look for exact/partial match in the 'Reason' column
    # If Malaria is detected, find rows where 'Malaria' is the Reason.
    primary_mask = medicine_db[reason_col].astype(str).str.contains(primary_disease, case=False, na=False)
    primary_results = medicine_db[primary_mask]

    # PRIORITY 2: If no direct disease match, search keywords in Description
    if primary_results.empty:
        filler_words = {'i', 'have', 'very', 'bad', 'with', 'and', 'some', 'also', 'the', 'is', 'a'}
        keywords = [word.strip(".,!").lower() for word in user_input_text.split() 
                    if word.lower() not in filler_words and len(word) > 4]
        
        if not keywords: return []
        
        secondary_mask = medicine_db[desc_col].astype(str).apply(
            lambda x: sum(1 for word in keywords if word in x.lower()) >= 1
        )
        final_results = medicine_db[secondary_mask].head(5)
    else:
        final_results = primary_results.head(5)

    matches = []
    for _, row in final_results.iterrows():
        matches.append({
            'Drug_Name': row[name_col],
            'Description': row[desc_col]
        })
    return matches

def send_to_doctor(receiver_email, report, drug_list):
    """Sends detailed medical alert via Gmail SMTP."""
    msg = EmailMessage()
    msg['Subject'] = f"🚨 {report['urgency']} Alert: {report['name']}"
    msg['From'] = st.secrets["EMAIL_USER"]
    msg['To'] = receiver_email
    med_text = "\n".join([f"- {m['Drug_Name']}: {m['Description'][:100]}..." for m in drug_list])
    body = f"PATIENT: {report['name']} ({report['age']})\nURGENCY: {report['urgency']}\nDIAGNOSIS: {report['disease']}\nDETECTION: {report['symptom_disease']}\n\nTHERAPY:\n{med_text}"
    msg.set_content(body)
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(st.secrets["EMAIL_USER"], st.secrets["EMAIL_PASS"])
            smtp.send_message(msg)
            return True
    except: return False

# --- 3. UI LAYOUT ---
st.title("🏥 Clinical AI: Intelligent Risk & Therapy Engine")
st.caption("Onkar Suresh Wagh | M.Sc. Data Science Final Project")
st.divider()

col_l, col_r = st.columns([1, 1], gap="large")
with col_l:
    st.subheader("👤 Patient Identity & Vitals")
    p_name = st.text_input("Full Name")
    doc_email = st.text_input("Doctor Email")
    p_age = st.number_input("Age", 1, 120, 23)
    v1, v2 = st.columns(2)
    hr = v1.number_input("Heart Rate", value=72.0)
    spo2 = v2.number_input("SpO2 %", value=98.0)
    bps = v1.number_input("BP Systolic", value=120.0)
    temp = v2.number_input("Temp °C", value=37.0)

with col_r:
    st.subheader("📋 Clinical Presentation")
    s_input = st.text_area("Enter Symptoms (e.g. 'I have high fever and shivering', 'Joint stiffness')")

# --- 4. EXECUTION ---
st.divider()
if st.button("RUN FULL DIAGNOSTIC & NOTIFY DOCTOR", type="primary", use_container_width=True):
    if model and scaler:
        # A. Vitals Engine
        raw_v = [p_age, hr, bps, 80.0, spo2, temp, 190.0, 95.0, 16.0]
        scaled = scaler.transform([raw_v])
        pred = model.predict(scaled, verbose=0)
        idx = np.argmax(pred)
        dl_disease = le.inverse_transform([idx])[0]
        prob = pred[0][idx] * 100
        
        # B. Safety Overrides (Fever/Hypoxia)
        if temp >= 40.0:
            dl_disease = "Hyperpyrexia (Critical Fever)"
            prob = 100.0
        elif temp >= 38.0 and dl_disease == "Normal":
            dl_disease = "Pyrexia (Fever)"
            prob = 90.0
        
        # C. Symptom Engine
        user_text = s_input.lower()
        matched_scores = {d: sum(1 for sym in s_set if sym in user_text) for d, s_set in disease_map.items()}
        res_disease = max(matched_scores, key=matched_scores.get, default="General Assessment") if any(matched_scores.values()) else "General Assessment"
        
        # D. Triage Status
        urgency = "IMMEDIATE ER" if spo2 < 90 or bps > 175 or temp >= 39.5 else "Stable"

        # E. Display Results
        st.markdown(f"""<div class="report-container"><h2 style='text-align: center;'>Clinical Diagnostic Report</h2><hr>
            <p><b>AI Diagnosis (Vitals):</b> {dl_disease} ({prob:.2f}%)</p>
            <p><b>Symptom Detection:</b> {res_disease}</p>
            <p><b>Status:</b> <span style="color: {'red' if urgency != 'Stable' else 'green'}; font-weight: bold;">{urgency}</span></p></div>""", unsafe_allow_html=True)

        # F. Strict Medication Search
        # Uses the Symptom detection first, then Vitals detection if needed
        final_query = res_disease if res_disease != "General Assessment" else dl_disease
        meds = get_medication_recs(s_input, final_query)
        
        if meds:
            st.subheader(f"💊 Recommended Therapy for {final_query}")
            for m in meds: 
                st.markdown(f'<div class="drug-card"><b>{m["Drug_Name"]}</b><br><small>{m["Description"]}</small></div>', unsafe_allow_html=True)
        else:
            st.warning("No matching medications found. Ensure condition name is in Excel Reason column.")
        
        # G. Email Alert
        if doc_email:
            with st.spinner("Emailing alert..."):
                if send_to_doctor(doc_email, {'name': p_name, 'age': p_age, 'disease': dl_disease, 'prob': f"{prob:.2f}", 'urgency': urgency, 'vitals': raw_v, 'symptom_disease': res_disease}, meds):
                    st.success(f"Clinical alert sent to {doc_email}! ✅")
    else:
        st.error("Missing ML Assets (model.h5). check file directory.")
