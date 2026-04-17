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
    .drug-card { background-color: rgba(26, 115, 232, 0.1); border-left: 5px solid #1a73e8; padding: 10px; margin-bottom: 10px; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_ml_assets():
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
    symptom_file = 'DiseaseAndSymptoms.csv'
    medicine_file = 'Medicine_description.xlsx'
    if not os.path.exists(symptom_file) or not os.path.exists(medicine_file):
        st.error("❌ CSV files missing.")
        return {}, pd.DataFrame()
    try:
        disease_df = pd.read_csv(symptom_file, encoding='latin1', on_bad_lines='skip', engine='python')
        medicine_db = pd.read_csv(medicine_file, encoding='latin1', on_bad_lines='skip', engine='python')
        
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
        st.error(f"Database Loading Error: {e}"); return {}, pd.DataFrame()

model, scaler, le = load_ml_assets()
disease_map, medicine_db = load_knowledge_bases()

# --- 2. CORE LOGIC ---

def get_medication_recs(user_input_text, detected_disease):
    """
    ULTRA-SMART SEARCH: Scans all columns for any mention of symptoms or diseases.
    """
    if medicine_db.empty: return []
    
    # 1. Standardize query words
    combined_query = f"{user_input_text} {detected_disease}".lower().replace(",", " ").replace("(", " ").replace(")", " ")
    keywords = [word.strip() for word in combined_query.split() if len(word) > 2]
    
    # Clinical Synonym Expansion
    synonyms = {
        "arthritis": ["arthriti", "joint", "inflammation", "pain", "swelling", "nsaid", "osteo"],
        "anemia": ["anemi", "iron", "blood", "weakness", "folic", "haemoglobin"],
        "jaundice": ["liver", "hepatitis", "bilirubin", "yellow", "bile"],
        "fever": ["pyrexia", "infection", "paracetamol", "cold", "feverish"],
        "malaria": ["quinine", "parasite", "fever", "chills", "mosquito"]
    }
    
    final_keywords = set(keywords)
    for key, syns in synonyms.items():
        if key in combined_query:
            final_keywords.update(syns)

    if not final_keywords: return []

    try:
        # Create a search string for every row by joining all column values
        # This ensures we find the word even if it is in the 'Reason' or 'Description' column
        def row_contains_keywords(row):
            row_content = " ".join(row.astype(str)).lower()
            return any(word in row_content for word in final_keywords)

        mask = medicine_db.apply(row_contains_keywords, axis=1)
        results = medicine_db[mask].head(5)
        
        final_list = []
        for _, r in results.iterrows():
            final_list.append({
                'Drug_Name': r[0] if len(r) > 0 else "Unknown",
                'Description': r[2] if len(r) > 2 else r[-1]
            })
        return final_list
    except: return []

def send_to_doctor(receiver_email, report, drug_list):
    msg = EmailMessage()
    msg['Subject'] = f"🚨 {report['urgency']} Alert: {report['name']}"
    msg['From'] = st.secrets["EMAIL_USER"]
    msg['To'] = receiver_email
    med_text = "\n".join([f"- {m['Drug_Name']}: {m['Description'][:100]}..." for m in drug_list])
    body = f"PATIENT: {report['name']}\nDETECTION: {report['symptom_disease']}\n\nTHERAPY:\n{med_text}"
    msg.set_content(body)
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(st.secrets["EMAIL_USER"], st.secrets["EMAIL_PASS"])
            smtp.send_message(msg); return True
    except: return False

# --- 3. UI LAYOUT ---
st.title("🏥 Clinical AI: Intelligent Risk & Therapy Engine")
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
    s_input = st.text_area("Enter Symptoms (e.g. 'joint pain', 'yellow eye', 'I have malaria')")

# --- 4. EXECUTION ---
st.divider()
if st.button("RUN FULL DIAGNOSTIC & NOTIFY DOCTOR", type="primary", use_container_width=True):
    if model and scaler:
        # A. Vitals Engine
        raw_v = [p_age, hr, bps, 80.0, spo2, temp, 190.0, 95.0, 16.0]
        scaled = scaler.transform([raw_v])
        pred = model.predict(scaled, verbose=0)
        idx = np.argmax(pred); dl_disease = le.inverse_transform([idx])[0]; prob = pred[0][idx] * 100
        
        # B. Symptom Engine (Detection)
        user_text = s_input.lower()
        matched_scores = {d: sum(1 for sym in s_set if sym in user_text) for d, s_set in disease_map.items()}
        res_disease = max(matched_scores, key=matched_scores.get, default="General Assessment") if any(matched_scores.values()) else "General Assessment"
        
        # C. Triage
        urgency = "IMMEDIATE ER" if spo2 < 90 or bps > 175 or temp >= 38.5 else "Stable"

        # D. Display Results
        st.markdown(f"""<div class="report-container"><h2 style='text-align: center;'>Clinical Diagnostic Report</h2><hr>
            <p><b>AI Vital Diagnosis:</b> {dl_disease} ({prob:.2f}%)</p>
            <p><b>Symptom Detection:</b> {res_disease}</p>
            <p><b>Status:</b> <span style="color: {'red' if urgency != 'Stable' else 'green'}; font-weight: bold;">{urgency}</span></p></div>""", unsafe_allow_html=True)

        # E. MULTI-COLUMN FUZZY SEARCH
        meds = get_medication_recs(s_input, res_disease)
        
        if meds:
            st.subheader(f"💊 Recommended Therapy")
            for m in meds: 
                st.markdown(f'<div class="drug-card"><b>{m["Drug_Name"]}</b><br><small>{m["Description"]}</small></div>', unsafe_allow_html=True)
        else:
            st.warning("No specific medications found. Try broader terms like 'pain' or 'liver'.")
        
        if doc_email:
            with st.spinner("Emailing alert..."):
                if send_to_doctor(doc_email, {'name': p_name, 'age': p_age, 'disease': dl_disease, 'prob': f"{prob:.2f}", 'urgency': urgency, 'vitals': raw_v, 'symptom_disease': res_disease}, meds):
                    st.success("Report emailed! ✅")
    else: st.error("Missing Assets.")
