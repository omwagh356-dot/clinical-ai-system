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

# Professional UI Styling
st.markdown("""
    <style>
    .report-container { border: 2px solid #1a73e8; padding: 20px; border-radius: 10px; margin-top: 20px; color: inherit; }
    .drug-card { background-color: rgba(26, 115, 232, 0.1); border-left: 5px solid #1a73e8; padding: 10px; margin-bottom: 10px; border-radius: 5px; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #1a73e8; color: white; }
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
        st.error(f"❌ Critical Error: Files missing. Ensure '{medicine_file}' is in the folder.")
        return {}, pd.DataFrame()

    try:
        # Load Symptoms (CSV)
        disease_df = pd.read_csv(symptom_file, encoding='latin1', on_bad_lines='skip', engine='python')
        
        # Load Medicines (EXCEL) - Targeting the Excel file directly
        medicine_db = pd.read_excel(medicine_file)
        
        # Aggressive Header Cleaning
        medicine_db.columns = [str(c).strip() for c in medicine_db.columns]
        
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

# --- 2. CORE LOGIC ---

def get_medication_recs(user_input_text, detected_disease):
    """Deep search across the Excel database."""
    if medicine_db.empty: return []
    
    # Standardize keywords
    keywords = set()
    input_source = f"{user_input_text} {detected_disease}".lower()
    for word in input_source.split():
        clean_word = word.strip(".,()[]")
        if len(clean_word) > 3: keywords.add(clean_word)
            
    # Clinical Synonyms for proactive matching
    synonyms = {
        "arthritis": ["joint", "inflammation", "pain", "swelling", "nsaid", "osteo", "rheumatoid"],
        "anemia": ["iron", "blood", "weakness", "folic", "haemoglobin", "pale"],
        "jaundice": ["liver", "hepatitis", "bilirubin", "yellow", "bile"],
        "fever": ["pyrexia", "infection", "paracetamol", "cold", "feverish"],
        "malaria": ["quinine", "parasite", "chills", "mosquito"]
    }
    for key, syn_list in synonyms.items():
        if key in input_source:
            keywords.update(syn_list)
            keywords.add(key)

    if not keywords: return []

    # Row-by-row Deep Search
    matches = []
    try:
        records = medicine_db.to_dict('records')
        for row in records:
            row_content = " ".join([str(val).lower() for val in row.values()])
            if any(word in row_content for word in keywords):
                cols = list(row.keys())
                matches.append({
                    'Drug_Name': row.get(cols[0], "N/A"),
                    'Description': row.get(cols[2], row.get(cols[-1], "N/A"))
                })
                if len(matches) >= 5: break
        return matches
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
    s_input = st.text_area("Enter Symptoms (e.g. 'joint stiffness', 'pain', 'very weak')")

# --- 4. EXECUTION ---
st.divider()
if st.button("RUN FULL DIAGNOSTIC & NOTIFY DOCTOR", type="primary"):
    if model and scaler:
        # A. AI Engines
        raw_v = [p_age, hr, bps, 80.0, spo2, temp, 190.0, 95.0, 16.0]
        scaled = scaler.transform([raw_v])
        pred = model.predict(scaled, verbose=0)
        idx = np.argmax(pred); dl_disease = le.inverse_transform([idx])[0]; prob = pred[0][idx] * 100
        
        user_text = s_input.lower()
        matched_scores = {d: sum(1 for sym in s_set if sym in user_text) for d, s_set in disease_map.items()}
        res_disease = max(matched_scores, key=matched_scores.get, default="General Assessment") if any(matched_scores.values()) else "General Assessment"
        
        urgency = "IMMEDIATE ER" if spo2 < 90 or bps > 175 or temp >= 38.5 else "Stable"

        # B. Report Display
        st.markdown(f"""<div class="report-container"><h2>Clinical Diagnostic Report</h2><hr>
            <p><b>AI Vital Diagnosis:</b> {dl_disease} ({prob:.2f}%)</p>
            <p><b>Symptom Detection:</b> {res_disease}</p>
            <p><b>Status:</b> <span style="color: {'red' if urgency != 'Stable' else 'green'}; font-weight: bold;">{urgency}</span></p></div>""", unsafe_allow_html=True)

        # C. Therapy Search
        meds = get_medication_recs(s_input, res_disease)
        if meds:
            st.subheader(f"💊 Recommended Therapy")
            for m in meds: st.markdown(f'<div class="drug-card"><b>{m["Drug_Name"]}</b><br><small>{m["Description"]}</small></div>', unsafe_allow_html=True)
        else:
            st.warning("No matching medications found. Ensure 'Arthritis' is in the Excel file.")
        
        if doc_email:
            with st.spinner("Emailing alert..."):
                if send_to_doctor(doc_email, {'name': p_name, 'age': p_age, 'disease': dl_disease, 'prob': f"{prob:.2f}", 'urgency': urgency, 'vitals': raw_v, 'symptom_disease': res_disease}, meds):
                    st.success("Report emailed! ✅")
    else: st.error("Assets Missing.")
