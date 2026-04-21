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
def load_all_assets():
    try:
        # Vitals Engine (Deep Learning)
        v_model = load_model("model/model.h5")
        v_scaler = joblib.load("scaler.pkl")
        v_le = joblib.load("label_encoder.pkl")
        
        # Symptom Engine (Trained Random Forest)
        s_model = joblib.load("symptom_model.pkl")
        s_le = joblib.load("symptom_encoder.pkl")
        s_features = joblib.load("symptom_features.pkl")
        
        return v_model, v_scaler, v_le, s_model, s_le, s_features
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None, None, None

@st.cache_data
def load_medicine_db():
    try:
        df = pd.read_excel('Medicine_description.xlsx')
        df.columns = [str(c).strip() for c in df.columns]
        return df
    except:
        return pd.DataFrame()

assets = load_all_assets()
medicine_db = load_medicine_db()

# --- 3. LOGIC FUNCTIONS ---

def get_clean_tokens(text):
    text = text.lower().replace("_", " ")
    text = re.sub(r'[^\w\s]', '', text)
    return text.split()

def predict_symptoms(user_input, s_model, s_le, s_features):
    # Standardize input
    tokens = [t.strip().lower().replace("_", " ") for t in user_input.split(",")]
    
    input_vector = np.zeros(len(s_features))
    
    # STRICT MATCHING: Only flip to 1 if the exact symptom name is found
    for i, feature in enumerate(s_features):
        if feature.lower().replace("_", " ") in tokens:
            input_vector[i] = 1
            
    if np.sum(input_vector) == 0:
        return "General Assessment", 0.0
    
    # Predict with Probability
    pred_prob = s_model.predict_proba([input_vector])
    idx = np.argmax(pred_prob)
    disease = s_le.inverse_transform([idx])[0]
    confidence = np.max(pred_prob) * 100
    
    return disease, confidence

def send_alert(receiver_email, report_data, meds):
    msg = EmailMessage()
    msg['Subject'] = f"🚨 {report_data['urgency']} Alert: {report_data['name']}"
    msg['From'] = st.secrets.get("EMAIL_USER", "clinical-ai@system.com")
    msg['To'] = receiver_email
    med_text = "\n".join([f"- {m['name']} (for {m['for']})" for m in meds])
    body = f"Patient: {report_data['name']}\nVitals Diag: {report_data['v_diag']} ({report_data['v_prob']})\nSymptom Diag: {report_data['s_diag']} ({report_data['s_prob']})\nStatus: {report_data['urgency']}\n\nTherapy:\n{med_text}"
    msg.set_content(body)
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(st.secrets["EMAIL_USER"], st.secrets["EMAIL_PASS"])
            smtp.send_message(msg); return True
    except: return False

# --- 4. HEADER ---
st.title("🏥 Clinical AI: Intelligent Risk & Therapy Engine")
st.markdown("### **M.Sc. Data Science Final Project**")
st.markdown("**Developer:** Onkar Suresh Wagh")
st.divider()

# --- 5. UI INPUTS ---
# --- 5. UI INPUTS ---
col_l, col_r = st.columns([1, 1], gap="large")

with col_l:
    st.subheader("👤 Patient Identity & Vitals")
    p_name = st.text_input("Full Name", "Onkar Wagh")
    p_age = st.number_input("Age", 1, 120, 23)
    v1, v2 = st.columns(2)
    hr = v1.number_input("Heart Rate (BPM)", 30.0, 250.0, 72.0)
    spo2 = v2.number_input("SpO2 (%)", 50.0, 100.0, 98.0)
    bps = v1.number_input("BP Systolic (mmHg)", 50.0, 250.0, 120.0)
    temp = v2.number_input("Temperature (°C)", 30.0, 45.0, 37.0)

with col_r:
    st.subheader("📋 Clinical Presentation")
    
    # 1. Load the features (symptoms) your model knows
    if assets[5] is not None:
        symptom_list = assets[5] # This is your symptom_features.pkl
        
        # 2. Multi-select Dropdown
        selected = st.multiselect(
            "Quick Select Symptoms:", 
            options=symptom_list,
            help="Select symptoms to automatically add them to the description box below."
        )
        
        # 3. Create the combined text for the box
        default_text = ", ".join(selected)
    else:
        default_text = ""

    # 4. The Text Area (populated by the dropdown)
    s_input = st.text_area(
        "Clinical Description", 
        value=default_text,
        placeholder="Selected symptoms will appear here. You can also type manually.",
        height=150
    )
    
    doc_email = st.text_input("Doctor Email for Alerts")
# --- 6. EXECUTION ---
st.divider()
if st.button("RUN FULL DIAGNOSTIC", type="primary"):
    if all(assets) and not medicine_db.empty:
        v_model, v_scaler, v_le, s_model, s_le, s_features = assets
        
        # A. Vitals Engine (Deep Learning)
        features = ['age', 'heart_rate', 'bp_systolic', 'bp_diastolic', 'spo2', 'temp', 'cholesterol', 'glucose', 'respiratory_rate']
        raw_v = [p_age, hr, bps, 80.0, spo2, temp, 190.0, 95.0, 16.0]
        v_scaled = v_scaler.transform(pd.DataFrame([raw_v], columns=features))
        v_preds = v_model.predict(v_scaled, verbose=0)
        v_diag = v_le.inverse_transform([np.argmax(v_preds)])[0]
        v_prob = np.max(v_preds) * 100
        
        # B. Symptom Engine (Trained Model)
        s_diag, s_prob = predict_symptoms(s_input, s_model, s_le, s_features)

        # C. Status
        urgency = "EMERGENCY" if spo2 < 90 or bps > 180 or temp >= 39.5 else "Stable"

        # D. Display
        st.markdown(f"""<div class="report-container">
            <h3 style='text-align: center;'>Clinical Diagnostic Report</h3><hr>
            <p><b>Vitals AI Prediction:</b> {v_diag} ({v_prob:.2f}%)</p>
            <p><b>Symptom AI Prediction:</b> {s_diag} {f'({s_prob:.2f}%)' if s_prob > 0 else ''}</p>
            <p><b>Status:</b> <span style="color:{'red' if urgency=='EMERGENCY' else 'green'}">{urgency}</span></p>
            </div>""", unsafe_allow_html=True)

        # E. Insights
        st.subheader("🛑 Actionable Insights")
        if urgency == "EMERGENCY":
            st.error("🚨 CRITICAL: Immediate intervention required. Visit ER.")
        else:
            st.info("✅ STABLE: Routine follow-up recommended.")

        # F. Therapy
        st.subheader("💊 Therapy Recommendations")
        found = [d for d in [v_diag, s_diag] if d not in ["Normal", "General Assessment"]]
        med_list = []
        cols = medicine_db.columns.tolist()
        
        for cond in found:
            mask = medicine_db[cols[1]].astype(str).str.contains(str(cond), case=False, na=False)
            results = medicine_db[mask].head(3)
            for _, row in results.iterrows():
                m = {'name': row[cols[0]], 'desc': row[cols[2]] if len(cols) > 2 else "N/A", 'for': cond}
                med_list.append(m)
                st.markdown(f'<div class="drug-card"><b>{m["name"]}</b> (Target: {m["for"]})<br><small>{m["desc"]}</small></div>', unsafe_allow_html=True)

        # G. Export
        report_txt = f"Patient: {p_name}\nVitals: {v_diag} ({v_prob:.2f}%)\nSymptoms: {s_diag}\nStatus: {urgency}"
        c1, c2 = st.columns(2)
        with c1: st.download_button("📄 Download Report", report_txt, file_name=f"Report_{p_name}.txt")
        with c2:
            if doc_email:
                with st.spinner("Sending Email Alert..."):
                    rep = {'name':p_name, 'v_diag':v_diag, 'v_prob':f"{v_prob:.2f}%", 's_diag':s_diag, 's_prob':f"{s_prob:.2f}%", 'urgency':urgency}
                    if send_alert(doc_email, rep, med_list):
                        st.success("Alert sent to physician!")
    else:
        st.error("Assets or Data missing. Ensure 'symptom_model.pkl' etc. are in the root folder.")
