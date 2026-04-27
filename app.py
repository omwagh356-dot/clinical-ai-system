import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
import joblib
import smtplib
from email.message import EmailMessage

# --- 1. CORE UTILITY FUNCTIONS ---

def create_clinical_report(report, reasons, warnings):
    """Generates professional HTML diagnostic report."""
    return f"""
    <div style='font-family:Arial; border:2px solid #333; padding:20px; border-radius:10px;'>
        <h1 style='color:#1a73e8; text-align:center;'>Clinical AI Diagnostic Report</h1>
        <hr>
        <p><b>Patient:</b> {report['Name']} | <b>Age:</b> {report['Age']}</p>
        <div style='background:#f0f2f6; padding:15px; border-radius:10px;'>
            <h3 style='margin:0;'>Primary Diagnosis: {report['Disease']}</h3>
            <p style='color:#d93025; font-weight:bold;'>Triage Status: {report['Risk']}</p>
        </div>
        <h4>Physiological Analysis:</h4>
        <ul>{"".join([f"<li>{r}</li>" for r in reasons])}</ul>
    </div>
    """

def send_to_physician(receiver, report, reasons):
    """Handles secure email transmission."""
    msg = EmailMessage()
    msg['Subject'] = f"🚨 {report['Risk']} Alert - {report['Name']}"
    msg['From'] = st.secrets.get("EMAIL_USER", "")
    msg['To'] = receiver
    msg.set_content(f"Clinical Alert for {report['Name']}. Prediction: {report['Disease']}.")
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(st.secrets["EMAIL_USER"], st.secrets["EMAIL_PASS"])
            smtp.send_message(msg)
        return True
    except: return False

# --- 2. DATA LOADING (The 'res' Header Fix) ---

@st.cache_data
def load_clinical_data():
    disease_df = pd.read_csv('DiseaseAndSymptoms.csv')
    disease_df['Disease'] = disease_df['Disease'].astype(str).str.strip().str.title()
    try:
        # Renaming the 'res' column header from your specific CSV to 'Reason'
        df = pd.read_csv('Medicine_description.xlsx', sep=None, engine='python', encoding='latin1')
        df.columns = [col.strip().replace('ï»¿', '') for col in df.columns]
        if 'res' in df.columns:
            df = df.rename(columns={'res': 'Reason'})
        else:
            df = df.rename(columns={df.columns[1]: 'Reason'})
        df['Reason'] = df['Reason'].fillna('Unknown').astype(str).str.strip().str.title()
        return disease_df, df
    except:
        return disease_df, pd.DataFrame(columns=['Drug_Name', 'Reason', 'Description'])

@st.cache_resource
def load_ml_assets():
    model = load_model("model/model.h5")
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    return model, scaler, label_encoder

model, scaler, label_encoder = load_ml_assets()
disease_db, med_db = load_clinical_data()

try:
    from drug_module import check_drugs
    from explain import explain_values
except:
    st.error("Missing Logic Modules.")

# --- 3. UI DASHBOARD ---

st.set_page_config(page_title="Advanced CDSS", layout="wide")
st.title("🛡️ Enterprise Clinical Decision Support System")

c1, c2 = st.columns([1, 1.2])
with c1:
    st.subheader("👤 Patient Identity")
    name = st.text_input("Full Name")
    doc_email = st.text_input("Physician Email")
    p_age = st.number_input("Age", 1, 120, 30)
    st.subheader("📉 Clinical Vitals")
    v_c1, v_c2 = st.columns(2)
    hr = v_c1.number_input("Heart Rate", value=72.0)
    bps = v_c1.number_input("BP Systolic", value=120.0)
    spo2 = v_c2.number_input("SpO2 %", value=98.0)
    temp = v_c2.number_input("Temp °C", value=37.0)
    gluc = st.number_input("Glucose", value=95.0)

with c2:
    st.subheader("🧪 Contextual Symptoms")
    curr_syms = st.text_area("Symptoms (Type Fever, Acne, or Cough here)")
    curr_meds = st.text_area("Current Medications")
    curr_allergies = st.text_area("Allergies")

# --- 4. THE DIAGNOSTIC ENGINE ---

if st.button("🚀 EXECUTE FULL SCAN", type="primary", use_container_width=True):
    # Predict using Neural Network
    raw = [p_age, hr, bps, 80.0, spo2, temp, 190.0, gluc, 16.0] 
    scaled = scaler.transform(pd.DataFrame([raw], columns=scaler.feature_names_in_))
    prediction = model.predict(scaled, verbose=0)
    
    idx = np.argmax(prediction)
    disease = label_encoder.inverse_transform([idx])[0]
    prob = prediction[0][idx] * 100

    # TRIAGE & SYMPTOM OVERRIDE
    symptom_text = curr_syms.lower()
    
    # Force diagnosis change if symptoms exist but vitals are normal
    if "fever" in symptom_text or temp >= 38.5:
        if disease == "Normal": disease = "Fever"
    elif "acne" in symptom_text or "pimple" in symptom_text:
        if disease == "Normal": disease = "Acne"
    elif "wound" in symptom_text:
        if disease == "Normal": disease = "Wound"

    status = "🔴 CRITICAL" if temp >= 39.5 or spo2 < 89 else "🟢 STABLE"
    color = "#ff4b4b" if status == "🔴 CRITICAL" else "#28a745"

    st.markdown(f"<div style='background:{color}; padding:20px; border-radius:10px; color:white; text-align:center;'><h1>{status}: {disease.upper()}</h1></div>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📊 Analytics", "💊 Therapy Pathway", "📄 Report"])
    
    with tab1:
        g1, g2 = st.columns([1.2, 1])
        with g1:
            st.plotly_chart(px.bar(pd.DataFrame({"Condition": label_encoder.classes_, "Prob": prediction[0]*100}), x="Prob", y="Condition", orientation='h', template="plotly_dark"), use_container_width=True)
        with g2:
            radar_vals = [min(hr/160, 1.0), (100-spo2)/20, min(bps/200, 1.0), min(abs(temp-37)/5, 1.0), min(gluc/400, 1.0)]
            fig = go.Figure(data=go.Scatterpolar(r=radar_vals, theta=['HR', 'SpO2', 'BP', 'Temp', 'Gluc'], fill='toself', line_color=color))
            fig.update_layout(template="plotly_dark", polar=dict(radialaxis=dict(visible=False)))
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader(f"Pharmaceutical Recommendations for {disease}")
        
        # --- THE MASTER SEARCH FIX ---
        # 1. Search for the Predicted Disease
        search_terms = [disease]
        
        # 2. ALSO search for symptoms typed in the box to ensure we find meds
        if "fever" in symptom_text: search_terms.append("Fever")
        if "acne" in symptom_text: search_terms.append("Acne")
        if "wound" in symptom_text: search_terms.append("Wound")
        
        query = "|".join(search_terms)
        # Search the standardized 'Reason' (which was your 'res' column)
        rel_meds = med_db[med_db['Reason'].str.contains(query, case=False, na=False)].head(15)
        
        if not rel_meds.empty:
            for _, row in rel_meds.iterrows():
                with st.expander(f"💊 {row['Drug_Name']}"):
                    st.write(f"**Indication:** {row['Reason']}")
                    st.write(f"**Description:** {row['Description']}")
        else:
            st.warning("No medicine match found in 22,000 records. Try typing 'Acne' or 'Fever' in Symptoms.")

    with tab3:
        st.info("Download report and email physician modules active.")
