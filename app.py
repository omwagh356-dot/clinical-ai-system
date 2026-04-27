import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
import joblib
import smtplib
from email.message import EmailMessage

# --- 1. CORE FUNCTIONS (Reports & Email) ---

def create_clinical_report(report, reasons, warnings):
    """Generates a professional HTML diagnostic report."""
    return f"""
    <div style='font-family:Arial; border:2px solid #333; padding:20px; border-radius:10px;'>
        <h1 style='color:#1a73e8; text-align:center;'>Clinical AI Diagnostic Report</h1>
        <hr>
        <p><b>Patient:</b> {report['Name']} | <b>Age:</b> {report['Age']} | <b>Gender:</b> {report['Gender']}</p>
        <div style='background:#f0f2f6; padding:15px; border-radius:10px;'>
            <h3 style='margin:0;'>Primary Diagnostic: {report['Disease']}</h3>
            <p style='color:#d93025; font-weight:bold;'>Triage Status: {report['Risk']}</p>
        </div>
        <h4>📊 Clinical Vitals Analysis</h4>
        <ul>{"".join([f"<li>{r}</li>" for r in reasons])}</ul>
        <p><b>Next Steps:</b> {report['Urgency']}</p>
    </div>
    """

def send_to_physician(receiver, report, reasons):
    """Handles secure SMTP transmission to the physician."""
    msg = EmailMessage()
    msg['Subject'] = f"🚨 {report['Risk']} Clinical Alert - {report['Name']}"
    msg['From'] = st.secrets.get("EMAIL_USER", "")
    msg['To'] = receiver
    body = f"Clinical Alert for {report['Name']}.\nPredicted: {report['Disease']}\nAnalysis: {reasons}"
    msg.set_content(body)
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(st.secrets["EMAIL_USER"], st.secrets["EMAIL_PASS"])
            smtp.send_message(msg)
        return True
    except: return False

# --- 2. DATA LOADING (Handles 'res' header and Byte Order Marks) ---

@st.cache_data
def load_clinical_data():
    disease_df = pd.read_csv('DiseaseAndSymptoms.csv')
    disease_df['Disease'] = disease_df['Disease'].astype(str).str.strip().str.title()
    try:
        # Load medicine file and standardize the 'res' header to 'Reason'
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

# Initialize Assets
model, scaler, label_encoder = load_ml_assets()
disease_db, med_db = load_clinical_data()

# Try to import helper modules
try:
    from drug_module import check_drugs
    from explain import explain_values
except ImportError:
    st.error("Missing Logic Modules.")

# --- 3. UI DASHBOARD ---

st.set_page_config(page_title="Advanced CDSS", layout="wide")
st.title("🛡️ Enterprise Clinical Decision Support System")

c1, c2 = st.columns([1, 1.2])

with c1:
    st.subheader("👤 Patient Identity")
    name = st.text_input("Patient Name")
    doc_email = st.text_input("Physician Email")
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    p_age = st.number_input("Age", 1, 120, 30)
    
    st.subheader("📉 Clinical Vitals")
    v1, v2, v3 = st.columns(3)
    hr = v1.number_input("Heart Rate", value=72.0)
    bps = v1.number_input("BP Systolic", value=120.0)
    spo2 = v2.number_input("SpO2 %", value=98.0)
    bpd = v2.number_input("BP Diastolic", value=80.0)
    temp = v3.number_input("Temp °C", value=37.0)
    gluc = v3.number_input("Glucose", value=95.0)

with c2:
    st.subheader("🧪 Clinical Context")
    # THE KEYWORD BOX: Medicine recommendations search for these words!
    curr_syms = st.text_area("Symptoms (e.g., Fever, Acne, Wound, Cough)")
    curr_meds = st.text_area("Current Medications")
    curr_allergies = st.text_area("Allergies")

# --- 4. THE EXECUTION ENGINE ---

st.divider()
if st.button("🚀 EXECUTE FULL SCAN", type="primary", width="stretch"):
    # 1. Neural Network Prediction
    raw = [p_age, hr, bps, bpd, spo2, temp, 190.0, gluc, 16.0] 
    scaled = scaler.transform(pd.DataFrame([raw], columns=scaler.feature_names_in_))
    prediction = model.predict(scaled, verbose=0)
    
    idx = np.argmax(prediction)
    disease = label_encoder.inverse_transform([idx])[0]
    prob = prediction[0][idx] * 100

    # 2. HEURISTIC OVERRIDE (Weightage Fix)
    symptom_text = curr_syms.lower()
    
    # If vitals are normal but user reports specific symptoms, adjust the diagnostic label
    if "fever" in symptom_text or "cough" in symptom_text or temp >= 38.5:
        if disease == "Normal": disease = "Fever / Infection"
    elif "pimple" in symptom_text or "acne" in symptom_text:
        if disease == "Normal": disease = "Acne"
    elif "wound" in symptom_text:
        if disease == "Normal": disease = "Wound"

    # Status & Colors
    status = "🔴 CRITICAL" if temp >= 39.5 or spo2 < 89 else "🟢 STABLE"
    color = "#ff4b4b" if status == "🔴 CRITICAL" else "#28a745"
    action = "Immediate Medical Intervention Required" if status == "🔴 CRITICAL" else "Routine Monitoring"
    
    reasons = explain_values(hr, (bps+bpd)/2, spo2, temp)
    warnings, _ = check_drugs(curr_meds.split(","), curr_syms.split(","), curr_allergies.split(","))

    st.markdown(f"<div style='background:{color}; padding:20px; border-radius:10px; color:white; text-align:center;'><h1>{status}: {disease.upper()}</h1><p>{action}</p></div>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📊 Analytics", "💊 Therapy Pathway", "📄 Report"])
    
    with tab1:
        g1, g2 = st.columns([1.3, 1])
        with g1:
            conf_df = pd.DataFrame({"Condition": label_encoder.classes_, "Prob": prediction[0]*100}).sort_values("Prob")
            st.plotly_chart(px.bar(conf_df, x="Prob", y="Condition", orientation='h', template="plotly_dark"), width="stretch")
        with g2:
            radar_vals = [min(hr/160, 1.0), (100-spo2)/20, min(bps/200, 1.0), min(abs(temp-37)/5, 1.0), min(gluc/400, 1.0)]
            fig = go.Figure(data=go.Scatterpolar(r=radar_vals, theta=['HR', 'SpO2', 'BP', 'Temp', 'Gluc'], fill='toself', line_color=color))
            fig.update_layout(template="plotly_dark", polar=dict(radialaxis=dict(visible=False)))
            st.plotly_chart(fig, width="stretch")

    with tab2:
        st.subheader(f"Therapeutic Pathway for {disease}")
        
        # --- THE MASTER SEARCH FIX ---
        # Search for AI label AND symptoms typed by the user
        search_terms = disease.split() 
        if "fever" in symptom_text: search_terms.append("Fever")
        if "acne" in symptom_text or "pimple" in symptom_text: search_terms.append("Acne")
        if "wound" in symptom_text: search_terms.append("Wound")
        
        query = "|".join(set(search_terms))
        
        # Searching the standardized 'Reason' column (the 22,000-row file)
        rel_meds = med_db[med_db['Reason'].str.contains(query, case=False, na=False)].head(15)
        
        if not rel_meds.empty:
            for _, row in rel_meds.iterrows():
                with st.expander(f"💊 {row['Drug_Name']}"):
                    st.write(f"**Indication:** {row['Reason']}")
                    st.write(f"**Pharmacological Insight:** {row['Description']}")
        else:
            st.warning("No medicines found. Ensure symptoms (Fever, Acne, Wound) are typed correctly.")

    with tab3:
        report_data = {"Name": name, "Age": p_age, "Gender": gender, "Disease": disease, "Prob": round(prob, 2), "Risk": status, "Urgency": action}
        html_report = create_clinical_report(report_data, reasons, warnings)
        st.download_button("📥 Download Physician Report", html_report, file_name=f"Report_{name}.html", mime="text/html", width="stretch")
        
        if doc_email and st.button("📧 Email Physician Now", width="stretch"):
            if send_to_physician(doc_email, report_data, reasons):
                st.success("Transmitted successfully! ✅")
            else:
                st.error("Check your Secrets configuration.")
