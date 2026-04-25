import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
import joblib
import smtplib
from email.message import EmailMessage

# --- 1. CORE LOGIC IMPORTS ---
try:
    from drug_module import check_drugs
    from explain import explain_values
except ImportError:
    st.error("⚠️ Critical modules missing! Ensure drug_module.py and explain.py are in the folder.")

# --- 2. KNOWLEDGE BASES ---
CLINICAL_DATABASE = {
    "Infection": {"icon": "🤒", "drugs": ["Acetaminophen", "Ceftriaxone"], "next_steps": "Blood Cultures, CBC, Lactic Acid check.", "pathway": "Sepsis Protocol"},
    "Respiratory Failure": {"icon": "🫁", "drugs": ["Oxygen", "Albuterol"], "next_steps": "ABG, Chest X-Ray, Intubation Eval.", "pathway": "Acute Respiratory Protocol"},
    "Hypertension": {"icon": "🩸", "drugs": ["Lisinopril", "Amlodipine"], "next_steps": "ECG, Urinalysis.", "pathway": "Hypertensive Management"},
    "Normal": {"icon": "✅", "drugs": ["Routine Care"], "next_steps": "Standard follow-up.", "pathway": "Wellness Tracking"},
    "Cardiac Emergency": {"icon": "💔", "drugs": ["Aspirin", "Nitroglycerin"], "next_steps": "12-Lead ECG, Troponin.", "pathway": "ACLS Protocol"},
    "Hyperglycemia": {"icon": "🍭", "drugs": ["Insulin (Regular)", "IV Fluids"], "next_steps": "BMP, Blood Ketones, pH.", "pathway": "DKA/HHS Protocol"}
}

SYMPTOM_DRUGS = {
    "chest pain": {"rec": "Aspirin (324mg), Nitroglycerin.", "safety": "🚨 CRITICAL: Possible Heart Attack. Call ER."},
    "fever": {"rec": "Acetaminophen (650mg).", "safety": "✅ Monitor for confusion or stiff neck."},
    "cough": {"rec": "Guaifenesin.", "safety": "⚠️ Avoid suppressants if mucus is thick/green."},
    "diarrhea": {"rec": "Loperamide, ORS.", "safety": "⚠️ Do not use if stool is bloody."},
    "headache": {"rec": "Ibuprofen.", "safety": "✅ Caution if following head trauma."},
    "shortness of breath": {"rec": "Oxygen/Albuterol.", "safety": "🚨 EMERGENCY: High risk of Respiratory Failure."}
}

# --- 3. HELPER FUNCTIONS ---
@st.cache_resource
def load_assets():
    model = load_model("model/model.h5")
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    return model, scaler, label_encoder

def get_triage_status(disease, prob, spo2, bps):
    if spo2 < 88 or bps > 190 or (disease != "Normal" and prob > 90):
        return "🔴 CRITICAL", "Immediate Physician Intervention Required", "#ff4b4b"
    elif disease != "Normal" or prob > 70:
        return "🟡 URGENT", "Priority Nursing Assessment", "#ffa500"
    else:
        return "🟢 STABLE", "Routine Monitoring", "#28a745"

def send_to_doctor(receiver, report, reasons, warnings):
    msg = EmailMessage()
    msg['Subject'] = f"🚨 {report['Risk']} Clinical Report: {report['Name']}"
    msg['From'] = st.secrets["EMAIL_USER"]
    msg['To'] = receiver
    body = f"""OFFICIAL CLINICAL REPORT
-------------------------------
PATIENT: {report['Name']}
AGE: {report['Age']} | GENDER: {report['Gender']}

DIAGNOSIS: {report['Disease']} ({report['Prob']}%)
TRIAGE: {report['Risk']}

SYMPTOMS/CONDITIONS:
{report['Symptoms']}

ANALYSIS:
{chr(10).join(['- ' + r for r in reasons])}

DRUG SAFETY WARNINGS:
{chr(10).join(['- ' + w for w in warnings]) if warnings else "No interactions flagged."}
-------------------------------
Verified via Clinical AI CDSS."""
    msg.set_content(body)
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(st.secrets["EMAIL_USER"], st.secrets["EMAIL_PASS"])
            smtp.send_message(msg)
        return True
    except: return False

def create_pdf_report(report, info, reasons, warnings):
    return f"""<div style='font-family:Arial; border:3px solid #333; padding:20px;'>
    <h1 style='color:#1a73e8; text-align:center;'>Clinical AI Diagnostic Report</h1>
    <hr>
    <p><b>Patient:</b> {report['Name']} | <b>Age:</b> {report['Age']} | <b>Gender:</b> {report['Gender']}</p>
    <div style='background:#f0f2f6; padding:10px; border-radius:5px;'>
        <h3>AI Prediction: {report['Disease']} ({report['Prob']}%)</h3>
        <p><b>Status:</b> {report['Risk']}</p>
    </div>
    <h4>Vitals Analysis</h4>
    <ul>{"".join([f"<li>{r}</li>" for r in reasons])}</ul>
    <h4>Drug Safety</h4>
    <ul>{"".join([f"<li>⚠️ {w}</li>" for w in warnings]) if warnings else "<li>No risks flagged</li>"}</ul>
    <p><b>Recommended Steps:</b> {info['next_steps']}</p>
    </div>"""

# --- 4. UI INTERFACE ---
st.set_page_config(page_title="Advanced Clinical AI", layout="wide")
model, scaler, label_encoder = load_assets()

st.title("🛡️ Next-Gen Clinical Decision Support System")

col_id, col_hist = st.columns(2)
with col_id:
    st.subheader("👤 Patient Identity")
    name = st.text_input("Full Name")
    doc_email = st.text_input("Doctor's Email")
    g_col, a_col = st.columns(2)
    gender = g_col.selectbox("Gender", ["Male", "Female", "Other"])
    age = a_col.number_input("Age", 1, 120, 30)
    
    st.subheader("📉 Clinical Vitals")
    v_c1, v_c2, v_c3 = st.columns(3)
    hr = v_c1.number_input("Heart Rate", value=72.0)
    bps = v_c1.number_input("BP Systolic", value=120.0)
    resp = v_c1.number_input("Resp. Rate", value=16.0)
    spo2 = v_c2.number_input("SpO2 %", value=98.0)
    bpd = v_c2.number_input("BP Diastolic", value=80.0)
    chol = v_c2.number_input("Cholesterol", value=190.0)
    temp = v_c3.number_input("Temp °C", value=37.0)
    gluc = v_c3.number_input("Glucose", value=95.0)

with col_hist:
    st.subheader("🧪 Contextual History")
    curr_drugs = st.text_area("Current Medications (comma separated)")
    curr_diseases = st.text_area("Known Conditions / Symptoms")
    curr_allergies = st.text_area("Allergies")
    
    if st.button("PRE-CHECK DRUG SAFETY", use_container_width=True):
        w_p, r_p = check_drugs(curr_drugs.split(","), curr_diseases.split(","), curr_allergies.split(","))
        for w in w_p: st.error(w)
        for r in r_p: st.success(r)

# --- 5. EXECUTION ---
st.divider()
if st.button("🚀 EXECUTE MULTIMODAL DIAGNOSTIC", type="primary", use_container_width=True):
    # 1. PREDICTION & DATAFRAME CREATION
    raw_vitals = [age, hr, bps, bpd, spo2, temp, chol, gluc, resp]
    inputs_df = pd.DataFrame([raw_vitals], columns=scaler.feature_names_in_)
    scaled = scaler.transform(inputs_df)
    pred = model.predict(scaled, verbose=0)
    
    idx = np.argmax(pred)
    disease = label_encoder.inverse_transform([idx])[0]
    prob = pred[0][idx] * 100

    # Create prob_df immediately after prediction to prevent NameError
    prob_df = pd.DataFrame({
        "Condition": label_encoder.classes_,
        "Prob": pred[0] * 100
    }).sort_values("Prob")

    # 2. TRIAGE & ANALYSIS
    status, action, color = get_triage_status(disease, prob, spo2, bps)
    st.markdown(f"<div style='background-color:{color}; padding:20px; border-radius:10px; color:white; text-align:center;'><h1>{status}</h1><p>{action}</p></div>", unsafe_allow_html=True)

    info = CLINICAL_DATABASE.get(disease, CLINICAL_DATABASE["Normal"])
    reasons = explain_values(hr, (bps+bpd)/2, spo2, temp)
    warnings, _ = check_drugs(curr_drugs.split(","), curr_diseases.split(","), curr_allergies.split(","))

    # 3. DASHBOARD TABS
    tab1, tab2, tab3 = st.tabs(["📊 Analytics & Explainability", "💊 Therapy Pathway", "📄 Export & Email"])
    
    with tab1:
        c_v1, c_v2 = st.columns([1.2, 1])
        with c_v1:
            st.markdown("#### 🎯 Differential Diagnosis (AI Confidence)")
            fig_bar = px.bar(prob_df, x="Prob", y="Condition", orientation='h', color="Prob", 
                             color_continuous_scale=[(0, "green"), (0.5, "yellow"), (1, "red")], template="plotly_dark")
            st.plotly_chart(fig_bar, use_container_width=True)
        with c_v2:
            st.markdown("#### 🕸️ Physiological Radar Fingerprint")
            radar_cats = ['HR', 'SpO2', 'BP Sys', 'Temp', 'Glucose']
            radar_vals = [min(hr/160, 1.0), (100-spo2)/20, min(bps/200, 1.0), min(abs(temp-37)/5, 1.0), min(gluc/400, 1.0)]
            fig_radar = go.Figure(data=go.Scatterpolar(r=radar_vals, theta=radar_cats, fill='toself', 
                                                       line=dict(color='#ff4b4b'), fillcolor='rgba(255, 75, 75, 0.3)'))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1], showticklabels=False)), 
                                    template="plotly_dark", showlegend=False)
            st.plotly_chart(fig_radar, use_container_width=True)

    with tab2:
        st.subheader(f"Standard Protocol: {disease}")
        st.info(f"**Clinical Pathway:** {info['pathway']}")
        user_in = curr_diseases.lower()
        found_sym = False
        for sym, adv in SYMPTOM_DRUGS.items():
            if sym in user_in:
                with st.container(border=True):
                    st.success(f"**For {sym.capitalize()}:** {adv['rec']}")
