import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
import joblib
import smtplib
from email.message import EmailMessage

# --- 1. EXTERNAL LOGIC ---
try:
    from drug_module import check_drugs
    from explain import explain_values
except ImportError:
    st.error("⚠️ Ensure drug_module.py and explain.py are in your GitHub repository.")

# --- 2. KNOWLEDGE BASES ---
CLINICAL_DATABASE = {
    "Infection": {"icon": "🤒", "drugs": ["Acetaminophen", "Ceftriaxone"], "next_steps": "Blood Cultures, CBC.", "pathway": "Sepsis Protocol"},
    "Respiratory Failure": {"icon": "🫁", "drugs": ["Oxygen", "Albuterol"], "next_steps": "ABG, Chest X-Ray.", "pathway": "Acute Respiratory Protocol"},
    "Hypertension": {"icon": "🩸", "drugs": ["Lisinopril", "Amlodipine"], "next_steps": "ECG, Urinalysis.", "pathway": "Hypertensive Management"},
    "Normal": {"icon": "✅", "drugs": ["None"], "next_steps": "Routine Checkup.", "pathway": "Standard Wellness"},
    "Cardiac Emergency": {"icon": "💔", "drugs": ["Aspirin", "Nitroglycerin"], "next_steps": "ECG, Troponin.", "pathway": "ACLS Protocol"}
}

SYMPTOM_DRUGS = {
    "chest pain": {"rec": "Aspirin (324mg).", "safety": "🚨 EMERGENCY: Possible Heart Attack."},
    "fever": {"rec": "Acetaminophen.", "safety": "✅ Monitor for confusion."},
    "shortness of breath": {"rec": "Oxygen.", "safety": "🚨 EMERGENCY: Respiratory Failure risk."}
}

# --- 3. CORE FUNCTIONS ---
@st.cache_resource
def load_assets():
    model = load_model("model/model.h5")
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    return model, scaler, label_encoder

model, scaler, label_encoder = load_assets()

def create_pdf_report(report_data, info, reasons, safety_warnings):
    return f"""<div style='font-family:Arial; border:2px solid #333; padding:20px;'>
    <h1>Clinical AI Report: {report_data['Name']}</h1>
    <p><b>Diagnosis:</b> {report_data['Disease']} ({report_data['Prob']}%)</p>
    <p><b>Vitals Analysis:</b> {", ".join(reasons)}</p>
    </div>"""

def send_to_doctor(receiver, report, reasons, warnings):
    msg = EmailMessage()
    msg['Subject'] = f"🚨 {report['Risk']} Risk: {report['Name']}"
    msg['From'] = st.secrets["EMAIL_USER"]
    msg['To'] = receiver
    body = f"Patient: {report['Name']}\nAge: {report['Age']}\nGender: {report['Gender']}\nResult: {report['Disease']}\nAnalysis: {reasons}"
    msg.set_content(body)
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(st.secrets["EMAIL_USER"], st.secrets["EMAIL_PASS"])
            smtp.send_message(msg)
        return True
    except: return False

# --- 4. UI LAYOUT ---
st.set_page_config(page_title="Advanced Clinical AI", layout="wide")
st.title("🛡️ Next-Gen Clinical Decision Support System")

c1, c2 = st.columns([1, 1])
with c1:
    st.subheader("👤 Patient Identity")
    name = st.text_input("Full Name")
    doc_email = st.text_input("Doctor's Email")
    g_col, a_col = st.columns(2)
    gender = g_col.selectbox("Gender", ["Male", "Female", "Other"])
    age = a_col.number_input("Age", 1, 120, 30)

    st.subheader("📉 Real-time Vitals")
    v1, v2, v3 = st.columns(3)
    hr = v1.number_input("Heart Rate", 40.0, 200.0, 72.0)
    bps = v1.number_input("BP Systolic", 70.0, 240.0, 120.0)
    resp = v1.number_input("Resp. Rate", 0.0, 50.0, 16.0)
    spo2 = v2.number_input("SpO2 %", 50.0, 100.0, 98.0)
    bpd = v2.number_input("BP Diastolic", 40.0, 140.0, 80.0)
    chol = v2.number_input("Cholesterol", 100.0, 400.0, 190.0)
    temp = v3.number_input("Temp °C", 34.0, 42.0, 37.0)
    gluc = v3.number_input("Glucose", 40.0, 600.0, 95.0)

with c2:
    st.subheader("🧪 Contextual History")
    curr_drugs = st.text_area("Current Medications")
    curr_diseases = st.text_area("Reported Symptoms / Known Conditions")
    curr_allergies = st.text_area("Allergies")
    
    if st.button("PRE-CHECK DRUG SAFETY", use_container_width=True):
        warnings, _ = check_drugs(curr_drugs.split(","), curr_diseases.split(","), curr_allergies.split(","))
        for w in warnings: st.error(w)

# --- 5. INNOVATIVE ANALYSIS ---
st.divider()
if st.button("🚀 EXECUTE MULTIMODAL DIAGNOSTIC", type="primary", use_container_width=True):
    # ML Prediction
    raw_vitals = [age, hr, bps, bpd, spo2, temp, chol, gluc, resp]
    inputs_df = pd.DataFrame([raw_vitals], columns=scaler.feature_names_in_)
    scaled = scaler.transform(inputs_df)
    pred = model.predict(scaled, verbose=0)
    idx = np.argmax(pred)
    disease = label_encoder.inverse_transform([idx])[0]
    prob = pred[0][idx] * 100

    # Triage Overrides
    risk, urgency = ("High", "IMMEDIATE ER") if (spo2 < 90 or temp > 39.5 or bps >= 180) else ("Low", "Routine")
    
    info = CLINICAL_DATABASE.get(disease, CLINICAL_DATABASE["Normal"])
    reasons = explain_values(hr, (bps+bpd)/2, spo2, temp)
    warnings, _ = check_drugs(curr_drugs.split(","), curr_diseases.split(","), curr_allergies.split(","))

    st.header(f"{info['icon']} Primary Diagnosis: {disease}")
    
    tab1, tab2, tab3 = st.tabs(["📊 Analytics & Explainability", "💊 Therapy & Pharmacy", "📄 Export & Email"])

    with tab1:
        col_graph1, col_graph2 = st.columns(2)
        
        with col_graph1:
            st.write("**Differential Diagnosis (AI Confidence)**")
            prob_df = pd.DataFrame({"Condition": label_encoder.classes_, "Probability": pred[0] * 100}).sort_values("Probability")
            fig = px.bar(prob_df, x="Probability", y="Condition", orientation='h', color="Probability", color_continuous_scale="Viridis")
            st.plotly_chart(fig, use_container_width=True)

        with col_graph2:
            st.write("**Physiological Radar (Patient Stress Fingerprint)**")
            radar_cats = ['HR', 'SpO2', 'BP Sys', 'Temp', 'Glucose']
            radar_vals = [hr/150, spo2/100, bps/200, (temp-30)/10, gluc/300]
            fig_radar = go.Figure(data=go.Scatterpolar(r=radar_vals, theta=radar_cats, fill='toself', name='Patient State'))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=False)
            st.plotly_chart(fig_radar, use_container_width=True)

    with tab2:
        st.subheader("Clinical Pathway & Meds")
        st.info(f"**Recommended Pathway:** {info['pathway']}")
        
        # Smart Symptom Mapping
        user_in = curr_diseases.lower()
        found_sym = False
        for sym, adv in SYMPTOM_DRUGS.items():
            if sym in user_in:
                st.success(f"**{sym.capitalize()} Management:** {adv['rec']}")
                st.warning(f"**Safety:** {adv['safety']}")
                found_sym = True
        
        st.write("**Predicted Standard Protocol:**")
        cols = st.columns(len(info['drugs']))
        for i, d in enumerate(info['drugs']): cols[i].button(d, disabled=True, key=f"d_{i}")

    with tab3:
        report_data = {"Name": name, "Age": age, "Gender": gender, "Disease": disease, "Prob": round(prob, 2), "Risk": risk, "Urgency": urgency, "Symptoms": curr_diseases, "vitals": raw_vitals}
        html_doc = create_pdf_report(report_data, info, reasons, warnings)
        st.download_button("Download Full Clinical Report", html_doc, file_name=f"{name}_Report.html", mime="text/html")
        
        if doc_email:
            if send_to_doctor(doc_email, report_data, reasons, warnings):
                st.toast(f"Report for {name} emailed!", icon="📧")
