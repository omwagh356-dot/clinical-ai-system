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

# --- 2. CLINICAL KNOWLEDGE BASES ---
CLINICAL_DATABASE = {
    "Infection": {"icon": "🤒", "drugs": ["Acetaminophen", "Ceftriaxone", "IV Saline"], "next_steps": "Blood Cultures, CBC, Lactic Acid check.", "safety": "Monitor for Septic Shock."},
    "Respiratory Failure": {"icon": "🫁", "drugs": ["Oxygen", "Albuterol", "Steroids"], "next_steps": "ABG, Chest X-Ray, Intubation Eval.", "safety": "Keep head of bed elevated."},
    "Hypertension": {"icon": "🩸", "drugs": ["Lisinopril", "Amlodipine"], "next_steps": "ECG, Urinalysis, BP monitoring.", "safety": "Risk of Stroke. Avoid sudden movement."},
    "Normal": {"icon": "✅", "drugs": ["Maintain regimen"], "next_steps": "Routine 6-month follow-up.", "safety": "Cleared for activity."},
    "Cardiac Emergency": {"icon": "💔", "drugs": ["Aspirin", "Nitroglycerin"], "next_steps": "12-Lead ECG, Troponin.", "safety": "Minimize movement. Prepare for ACLS."}
}

SYMPTOM_DRUGS = {
    "chest pain": {"rec": "Aspirin (324mg), Nitroglycerin.", "safety": "🚨 CRITICAL: Possible Heart Attack. ER immediately."},
    "fever": {"rec": "Acetaminophen (650mg).", "safety": "✅ Monitor for confusion or stiff neck."},
    "cough": {"rec": "Guaifenesin or Dextromethorphan.", "safety": "⚠️ Avoid suppressants if mucus is thick/green."},
    "diarrhea": {"rec": "Loperamide, ORS.", "safety": "⚠️ Do not use if stool is bloody or fever is high."},
    "headache": {"rec": "Ibuprofen.", "safety": "✅ Seek care if it's the 'worst headache of your life'."},
    "shortness of breath": {"rec": "Oxygen/Albuterol.", "safety": "🚨 EMERGENCY: Monitor SpO2 immediately."}
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
    msg['Subject'] = f"🚨 {report['Risk']} Risk Report: {report['Name']}"
    msg['From'] = st.secrets["EMAIL_USER"]
    msg['To'] = receiver
    body = f"Patient: {report['Name']}\nAge: {report['Age']}\nResult: {report['Disease']} ({report['Prob']}%)\nAnalysis: {reasons}\nWarnings: {warnings}"
    msg.set_content(body)
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(st.secrets["EMAIL_USER"], st.secrets["EMAIL_PASS"])
            smtp.send_message(msg)
        return True
    except: return False

def create_pdf_report(report, info, reasons, warnings):
    return f"""<div style='font-family:Arial; border:2px solid #333; padding:20px;'>
    <h1 style='color:#1a73e8;'>Clinical AI Diagnostic Report</h1>
    <p><b>Patient:</b> {report['Name']} | <b>Age:</b> {report['Age']} | <b>Gender:</b> {report['Gender']}</p>
    <hr>
    <h3>Diagnosis: {report['Disease']} ({report['Prob']}%)</h3>
    <p><b>Vitals Analysis:</b> {", ".join(reasons)}</p>
    <p><b>Safety Alerts:</b> {", ".join(warnings) if warnings else "None"}</p>
    <p><b>Protocol:</b> {info['next_steps']}</p>
    </div>"""

# --- 4. UI INTERFACE ---
st.set_page_config(page_title="Pro-Clinical AI", layout="wide")
model, scaler, label_encoder = load_assets()

st.title("🛡️ Advanced Clinical Decision Support System")

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
    st.subheader("💊 Safety & History")
    curr_drugs = st.text_area("Current Medications (comma separated)")
    curr_diseases = st.text_area("Known Conditions / Symptoms")
    curr_allergies = st.text_area("Allergies")
    
    if st.button("PRE-CHECK DRUG SAFETY", use_container_width=True):
        warnings, recs = check_drugs(curr_drugs.split(","), curr_diseases.split(","), curr_allergies.split(","))
        for w in warnings: st.error(w)
        for r in recs: st.success(r)

# --- 5. EXECUTION ---
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

    # Triage Power Feature
    status, action, color = get_triage_status(disease, prob, spo2, bps)
    st.markdown(f"<div style='background-color:{color}; padding:20px; border-radius:10px; color:white; text-align:center;'><h1>{status}</h1><p>{action}</p></div>", unsafe_allow_html=True)

    info = CLINICAL_DATABASE.get(disease, CLINICAL_DATABASE["Normal"])
    reasons = explain_values(hr, (bps+bpd)/2, spo2, temp)
    warnings, _ = check_drugs(curr_drugs.split(","), curr_diseases.split(","), curr_allergies.split(","))

    # TABS FOR UNIQUE ANALYTICS
    tab1, tab2, tab3 = st.tabs(["📊 Analytics & Explainability", "💊 Therapy", "📄 Export & Email"])
    
    with tab1:
        col_graph1, col_graph2 = st.columns([1.2, 1]) # Adjusting column width
        
        with col_graph1:
            st.markdown("#### 🎯 Differential Diagnosis (AI Confidence)")
            # Using a custom color scale: Green for low prob, Red for high prob
            fig = px.bar(
                prob_df, 
                x="Prob", 
                y="Condition", 
                orientation='h', 
                color="Prob",
                color_continuous_scale=[(0, "green"), (0.5, "yellow"), (1, "red")],
                labels={'Prob': 'Confidence %'},
                template="plotly_dark"
            )
            fig.update_layout(showlegend=False, height=450, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig, use_container_width=True)

        with col_graph2:
            st.markdown("#### 🕸️ Physiological Stress Fingerprint")
            # Normalizing the radar to be more 'dramatic' for abnormal values
            radar_cats = ['HR', 'SpO2', 'BP Sys', 'Temp', 'Glucose']
            # Improved normalization for visibility
            radar_vals = [
                min(hr/160, 1.0), 
                (100-spo2)/20, # Higher value = more 'danger' (Hypoxia)
                min(bps/200, 1.0), 
                min(abs(temp-37)/5, 1.0), # Deviation from normal body temp
                min(gluc/400, 1.0)
            ]
            
            fig_radar = go.Figure(data=go.Scatterpolar(
                r=radar_vals, 
                theta=radar_cats, 
                fill='toself',
                line=dict(color='#ff4b4b', width=2),
                fillcolor='rgba(255, 75, 75, 0.3)'
            ))
            
            fig_radar.update_layout(
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(visible=True, range=[0, 1], showticklabels=False),
                    angularaxis=dict(gridcolor="gray")
                ),
                template="plotly_dark",
                showlegend=False,
                height=450,
                margin=dict(l=40, r=40, t=50, b=40)
            )
            st.plotly_chart(fig_radar, use_container_width=True)
    with tab2:
        user_in = curr_diseases.lower()
        found_sym = False
        for sym, adv in SYMPTOM_DRUGS.items():
            if sym in user_in:
                st.info(f"**For {sym.capitalize()}:** {adv['rec']}\n\n*Safety Check:* {adv['safety']}")
                found_sym = True
        st.subheader(f"Standard Protocol: {disease}")
        for d in info['drugs']: st.success(f"✔️ {d}")

    with tab3:
        report = {"Name": name, "Age": age, "Gender": gender, "Disease": disease, "Prob": round(prob, 2), "Risk": status, "Urgency": action, "vitals": raw_vitals}
        html_doc = create_pdf_report(report, info, reasons, warnings)
        st.download_button("Download Full Report", html_doc, file_name=f"{name}_Report.html", mime="text/html")
        if doc_email and send_to_doctor(doc_email, report, reasons, warnings):
            st.toast("Report Transmitted to MD", icon="📧")
