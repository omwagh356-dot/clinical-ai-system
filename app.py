import os
import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import smtplib
from email.message import EmailMessage
import base64

# --- 1. EXTERNAL LOGIC IMPORTS ---
try:
    from drug_module import check_drugs
    from explain import explain_values
except ImportError:
    st.error("⚠️ Missing 'drug_module.py' or 'explain.py' on GitHub.")

# --- 2. Adding some databases of medicine and disease symptoms ---
@st.cache_data
def load_clinical_data():
    # Load symptoms dataset
    symptom_df = pd.read_csv('DiseaseAndSymptoms.csv')
    
    # Load medicine dataset
    med_df = pd.read_csv('Medicine_description.xlsx - Sheet1.csv')
    
    # Pre-process symptoms: create a mapping of Disease -> List of Symptoms
    # This makes searching much faster
    disease_symptom_map = {}
    for _, row in symptom_df.iterrows():
        disease = row['Disease']
        # Collect all non-null symptoms for this row
        symptoms = [str(s).strip().replace("_", " ").lower() for s in row[1:] if pd.notna(s)]
        if disease not in disease_symptom_map:
            disease_symptom_map[disease] = set(symptoms)
        else:
            disease_symptom_map[disease].update(symptoms)
            
    return disease_symptom_map, med_df

# Call this at the start of your app
disease_map, medicine_db = load_clinical_data()
# 3 adding some Symptom based Disease prediction system
def predict_disease_from_symptoms(user_input_string):
    user_symptoms = [s.strip().lower() for s in user_input_string.split(",")]
    scores = {}
    
    for disease, symptoms_set in disease_map.items():
        # Count how many user symptoms match the symptoms for this disease
        match_count = len(set(user_symptoms).intersection(symptoms_set))
        if match_count > 0:
            scores[disease] = match_count
            
    if not scores:
        return "Normal", 0
    
    # Return the disease with the highest match count
    best_match = max(scores, key=scores.get)
    return best_match, scores[best_match]
# --- 2. KNOWLEDGE BASES
#SYMPTOM_DRUGS = {
 #   "chest pain": {"rec": "Aspirin (324mg), Nitroglycerin.", "safety": "🚨 CRITICAL: High risk of Heart Attack. Proceed to ER."},
  #  "cough": {"rec": "Dextromethorphan or Guaifenesin.", "safety": "⚠️ Avoid suppressants if cough is productive with fever."},
   # "fever": {"rec": "Acetaminophen (650mg) or Ibuprofen.", "safety": "✅ Monitor for stiff neck or confusion."},
    #"cold": {"rec": "Decongestants and Vitamin C.", "safety": "⚠️ Decongestants raise Blood Pressure."},
    #"diarrhea": {"rec": "Loperamide and ORS.", "safety": "⚠️ Do not use if stool is bloody or fever is high."},
    #"headache": {"rec": "Ibuprofen or Naproxen.", "safety": "✅ Seek care if it is the 'worst headache of your life'."},
    #"nausea": {"rec": "Ginger or Ondansetron.", "safety": "⚠️ Persistent vomiting leads to Electrolyte Imbalance."},
    #"shortness of breath": {"rec": "Albuterol or Oxygen.", "safety": "🚨 EMERGENCY: High risk of Respiratory Failure."},
    #"dizziness": {"rec": "Meclizine or hydration.", "safety": "⚠️ Risk of Stroke if slurred speech is present."}
#}

#CLINICAL_DATABASE = {
   # "Infection": {"icon": "🤒", "severity": "High", "drugs": ["Acetaminophen", "Ceftriaxone"], "next_steps": "Blood Cultures, CBC.", "safety": "Monitor for Septic Shock."},
   # "Respiratory Failure": {"icon": "🫁", "severity": "Critical", "drugs": ["Oxygen", "Albuterol"], "next_steps": "ABG, Chest X-Ray.", "safety": "Keep head of bed elevated."},
   # "Hypertension": {"icon": "🩸", "severity": "High", "drugs": ["Lisinopril", "Amlodipine"], "next_steps": "ECG, Urinalysis.", "safety": "Risk of Stroke."},
  #  "Normal": {"icon": "✅", "severity": "Stable", "drugs": ["Maintain regimen"], "next_steps": "Routine follow-up.", "safety": "Cleared for standard activity."},
 #   "Cardiac Emergency": {"icon": "💔", "severity": "Critical", "drugs": ["Aspirin", "Nitroglycerin"], "next_steps": "12-Lead ECG.", "safety": "Minimize movement."}
#}
# 4 new function for drug recommend

def get_recommendations(disease_name):
    # Filter the medicine database where "Reason" matches our disease
    # We use case-insensitive matching
    recommendations = medicine_db[medicine_db['Reason'].str.contains(disease_name, case=False, na=False)]
    
    if recommendations.empty:
        return []
    
    # Return a list of dictionaries with Drug Name and Description
    return recommendations[['Drug_Name', 'Description']].head(5).to_dict('records')


# --- 3. PAGE CONFIG & ASSETS ---
st.set_page_config(page_title="Clinical AI Portal", page_icon="🏥", layout="wide")

@st.cache_resource
def load_assets():
    model = load_model("model/model.h5")
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    return model, scaler, label_encoder

model, scaler, label_encoder = load_assets()

# --- 4. CORE FUNCTIONS ---
def create_pdf_report(report_data, info, reasons, safety_warnings):
    html_content = f"""
    <div style="font-family: Arial; padding: 20px; border: 2px solid #333;">
        <h1 style="color: #1a73e8; text-align: center;">Clinical AI Diagnostic Report</h1>
        <hr>
        <p><b>PATIENT NAME:</b> {report_data['Name']}</p>
        <p><b>AGE:</b> {report_data['Age']} | <b>GENDER:</b> {report_data['Gender']}</p>
        <div style="background: #f8f9fa; padding: 15px; border-radius: 10px;">
            <p><b>Prediction:</b> {report_data['Disease']} ({report_data['Prob']}%)</p>
            <p><b>Urgency:</b> {report_data['Urgency']}</p>
        </div>
        <h3>Analysis & Safety</h3>
        <ul>{"".join([f"<li>🚩 {r}</li>" for r in reasons])}</ul>
        <p><b>Meds:</b> {", ".join(info['drugs'])}</p>
    </div>
    """
    return html_content

def send_to_doctor(receiver_email, report, reasons, safety_warnings):
    msg = EmailMessage()
    msg['Subject'] = f"🚨 {report['Risk']} Risk Report - {report['Name']}"
    msg['From'] = st.secrets["EMAIL_USER"]
    msg['To'] = receiver_email
    
    body = f"""
    OFFICIAL CLINICAL REPORT
    -------------------------------
    PATIENT DETAILS:
    - Full Name: {report['Name']}
    - Age: {report['Age']}
    - Gender: {report['Gender']}
    
    DIAGNOSTIC SUMMARY:
    - Result: {report['Disease']} ({report['Prob']}%)
    - Urgency: {report['Urgency']}
    
    SYMPTOMS & CONDITIONS:
    {report['Symptoms']}

    VITALS ANALYSIS:
    {chr(10).join(['- ' + r for r in reasons])}

    DRUG SAFETY WARNINGS:
    {chr(10).join(['- ' + w for w in safety_warnings]) if safety_warnings else "No interactions flagged."}
    -------------------------------
    Generated via Clinical AI. MD Verification required.
    """
    msg.set_content(body)
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(st.secrets["EMAIL_USER"], st.secrets["EMAIL_PASS"])
            smtp.send_message(msg)
        return True
    except: return False

# --- 5. UI LAYOUT ---
st.title("🏥 Clinical AI Diagnostic Dashboard")

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("👤 Patient Identity")
    name = st.text_input("Full Name (Patient)")
    doc_email = st.text_input("Doctor's Email")
    g_col, a_col = st.columns(2)
    gender = g_col.selectbox("Gender", ["Male", "Female", "Other"])
    age = a_col.number_input("Age", value=30)

    st.subheader("📉 Clinical Vitals")
    v1, v2, v3 = st.columns(3)
    hr = v1.number_input("Heart Rate", value=72.0)
    bps = v1.number_input("BP Systolic", value=120.0)
    resp = v1.number_input("Resp. Rate", value=16.0)
    spo2 = v2.number_input("SpO2 %", value=98.0)
    bpd = v2.number_input("BP Diastolic", value=80.0)
    chol = v2.number_input("Cholesterol", value=190.0)
    temp = v3.number_input("Temp °C", value=37.0)
    gluc = v3.number_input("Glucose", value=95.0)

with col_right:
    st.subheader("💊 Medication Safety")
    curr_drugs = st.text_area("Current Medications (comma separated)")
    curr_diseases = st.text_area("Known Conditions / Symptoms")
    curr_allergies = st.text_area("Allergies")
    
    if st.button("CHECK DRUG INTERACTIONS", use_container_width=True):
        warnings_check, recs_check = check_drugs(curr_drugs.split(","), curr_diseases.split(","), curr_allergies.split(","))
        for w in warnings_check: st.error(w)
        for r in recs_check: st.success(r)

# --- 6. EXECUTION BLOCK ---
st.divider()
if st.button("RUN FULL DIAGNOSTIC", type="primary", use_container_width=True):
    # ML Prediction
    inputs_raw = [age, hr, bps, bpd, spo2, temp, chol, gluc, resp]
    inputs_df = pd.DataFrame([inputs_raw], columns=scaler.feature_names_in_)
    scaled = scaler.transform(inputs_df)
    pred = model.predict(scaled, verbose=0)
    idx = np.argmax(pred)
    disease = label_encoder.inverse_transform([idx])[0]
    prob = pred[0][idx] * 100

    # Triage Overrides
    risk, urgency = "Low", "Routine follow-up"
    if spo2 < 90 or temp > 39.5 or bps >= 180:
        risk, urgency = "High", "IMMEDIATE ER VISIT REQUIRED"

    # Data Processing
    info = CLINICAL_DATABASE.get(disease, CLINICAL_DATABASE["Normal"])
    reasons = explain_values(hr, (bps+bpd)/2, spo2, temp)
    warnings, _ = check_drugs(curr_drugs.split(","), curr_diseases.split(","), curr_allergies.split(","))

    # Unified Report Data
    report_data = {
        "Name": name if name else "Unknown Patient",
        "Age": age,
        "Gender": gender,
        "Disease": disease,
        "Prob": f"{prob:.2f}",
        "Risk": risk,
        "Urgency": urgency,
        "Symptoms": curr_diseases if curr_diseases else "None reported",
        "vitals": inputs_raw
    }

    # UI Result Display
    st.header(f"{info['icon']} Diagnosis: {disease}")
    tab1, tab2, tab3 = st.tabs(["🎯 Results", "💊 Therapy", "📄 Export & Email"])
    
    with tab1:
        st.metric("AI Confidence", f"{prob:.2f}%", delta=risk)
        st.write(f"**Urgency:** {urgency}")
        st.write("**Reasoning:**", ", ".join(reasons))
        
    with tab2:
        st.subheader("🔍 Symptom Analysis & Recommendations")
    
        if curr_diseases:
        # 1. Predict Disease from symptoms entered in the text area
            predicted_disease, score = predict_disease_from_symptoms(curr_diseases)
        
            if score > 0:
                st.info(f"**Detected Condition:** {predicted_disease} (Matches {score} symptoms)")
            
            # 2. Get real drugs for this condition from your Medicine_description CSV
                drug_recs = get_recommendations(predicted_disease)
            
                if drug_recs:
                    st.write("### Recommended Medications (from Database)")
                    for drug in drug_recs:
                        with st.expander(f"💊 {drug['Drug_Name']}"):
                            st.write(f"**Description:** {drug['Description']}")
                else:
                    st.warning("No specific drugs found in the database for this condition.")
            else:
                st.write("Type symptoms (e.g., 'itching, skin rash') to see recommendations.")

    with tab3:
        html_report = create_pdf_report(report_data, info, reasons, warnings)
        st.download_button("Download Medical Report (HTML)", data=html_report, file_name=f"Report_{name}.html", mime="text/html", use_container_width=True)

        if doc_email:
            with st.spinner("Emailing doctor..."):
                if send_to_doctor(doc_email, report_data, reasons, warnings):
                    st.success(f"Report for {name} sent to {doc_email}! ✅")
                else:
                    st.error("Email failed. Check your Secrets.")
