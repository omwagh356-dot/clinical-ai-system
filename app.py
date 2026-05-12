import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import smtplib
from email.message import EmailMessage
import os
import shap
# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Clinical AI Decision Support System",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.stApp{
    background: linear-gradient(to right,#0f2027,#203a43,#2c5364);
    color:white;
}

.main-title{
    font-size:42px;
    font-weight:bold;
    color:white;
}

.status-box{
    padding:20px;
    border-radius:15px;
    text-align:center;
    font-size:28px;
    font-weight:bold;
    margin-top:20px;
    margin-bottom:20px;
}

.med-card{
    background:rgba(0,0,0,0.35);
    padding:15px;
    border-radius:12px;
    margin-bottom:10px;
    border-left:5px solid #00ff99;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# CHECK FILES
# =========================================================

required_files = [
    "model.pkl",
    "scaler.pkl",
    "label_encoder.pkl",
    "features.pkl"
]

for file in required_files:

    if not os.path.exists(file):

        st.error(f"❌ Missing file: {file}")

        st.stop()

# =========================================================
# LOAD FILES
# =========================================================

model = joblib.load("model.pkl")

scaler = joblib.load("scaler.pkl")

label_encoder = joblib.load("label_encoder.pkl")

features = joblib.load("features.pkl")

# =========================================================
# LOAD MEDICINE DATABASE
# =========================================================

@st.cache_data
def load_medicine_db():

    try:

        df = pd.read_excel(
            "Medicine_description.xlsx"
        )

        df.columns = [
            c.strip()
            for c in df.columns
        ]

        if "res" in df.columns:

            df = df.rename(
                columns={
                    "res":"Reason"
                }
            )

        df["Reason"] = (
            df["Reason"]
            .astype(str)
        )

        return df

    except:

        return pd.DataFrame(
            columns=[
                "Drug_Name",
                "Reason",
                "Description"
            ]
        )

med_db = load_medicine_db()

# =========================================================
# ADVANCED SYMPTOM NLP ENGINE
# =========================================================

def encode_symptoms(
    text,
    feature_list
):

    text = text.lower().strip()

    # -----------------------------------------------------
    # CLINICAL NLP DICTIONARY
    # -----------------------------------------------------

    symptom_map = {

        "fever": [
            "fever",
            "high fever",
            "body hot",
            "temperature",
            "high temperature"
        ],

        "cough": [
            "cough",
            "coughing",
            "dry cough"
        ],

        "headache": [
            "headache",
            "migraine",
            "head pain"
        ],

        "chest_pain": [
            "chest pain",
            "tight chest",
            "heart pain",
            "pain in chest"
        ],

        "shortness_of_breath": [
            "difficulty breathing",
            "breathing problem",
            "breathlessness",
            "shortness of breath"
        ],

        "rash": [
            "rash",
            "skin rash",
            "red spots",
            "skin allergy"
        ],

        "fatigue": [
            "fatigue",
            "weakness",
            "tired",
            "exhausted"
        ],

        "vomiting": [
            "vomiting",
            "nausea",
            "throwing up"
        ],

        "dizziness": [
            "dizziness",
            "dizzy",
            "lightheaded"
        ],

        "body_pain": [
            "body pain",
            "muscle pain",
            "body ache"
        ],

        "cold": [
            "cold",
            "runny nose",
            "blocked nose"
        ]
    }

    # -----------------------------------------------------
    # CREATE FEATURE VECTOR
    # -----------------------------------------------------

    vector = []

    vital_features = [
        "age",
        "hr",
        "bp",
        "spo2",
        "temp",
        "glucose"
    ]

    for feature in feature_list:

        # Skip vitals
        if feature in vital_features:
            continue

        found = 0

        # -------------------------------------------------
        # NLP MAPPING
        # -------------------------------------------------

        if feature in symptom_map:

            for keyword in symptom_map[feature]:

                if keyword in text:

                    found = 1
                    break

        # -------------------------------------------------
        # FALLBACK MATCHING
        # -------------------------------------------------

        else:

            clean_feature = (
                feature
                .replace("_"," ")
            )

            if clean_feature in text:

                found = 1

        vector.append(found)

    return vector

# =========================================================
# EMAIL FUNCTION
# =========================================================

def send_email(
    receiver,
    patient_name,
    disease,
    status
):

    try:

        msg = EmailMessage()

        msg["Subject"] = (
            f"🚨 Clinical Alert - {status}"
        )

        msg["From"] = (
            st.secrets["EMAIL_USER"]
        )

        msg["To"] = receiver

        msg.set_content(f"""
Patient Name : {patient_name}

Clinical Assessment : {disease}

Status : {status}
        """)

        with smtplib.SMTP_SSL(
            "smtp.gmail.com",
            465
        ) as smtp:

            smtp.login(
                st.secrets["EMAIL_USER"],
                st.secrets["EMAIL_PASS"]
            )

            smtp.send_message(msg)

        return True

    except:

        return False

# =========================================================
# REPORT FUNCTION
# =========================================================

def generate_report(
    name,
    ml_prediction,
    clinical_prediction,
    confidence,
    severity,
    risk,
    status
):

    return f"""
    <html>

    <body>

    <h1>Clinical AI Report</h1>

    <hr>

    <h2>Patient Information</h2>

    <p><b>Name:</b> {name}</p>

    <h2>Prediction</h2>

    <p><b>ML Prediction:</b>
    {ml_prediction}</p>

    <p><b>Clinical Assessment:</b>
    {clinical_prediction}</p>

    <p><b>Confidence:</b>
    {confidence}%</p>

    <p><b>Severity:</b>
    {severity}</p>

    <p><b>Risk Score:</b>
    {risk}</p>

    <p><b>Status:</b>
    {status}</p>

    </body>

    </html>
    """

# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class='main-title'>

🛡️ Clinical AI Decision Support System

</div>
""", unsafe_allow_html=True)

st.caption(
    "Research-Level Explainable Clinical Intelligence"
)

st.caption(
    "MSc Data Science Project | Onkar Suresh Wagh"
)

# =========================================================
# USER INPUTS
# =========================================================

col1, col2 = st.columns(2)

with col1:

    name = st.text_input(
        "Patient Name"
    )

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=30
    )

    hr = st.number_input(
        "Heart Rate",
        value=72.0
    )

    bp = st.number_input(
        "Blood Pressure",
        value=120.0
    )

with col2:

    spo2 = st.number_input(
        "SpO2",
        value=98.0
    )

    temp = st.number_input(
        "Temperature",
        value=37.0
    )

    gluc = st.number_input(
        "Glucose",
        value=90.0
    )

    email = st.text_input(
        "Doctor Email"
    )

symptoms = st.text_area(
    "Symptoms (Example: fever, chest pain, cough)"
)

# =========================================================
# RUN BUTTON
# =========================================================

if st.button("🚀 Run Diagnosis"):

    symptom_text = symptoms.lower()

    # =====================================================
    # ENCODE SYMPTOMS
    # =====================================================

    symptom_vector = encode_symptoms(
        symptoms,
        features
    )

    # =====================================================
    # VITALS
    # =====================================================

    vitals = [
        age,
        hr,
        bp,
        spo2,
        temp,
        gluc
    ]

    input_data = (
        symptom_vector + vitals
    )

    # =====================================================
    # CREATE DATAFRAME
    # =====================================================

    input_df = pd.DataFrame(
        [input_data],
        columns=features
    )

    # =====================================================
    # FIX FEATURE ALIGNMENT
    # =====================================================

    expected_features = (
        scaler.feature_names_in_
    )

    for col in expected_features:

        if col not in input_df.columns:

            input_df[col] = 0

    input_df = input_df[
        expected_features
    ]

    # =====================================================
    # SCALE INPUT
    # =====================================================

    scaled = scaler.transform(
        input_df
    )

    # =====================================================
    # MODEL PREDICTION
    # =====================================================

    prob = model.predict_proba(
        scaled
    )

    pred_index = np.argmax(
        prob[0]
    )

    ml_prediction = (
        label_encoder
        .inverse_transform(
            [pred_index]
        )[0]
    )

    confidence = float(
        prob[0][pred_index] * 100
    )

    # =====================================================
    # CLINICAL OVERRIDE ENGINE
    # =====================================================

    clinical_prediction = (
        ml_prediction
    )

    override_reason = None

    # -----------------------------------------------------
    # CARDIAC RISK
    # -----------------------------------------------------

    if (
        hr >= 145
        or "chest pain" in symptom_text
    ):

        clinical_prediction = (
            "Cardiac Risk"
        )

        confidence = max(
            confidence,
            96
        )

        override_reason = (
            "Extreme tachycardia / chest pain detected"
        )

    # -----------------------------------------------------
    # DIABETES
    # -----------------------------------------------------

    elif gluc > 200:

        clinical_prediction = (
            "Diabetes"
        )

        confidence = max(
            confidence,
            95
        )

        override_reason = (
            "High glucose detected"
        )

    # -----------------------------------------------------
    # FEVER
    # -----------------------------------------------------

    elif temp >= 39:

        clinical_prediction = (
            "Fever"
        )

        confidence = max(
            confidence,
            90
        )

        override_reason = (
            "Very high temperature"
        )

    # -----------------------------------------------------
    # RESPIRATORY
    # -----------------------------------------------------

    elif spo2 < 90:

        clinical_prediction = (
            "Respiratory Disease"
        )

        confidence = max(
            confidence,
            92
        )

        override_reason = (
            "Low oxygen saturation"
        )

    # -----------------------------------------------------
    # ALLERGY
    # -----------------------------------------------------

    elif "rash" in symptom_text:

        clinical_prediction = (
            "Allergy"
        )

        confidence = max(
            confidence,
            85
        )

        override_reason = (
            "Rash symptom detected"
        )

    # =====================================================
    # RISK SCORE
    # =====================================================

    risk = 0

    if hr >= 145:
        risk += 3

    if "chest pain" in symptom_text:
        risk += 3

    if temp > 39:
        risk += 2

    if spo2 < 90:
        risk += 3

    if gluc > 200:
        risk += 2

    # =====================================================
    # SEVERITY
    # =====================================================

    severity = "Mild"

    if risk >= 6:

        severity = "Critical"

    elif risk >= 4:

        severity = "Severe"

    elif risk >= 2:

        severity = "Moderate"

    # =====================================================
    # STATUS
    # =====================================================

    status = "🟢 STABLE"

    if severity in [
        "Severe",
        "Critical"
    ]:

        status = "🔴 CRITICAL"

    # =====================================================
    # RESULT COLOR
    # =====================================================

    color = (
        "#28a745"
        if "STABLE" in status
        else "#ff4b4b"
    )

    # =====================================================
    # DISPLAY RESULTS
    # =====================================================

    st.markdown(f"""
    <div class='status-box'
    style='background:{color};'>

    <h2>🧠 AI Clinical Decision</h2>

    <hr>

    <h3>
    ML Model Prediction :
    {ml_prediction}
    ({round(confidence,2)}%)
    </h3>

    <br>

    <h2>
    🏥 Final Clinical Assessment :
    {clinical_prediction}
    </h2>

    </div>
    """, unsafe_allow_html=True)


    # =====================================================
    # METRICS
    # =====================================================

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "⚠️ Risk Score",
            risk
        )

    with c2:

        st.metric(
            "🔥 Severity",
            severity
        )

    with c3:

        st.metric(
            "🧠 Confidence",
            f"{round(confidence,2)}%"
        )

    # =====================================================
    # ALERTS
    # =====================================================

    if severity == "Critical":

        st.error("""
🚨 EMERGENCY ALERT

Immediate medical attention recommended.
        """)

    elif severity == "Severe":

        st.warning("""
⚠️ HIGH RISK PATIENT

Urgent physician consultation recommended.
        """)

    # =====================================================
    # EMAIL ALERT
    # =====================================================

    if email:

        send_email(
            email,
            name,
            clinical_prediction,
            status
        )

    # =====================================================
    # TABS
    # =====================================================

    tab1, tab2, tab3 = st.tabs([
        "📊 Dashboard",
        "🔍 Explainability",
        "💊 Treatment"
    ])

    # =====================================================
    # DASHBOARD
    # =====================================================

    with tab1:

        prob_df = pd.DataFrame({

            "Disease":
            label_encoder.classes_,

            "Probability":
            prob[0] * 100
        })

        fig = px.bar(
            prob_df.sort_values(
                by="Probability",
                ascending=True
            ),
            x="Probability",
            y="Disease",
            orientation='h',
            text="Probability",
            title="Disease Probability Distribution"
        )

        fig.update_layout(
            template="plotly_dark"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # =====================================================
    # EXPLAINABILITY
    # =====================================================

    with tab2:

        st.subheader(
            "Clinical Reasoning"
        )

        reasons = []

        if hr >= 145:

            reasons.append(
                "Extreme tachycardia detected"
            )

        if "chest pain" in symptom_text:

            reasons.append(
                "Chest pain indicates cardiac risk"
            )

        if gluc > 200:

            reasons.append(
                "High glucose indicates diabetes risk"
            )

        if temp > 38:

            reasons.append(
                "High temperature suggests infection"
            )

        if spo2 < 92:

            reasons.append(
                "Low oxygen saturation detected"
            )

        if override_reason:

            reasons.append(
                f"Override Applied: "
                f"{override_reason}"
            )

        for reason in reasons:

            st.success(reason)

    # =====================================================
    # TREATMENT
    # =====================================================

    with tab3:

        st.subheader(
            "💊 Recommended Medicines"
        )

        search_terms = [
            clinical_prediction.lower()
        ]

        query = "|".join(
            search_terms
        )

        meds = med_db[
            med_db["Reason"]
            .str.lower()
            .str.contains(
                query,
                na=False
            )
        ]

        if meds.empty:

            meds = med_db.head(5)

        for _, row in meds.head(10).iterrows():

            st.markdown(f"""
            <div class='med-card'>

            <b>{row['Drug_Name']}</b><br>

            <i>{row['Reason']}</i><br><br>

            <small>
            {row['Description']}
            </small>

            </div>
            """, unsafe_allow_html=True)

    # =====================================================
    # REPORT DOWNLOAD
    # =====================================================

    report = generate_report(
        name,
        ml_prediction,
        clinical_prediction,
        round(confidence,2),
        severity,
        risk,
        status
    )

    st.download_button(
        "📄 Download Clinical Report",
        report,
        file_name=f"{name}_report.html",
        mime="text/html"
    )
