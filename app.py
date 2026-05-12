import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import smtplib
from email.message import EmailMessage
import os

# =========================================================
# CONFIG
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
    font-size:30px;
    font-weight:bold;
    margin-top:20px;
    margin-bottom:20px;
}

.metric-card{
    background:rgba(255,255,255,0.08);
    padding:20px;
    border-radius:12px;
    text-align:center;
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
        df = pd.read_excel("Medicine_description.xlsx")

        df.columns = [c.strip() for c in df.columns]

        if "res" in df.columns:
            df = df.rename(columns={"res": "Reason"})

        df["Reason"] = df["Reason"].astype(str)

        return df

    except:
        return pd.DataFrame(
            columns=["Drug_Name", "Reason", "Description"]
        )

med_db = load_medicine_db()

# =========================================================
# ENCODE SYMPTOMS
# =========================================================
def encode_symptoms(text, feature_list):

    text = text.lower()

    vector = []

    for feature in feature_list:

        if feature in [
            "age",
            "hr",
            "bp",
            "spo2",
            "temp",
            "glucose"
        ]:
            continue

        words = feature.replace("_", " ").split()

        if any(word in text for word in words):
            vector.append(1)
        else:
            vector.append(0)

    return vector

# =========================================================
# EMAIL ALERT
# =========================================================
def send_email(receiver, name, disease, status):

    try:

        msg = EmailMessage()

        msg["Subject"] = f"🚨 Clinical Alert - {status}"
        msg["From"] = st.secrets["EMAIL_USER"]
        msg["To"] = receiver

        msg.set_content(f"""
Patient Name : {name}

Diagnosis : {disease}

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
# GENERATE REPORT
# =========================================================
def generate_report(
    name,
    disease,
    confidence,
    status,
    severity,
    risk
):

    return f"""
    <html>
    <body>

    <h1>Clinical AI Report</h1>

    <hr>

    <h2>Patient Information</h2>

    <p><b>Name:</b> {name}</p>

    <h2>Diagnosis</h2>

    <p><b>Disease:</b> {disease}</p>

    <p><b>Confidence:</b> {confidence}%</p>

    <p><b>Status:</b> {status}</p>

    <p><b>Severity:</b> {severity}</p>

    <p><b>Risk Score:</b> {risk}</p>

    </body>
    </html>
    """

# =========================================================
# HEADER
# =========================================================
st.markdown(
    "<div class='main-title'>🛡️ Clinical AI Decision Support System</div>",
    unsafe_allow_html=True
)

st.caption(
    "Research-Level Explainable Clinical Intelligence System"
)

st.caption(
    "MSc Data Science Project | Onkar Suresh Wagh"
)

# =========================================================
# INPUT UI
# =========================================================
col1, col2 = st.columns(2)

with col1:

    name = st.text_input("Patient Name")

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
    "Symptoms (example: fever, cough, headache, rash)"
)

# =========================================================
# RUN DIAGNOSIS
# =========================================================
if st.button("🚀 Run Diagnosis"):

    symptom_text = symptoms.lower()

    # =====================================================
    # FEATURE ENGINEERING
    # =====================================================
    symptom_vector = encode_symptoms(
        symptoms,
        features
    )

    vitals = [
        age,
        hr,
        bp,
        spo2,
        temp,
        gluc
    ]

    input_data = symptom_vector + vitals

    input_df = pd.DataFrame(
        [input_data],
        columns=features
    )

    # =====================================================
    # SCALE DATA
    # =====================================================
    scaled = scaler.transform(input_df)

    # =====================================================
    # PREDICTION
    # =====================================================
    prob = model.predict_proba(scaled)

    pred_index = np.argmax(prob[0])

    disease = label_encoder.inverse_transform(
        [pred_index]
    )[0]

    confidence = float(
        prob[0][pred_index] * 100
    )


    # =====================================================
    # CLINICAL OVERRIDE SYSTEM
    # =====================================================
    override_reason = None

    if gluc > 200:
        disease = "Diabetes"
        confidence = max(confidence, 95)
        override_reason = "High glucose detected"

    elif temp >= 39:
        disease = "Fever"
        confidence = max(confidence, 90)
        override_reason = "High body temperature"

    elif spo2 < 90:
        disease = "Respiratory Disease"
        confidence = max(confidence, 92)
        override_reason = "Low oxygen saturation"

    elif "rash" in symptom_text:
        disease = "Allergy"
        confidence = max(confidence, 85)
        override_reason = "Skin rash detected"

    # =====================================================
    # RISK SCORE
    # =====================================================
    risk = 0

    if temp > 39:
        risk += 2

    if spo2 < 90:
        risk += 3

    if gluc > 200:
        risk += 2

    severity = "Mild"

    if risk >= 4:
        severity = "Severe"

    elif risk >= 2:
        severity = "Moderate"

    # =====================================================
    # STATUS
    # =====================================================
    status = "🟢 STABLE"

    if severity == "Severe":
        status = "🔴 CRITICAL"

    status_color = (
        "#28a745"
        if "STABLE" in status
        else "#ff4b4b"
    )

    # =====================================================
    # DISPLAY RESULT
    # =====================================================
    st.markdown(f"""
    <div class='status-box'
    style='background:{status_color};'>

    {status} : {disease} ({round(confidence,2)}%)

    </div>
    """, unsafe_allow_html=True)

    # =====================================================
    # METRICS
    # =====================================================
    m1, m2, m3 = st.columns(3)

    with m1:
        st.metric(
            "⚠️ Risk Score",
            risk
        )

    with m2:
        st.metric(
            "🔥 Severity",
            severity
        )

    with m3:
        st.metric(
            "🧠 Confidence",
            f"{round(confidence,2)}%"
        )

    # =====================================================
    # SEND EMAIL
    # =====================================================
    if email:
        send_email(
            email,
            name,
            disease,
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

        st.subheader(
            "Disease Probability Distribution"
        )

        # =============================================
        # IMPORTANT FIXED PROBABILITY MAPPING
        # =============================================
        decoded_labels = label_encoder.inverse_transform(
            model.classes_
        )

        prob_df = pd.DataFrame({
            "Disease": label_encoder.classes_,
            "Probability": prob[0] * 100
        })


        prob_df = prob_df.sort_values(
            by="Probability",
            ascending=True
        )

        fig = px.bar(
            prob_df,
            x="Probability",
            y="Disease",
            orientation='h',
            text="Probability",
            title="AI Disease Prediction Analysis"
        )

        fig.update_traces(
            texttemplate='%{text:.2f}%'
        )

        fig.update_layout(
            template="plotly_dark",
            height=500
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # =============================================
        # DEBUG INFO
        # =============================================
        with st.expander("🔬 Research Debug Information"):

            st.write(
                "Model Classes:",
                model.classes_
            )

            st.write(
                "Encoder Classes:",
                label_encoder.classes_
            )

            st.write(
                "Raw Probabilities:",
                prob[0]
            )

            st.write(
                "Final Prediction:",
                disease
            )

        with st.expander("Model Metadata"):

            st.write(
                "Model Classes:",
            model.classes_
            )

            st.write(
                "Encoder Classes:",
            label_encoder.classes_
            )


    # =====================================================
    # EXPLAINABILITY
    # =====================================================
    with tab2:

        st.subheader(
            "Explainable AI Feature Importance"
        )

        if hasattr(
            model,
            "feature_importances_"
        ):

            importance_df = pd.DataFrame({
                "Feature": input_df.columns,
                "Importance": model.feature_importances_
            })

            importance_df = importance_df.sort_values(
                by="Importance",
                ascending=False
            )

            st.dataframe(
                importance_df.head(15)
            )

            fig2 = px.bar(
                importance_df.head(10),
                x="Importance",
                y="Feature",
                orientation='h',
                title="Top Clinical Features"
            )

            fig2.update_layout(
                template="plotly_dark"
            )

            st.plotly_chart(
                fig2,
                use_container_width=True
            )

        # =============================================
        # CLINICAL REASONING
        # =============================================
        st.subheader(
            "Clinical Reasoning"
        )

        reasons = []

        if gluc > 200:
            reasons.append(
                "High glucose strongly indicates diabetes"
            )

        if temp > 38:
            reasons.append(
                "Elevated temperature indicates infection"
            )

        if spo2 < 92:
            reasons.append(
                "Low oxygen saturation detected"
            )

        if "rash" in symptom_text:
            reasons.append(
                "Rash pattern supports allergy"
            )

        if override_reason:
            reasons.append(
                f"Clinical Override Applied: {override_reason}"
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

        search_terms = [disease.lower()]

        if "fever" in symptom_text:
            search_terms.append("fever")

        if "cough" in symptom_text:
            search_terms.append("cold")

        query = "|".join(search_terms)

        meds = med_db[
            med_db["Reason"]
            .str.lower()
            .str.contains(query, na=False)
        ]

        if meds.empty:
            meds = med_db.head(5)

        for _, row in meds.head(10).iterrows():

            st.markdown(f"""
            <div class='med-card'>

            <b>{row['Drug_Name']}</b><br>

            <i>{row['Reason']}</i><br><br>

            <small>{row['Description']}</small>

            </div>
            """, unsafe_allow_html=True)

    # =====================================================
    # REPORT DOWNLOAD
    # =====================================================
    report = generate_report(
        name,
        disease,
        round(confidence, 2),
        status,
        severity,
        risk
    )

    st.download_button(
        "📄 Download Clinical Report",
        report,
        file_name=f"{name}_clinical_report.html",
        mime="text/html"
    )
