
import streamlit as st
import pandas as pd
import joblib

# -----------------------------------
# PAGE CONFIGURATION
# -----------------------------------

st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="🌿",
    layout="wide"
)

# -----------------------------------
# CUSTOM GREEN AND WHITE DESIGN
# -----------------------------------

st.markdown("""
<style>

.stApp {
    background-color: #f4faf5;
}

[data-testid="stSidebar"] {
    background-color: #e3f3e7;
}

.main-title {
    color: #176b3a;
    font-size: 38px;
    font-weight: bold;
    text-align: center;
}

.subtitle {
    color: #557564;
    text-align: center;
    font-size: 17px;
}

.section-title {
    color: #176b3a;
    font-size: 23px;
    font-weight: bold;
}

.result-card {
    background-color: white;
    padding: 25px;
    border-radius: 15px;
    border: 1px solid #d5eadb;
    text-align: center;
}

.footer {
    text-align: center;
    color: #66806e;
    padding: 20px;
    font-size: 14px;
}

.stButton > button {
    background-color: #198754;
    color: white;
    border-radius: 10px;
    border: none;
    height: 48px;
    font-size: 17px;
    font-weight: bold;
    width: 100%;
}

.stButton > button:hover {
    background-color: #126b40;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------------
# LOAD RANDOM FOREST MODEL
# -----------------------------------

model = joblib.load("model/diabetes_model.pkl")

# -----------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------

st.sidebar.title("🌿 Diabetes Care")
st.sidebar.write("Machine Learning Prediction System")

page = st.sidebar.radio(
    "Navigation",
    ["Home", "About Project", "Disclaimer"]
)

# -----------------------------------
# HOME PAGE
# -----------------------------------

if page == "Home":

    st.markdown(
        '<div class="main-title">🌿 Diabetes Risk Prediction</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Machine Learning Based Prediction System</div>',
        unsafe_allow_html=True
    )

    st.write("")

    st.success(
        "Welcome! Enter the patient information and symptoms below."
    )

    st.markdown(
        '<div class="section-title">👤 Patient Information</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=25
        )

    with col2:
        gender = st.selectbox(
            "Gender",
            ["Female", "Male"]
        )

    st.divider()

    st.markdown(
        '<div class="section-title">🩺 Symptoms Information</div>',
        unsafe_allow_html=True
    )

    questions = [
        "Polyuria",
        "Polydipsia",
        "sudden weight loss",
        "weakness",
        "Polyphagia",
        "Genital thrush",
        "visual blurring",
        "Itching",
        "Irritability",
        "delayed healing",
        "partial paresis",
        "muscle stiffness",
        "Alopecia",
        "Obesity"
    ]

    inputs = {
        "Age": age,
        "Gender": 1 if gender == "Male" else 0
    }

    col1, col2 = st.columns(2)

    for i, question in enumerate(questions):
        with col1 if i % 2 == 0 else col2:
            answer = st.selectbox(
                question,
                ["No", "Yes"],
                key=question
            )

        inputs[question] = 1 if answer == "Yes" else 0

    st.write("")

    # -----------------------------------
    # PREDICTION
    # -----------------------------------

    if st.button("🔍 Predict Diabetes Risk"):

        input_df = pd.DataFrame([inputs])

        input_df = input_df[model.feature_names_in_]

        prediction = model.predict(input_df)[0]

        probabilities = model.predict_proba(input_df)[0]

        class_probabilities = dict(
            zip(model.classes_, probabilities)
        )

        negative = class_probabilities[0] * 100
        positive = class_probabilities[1] * 100

        st.divider()

        st.markdown(
            '<div class="section-title">📊 Prediction Result</div>',
            unsafe_allow_html=True
        )

        if prediction == 1:
            st.warning("Model Prediction: Positive")
        else:
            st.success("Model Prediction: Negative")

        result_col1, result_col2 = st.columns(2)

        with result_col1:
            st.metric(
                "Negative Probability",
                f"{negative:.2f}%"
            )
            st.progress(int(round(negative)))

        with result_col2:
            st.metric(
                "Positive Probability",
                f"{positive:.2f}%"
            )
            st.progress(int(round(positive)))

        with st.expander("View Input Details"):
            st.dataframe(input_df)

        st.caption(
            "Probabilities represent the model's estimates, "
            "not an individual's confirmed medical risk."
        )

# -----------------------------------
# ABOUT PROJECT PAGE
# -----------------------------------

elif page == "About Project":

    st.title("📘 About the Project")

    st.write("""
    **Diabetes Risk Prediction Using Machine Learning**

    This project uses a Random Forest classification model
    to classify questionnaire responses into Positive or
    Negative categories.

    **Technology Stack:**
    - Python
    - Pandas
    - Scikit-learn
    - Random Forest Algorithm
    - Streamlit

    **Dataset:**
    UCI Early Stage Diabetes Risk Prediction dataset.

    **Main Modules:**
    1. Dataset collection
    2. Data preprocessing
    3. Model training
    4. Model evaluation
    5. Prediction interface
    """)

# -----------------------------------
# DISCLAIMER PAGE
# -----------------------------------

elif page == "Disclaimer":

    st.title("⚠️ Medical Disclaimer")

    st.write("""
    This application is developed for educational and
    academic project purposes.

    The prediction is based on questionnaire data and
    machine learning. It is not a medical diagnosis.

    The result should not replace professional medical
    consultation, laboratory testing, or clinical evaluation.

    Please consult a qualified healthcare professional
    for medical advice.
    """)

# -----------------------------------
# FOOTER
# -----------------------------------

st.markdown("""
<div class="footer">
    🌿 Diabetes Risk Prediction System |
    Developed using Python and Machine Learning
</div>
""", unsafe_allow_html=True)
