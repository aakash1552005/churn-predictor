import streamlit as st
import joblib
import numpy as np
import os
from xgboost import XGBClassifier

@st.cache_resource
def load_model():
    model = joblib.load("xgb_model.pkl")
    feature_names = joblib.load("feature_names.pkl")
    return model, feature_names

model, feature_names = load_model()

st.title("Customer Churn Predictor")
st.write("Predict whether a customer will leave or stay!")
st.write("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Customer Details")
    gender = st.selectbox("Gender", ["Female", "Male"])
    SeniorCitizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    Partner = st.selectbox("Has Partner", ["No", "Yes"])
    Dependents = st.selectbox("Has Dependents", ["No", "Yes"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    PhoneService = st.selectbox("Phone Service", ["No", "Yes"])
    MultipleLines = st.selectbox(
        "Multiple Lines",
        ["No", "Yes", "No phone service"]
    )
    InternetService = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )
    OnlineSecurity = st.selectbox(
        "Online Security",
        ["No", "Yes", "No internet service"]
    )
    OnlineBackup = st.selectbox(
        "Online Backup",
        ["No", "Yes", "No internet service"]
    )

with col2:
    st.subheader("Service Details")
    DeviceProtection = st.selectbox(
        "Device Protection",
        ["No", "Yes", "No internet service"]
    )
    TechSupport = st.selectbox(
        "Tech Support",
        ["No", "Yes", "No internet service"]
    )
    StreamingTV = st.selectbox(
        "Streaming TV",
        ["No", "Yes", "No internet service"]
    )
    StreamingMovies = st.selectbox(
        "Streaming Movies",
        ["No", "Yes", "No internet service"]
    )
    Contract = st.selectbox(
        "Contract Type",
        ["Month-to-month", "One year", "Two year"]
    )
    PaperlessBilling = st.selectbox(
        "Paperless Billing",
        ["No", "Yes"]
    )
    PaymentMethod = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )
    MonthlyCharges = st.slider(
        "Monthly Charges ($)",
        18.0, 120.0, 65.0
    )
    TotalCharges = st.slider(
        "Total Charges ($)",
        18.0, 8700.0, 1500.0
    )

st.write("---")

if st.button("Predict Churn", use_container_width=True):

    # Encode inputs exactly like training data
    def encode(val, options):
        return options.index(val)

    input_data = np.array([[
        encode(gender, ["Female", "Male"]),
        1 if SeniorCitizen == "Yes" else 0,
        encode(Partner, ["No", "Yes"]),
        encode(Dependents, ["No", "Yes"]),
        tenure,
        encode(PhoneService, ["No", "Yes"]),
        encode(MultipleLines,
            ["No", "No phone service", "Yes"]),
        encode(InternetService,
            ["DSL", "Fiber optic", "No"]),
        encode(OnlineSecurity,
            ["No", "No internet service", "Yes"]),
        encode(OnlineBackup,
            ["No", "No internet service", "Yes"]),
        encode(DeviceProtection,
            ["No", "No internet service", "Yes"]),
        encode(TechSupport,
            ["No", "No internet service", "Yes"]),
        encode(StreamingTV,
            ["No", "No internet service", "Yes"]),
        encode(StreamingMovies,
            ["No", "No internet service", "Yes"]),
        encode(Contract,
            ["Month-to-month", "One year", "Two year"]),
        encode(PaperlessBilling, ["No", "Yes"]),
        encode(PaymentMethod, [
            "Bank transfer (automatic)",
            "Credit card (automatic)",
            "Electronic check",
            "Mailed check"
        ]),
        MonthlyCharges,
        TotalCharges
    ]])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    st.write("---")

    if prediction == 1:
        st.error(
            f"This customer is LIKELY TO CHURN! "
            f"Risk: {probability[1]*100:.1f}%"
        )
        st.write("### Retention Suggestions:")
        if Contract == "Month-to-month":
            st.write("Offer discount for switching to annual contract")
        if MonthlyCharges > 70:
            st.write("Offer loyalty discount on monthly charges")
        if tenure < 12:
            st.write("Assign dedicated support for first year customers")
        if InternetService == "Fiber optic":
            st.write("Check Fiber Optic service quality for this customer")
    else:
        st.success(
            f"This customer is LIKELY TO STAY! "
            f"Confidence: {probability[0]*100:.1f}%"
        )

    st.write("---")
    st.subheader("Prediction Breakdown")
    col3, col4 = st.columns(2)
    with col3:
        st.metric("Churn Risk", f"{probability[1]*100:.1f}%")
        st.metric("Stay Probability", f"{probability[0]*100:.1f}%")
    with col4:
        st.metric("Contract Type", Contract)
        st.metric("Monthly Charges", f"${MonthlyCharges}")
