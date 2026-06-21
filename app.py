import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np

from database import create_tables, register_user, login_user, save_prediction

create_tables()

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("loan_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Loan Approval System", layout="centered")

# ---------------- UI STYLE ----------------
st.markdown("""
<style>
body { background-color: #0e1117; }

.stApp {
    background-color: #0e1117;
    color: white;
}

h1, h2, h3 { color: white !important; }

input, .stNumberInput, .stSelectbox {
    background-color: #1c1f26 !important;
    color: white !important;
}

.stButton>button {
    background: linear-gradient(135deg, #00c6ff, #0072ff);
    color: white;
    border-radius: 10px;
    border: none;
}
.stButton>button:hover {
    transform: scale(1.03);
}
</style>
""", unsafe_allow_html=True)

# ---------------- SESSION ----------------
if "page" not in st.session_state:
    st.session_state.page = "landing"

if "user" not in st.session_state:
    st.session_state.user = None

# ---------------- LANDING PAGE ----------------
if st.session_state.page == "landing":

    st.markdown("""
    <h1 style='text-align:center;'>💳 Smart Loan Approval System</h1>
    <p style='text-align:center;color:#ccc;'>
    AI-based loan prediction with explainability.
    </p>
    """, unsafe_allow_html=True)

    if st.button("🚀 Get Started"):
        st.session_state.page = "auth"

# ---------------- AUTH PAGE ----------------
elif st.session_state.page == "auth":

    st.title("🔐 Login / Register")

    choice = st.radio("Choose", ["Login", "Register"])

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if choice == "Register":
        if st.button("Register"):
            result = register_user(username, password)
            if result == True:
                st.success("Registered successfully!")
            elif result == "exists":
                st.error("User already exists")
            else:
                st.error("Error")

    else:
        if st.button("Login"):
            if login_user(username, password):
                st.session_state.user = username
                st.session_state.page = "form"
            else:
                st.error("Invalid credentials")

# ---------------- FORM PAGE ----------------
elif st.session_state.page == "form":

    st.title("🏦 Loan Application")

    no_of_dependents = st.number_input("Dependents", 0, 10, 1)
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])

    income_annum = st.number_input("Annual Income", 0)
    loan_amount = st.number_input("Loan Amount", 0)
    loan_term = st.number_input("Loan Term (months)", 0)
    cibil_score = st.number_input("CIBIL Score", 300, 900)

    residential_assets_value = st.number_input("Residential Assets", 0)
    commercial_assets_value = st.number_input("Commercial Assets", 0)
    luxury_assets_value = st.number_input("Luxury Assets", 0)
    bank_asset_value = st.number_input("Bank Balance", 0)

    education = 1 if education == "Graduate" else 0
    self_employed = 1 if self_employed == "Yes" else 0

    if st.button("🔮 Predict Loan"):

        # ---------------- PREPARE INPUT ----------------
        input_data = np.array([[
            no_of_dependents,
            education,
            self_employed,
            income_annum,
            loan_amount,
            loan_term,
            cibil_score,
            residential_assets_value,
            commercial_assets_value,
            luxury_assets_value,
            bank_asset_value
        ]])

        input_data = scaler.transform(input_data)

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1] * 100

        st.markdown("---")

        # ---------------- RESULT ----------------
        if prediction == 1:
            st.success("✔ Loan Approved")
        else:
            st.error("✖ Loan Rejected")

        st.write(f"📊 Approval Probability: {probability:.2f}%")

        # ---------------- SAVE ----------------
        if st.session_state.user:
            save_prediction(
                st.session_state.user,
                income_annum,
                loan_amount,
                prediction,
                probability
            )

    # ---------------- LOGOUT ----------------
    if st.button("Logout"):
        st.session_state.user = None
        st.session_state.page = "landing"
