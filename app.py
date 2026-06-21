import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
from database import create_tables, register_user, login_user, save_prediction
create_tables()

st.set_page_config(page_title="AI Loan Approval System", layout="centered")
st.markdown("""
<style>
body {
    background-color: #0e1117;
}

.stApp {
    background-color: #0e1117;
    color: white;
}

/* Remove green headers */
h1, h2, h3 {
    color: white !important;
}

/* Input fields */
input, .stNumberInput, .stSelectbox {
    background-color: #1c1f26 !important;
    color: white !important;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(135deg, #00c6ff, #0072ff);
    color: white;
    border-radius: 10px;
    border: none;
}
.stButton>button:hover {
    background: linear-gradient(135deg, #0072ff, #00c6ff);
    transform: scale(1.03);
}
</style>
""", unsafe_allow_html=True)

# ---------- SESSION STATE ----------
if "page" not in st.session_state:
    st.session_state.page = "landing"

if "user" not in st.session_state:
    st.session_state.user = None

# ---------- PAGE 1 (HOME) ----------
if st.session_state.page == "landing":

    st.markdown("""
    <h1 style='text-align: center; color: white;'>💳 Smart Loan Approval System</h1>
    <p style='text-align: center; font-size:18px; color: #cccccc;'>

    Get instant insights into your loan approval chances using AI.<br><br>

    ✔ Understand your approval probability<br>
    ✔ Identify financial risks instantly<br>
    ✔ Make smarter financial decisions<br><br>

    No paperwork. No waiting. Just results.

    </p>
    """, unsafe_allow_html=True)

    st.markdown("""
    <style>
    .start-btn {
        display: flex;
        justify-content: center;
        margin-top: 30px;
    }
    .start-btn button {
        background: linear-gradient(135deg, #ff7a18, #ff4e50);
        color: white;
        padding: 12px 28px;
        border-radius: 10px;
        border: none;
        font-size: 16px;
        font-weight: bold;
        transition: 0.3s;
    }
    .start-btn button:hover {
        transform: scale(1.05);
        box-shadow: 0px 0px 15px rgba(255, 78, 80, 0.6);
    }
    </style>
    """, unsafe_allow_html=True)

    col = st.columns(3)
    with col[1]:
        if st.button("🚀 Get Started"):
            st.session_state.page = "auth"

# ---------- PAGE 2 (auth) ----------
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
                st.error("Username already taken ⚠️ Try a different one")
            else:
                st.error("Something went wrong")

    else:
        if st.button("Login"):
            if login_user(username, password):
                st.session_state.user = username
                st.session_state.page = "form"
            else:
                st.error("Invalid credentials")
# ---------- PAGE 3 (form) ----------
elif st.session_state.page == "form":

    st.markdown("<h2 style='text-align:center;'>Loan Application</h2>", unsafe_allow_html=True)

    # ---------- PERSONAL ----------
    st.subheader("👤 Personal Details")
    no_of_dependents = st.number_input("Dependents", 0, 10, 1)

    # ---------- EDUCATION ----------
    st.subheader("🎓 Education")
    education = st.selectbox("Education Level", ["Graduate", "Not Graduate"])

    # ---------- EMPLOYMENT ----------
    st.subheader("💼 Employment")
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])

    # ---------- FINANCIAL ----------
    st.subheader("💰 Financial Details")
    income_annum = st.number_input("Annual Income", 0)
    loan_amount = st.number_input("Loan Amount Requested", 0)
    loan_term = st.number_input("Loan Term (months)", 0)

    # ---------- CREDIT ----------
    st.subheader("📊 Credit Profile")
    cibil_score = st.number_input("CIBIL Score", 300, 900)

    # ---------- ASSETS ----------
    st.subheader("🏦 Assets")
    residential_assets_value = st.number_input("Residential Assets Value", 0)
    commercial_assets_value = st.number_input("Commercial Assets Value", 0)
    luxury_assets_value = st.number_input("Luxury Assets Value", 0)
    bank_asset_value = st.number_input("Bank Balance", 0)

    # Encode
    education = 1 if education == "Graduate" else 0
    self_employed = 1 if self_employed == "Yes" else 0

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # ---------- PREDICT BUTTON ----------
    with col1:
        if st.button("🔮 Predict"):

            data = {
                "no_of_dependents": no_of_dependents,
                "education": education,
                "self_employed": self_employed,
                "income_annum": income_annum,
                "loan_amount": loan_amount,
                "loan_term": loan_term,
                "cibil_score": cibil_score,
                "residential_assets_value": residential_assets_value,
                "commercial_assets_value": commercial_assets_value,
                "luxury_assets_value": luxury_assets_value,
                "bank_asset_value": bank_asset_value
            }

            with st.spinner("⏳ Predicting..."):
                try:
                    response = requests.post(
                        "http://127.0.0.1:8000/predict",
                        json=data,
                        timeout=30
                    )

                    if response.status_code == 200:
                        result = response.json()

                        if "error" in result:
                            st.error(f"Prediction failed: {result['error']}")
                            st.stop()

                        st.markdown("----")

                        # ===== RESULT =====
                        if result["prediction"] == 1:
                            st.success("✔ Loan Approved")
                        else:
                            st.error("✖ Loan Rejected")

                        st.write(f"📊 Approval Probability: {result['probability']}%")
                        st.write(f"⚠ Risk Level: {result['risk_level']}")

                        # ===== KEY FACTORS =====
                        st.write("### Key Factors:")
                        for r in result["reasons"]:
                            st.write("•", r)

                        # ===== SHAP TEXT =====
                        if "shap_values" in result and result["shap_values"]:
                            st.subheader("Why this decision?")

                            shap_df = pd.DataFrame(
                                result["shap_values"].items(),
                                columns=["feature", "impact"]
                            )
                            shap_df["abs_impact"] = shap_df["impact"].abs()
                            shap_df = shap_df.sort_values("abs_impact", ascending=True).tail(8)

                            colors = [
                                "#2ecc71" if value > 0 else "#e74c3c"
                                for value in shap_df["impact"]
                            ]

                            fig, ax = plt.subplots(figsize=(8, 4.5))
                            ax.barh(shap_df["feature"], shap_df["impact"], color=colors)
                            ax.axvline(0, color="white", linewidth=1)
                            ax.set_facecolor("#0e1117")
                            fig.patch.set_facecolor("#0e1117")
                            ax.tick_params(colors="white")
                            ax.set_xlabel("Impact on Approval", color="white")
                            ax.set_ylabel("Feature", color="white")
                            ax.set_title("Top Factors Behind This Prediction", color="white")

                            for spine in ax.spines.values():
                                spine.set_color("#888888")

                            st.pyplot(fig, clear_figure=True)
                            st.caption("Green bars increase approval chance, red bars reduce it.")

                        # ===== SAVE TO DATABASE =====
                        if st.session_state.user:
                            save_prediction(
                                st.session_state.user,
                                income_annum,
                                loan_amount,
                                result["prediction"],
                                result["probability"]
                            )

                    else:
                        try:
                            error_detail = response.json().get("detail", response.text)
                        except Exception:
                            error_detail = response.text
                        st.error(f"API Error: {error_detail}")

                except Exception as e:
                    st.error(f"Connection Error: {e}")

    # ---------- BACK BUTTON ----------
    with col2:
        if st.button("Logout"):
            st.session_state.user = None
            st.session_state.page = "landing"
