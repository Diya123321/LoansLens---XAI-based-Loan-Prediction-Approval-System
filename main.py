import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib

# Load model artifacts
model = joblib.load("loan_model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

# Create SHAP explainer once
explainer = shap.Explainer(model)

app = FastAPI()


class LoanData(BaseModel):
    no_of_dependents: int
    education: int
    self_employed: int
    income_annum: float
    loan_amount: float
    loan_term: float
    cibil_score: float
    residential_assets_value: float
    commercial_assets_value: float
    luxury_assets_value: float
    bank_asset_value: float


def assess_application(data: LoanData, approval_probability: float):
    income = max(float(data.income_annum), 1.0)
    loan_amount = float(data.loan_amount)
    total_assets = (
        float(data.residential_assets_value)
        + float(data.commercial_assets_value)
        + float(data.luxury_assets_value)
        + float(data.bank_asset_value)
    )
    loan_to_income = loan_amount / income
    bank_to_loan = float(data.bank_asset_value) / max(loan_amount, 1.0)
    asset_coverage = total_assets / max(loan_amount, 1.0)

    risk_reasons = []
    strengths = []
    penalty = 0.0

    if data.cibil_score < 600:
        penalty += 0.22
        risk_reasons.append("CIBIL score is below the safer approval range")
    elif data.cibil_score < 700:
        penalty += 0.08
        risk_reasons.append("CIBIL score is only moderately strong")
    elif data.cibil_score >= 750:
        strengths.append("Strong CIBIL score")

    if loan_to_income > 0.8:
        penalty += 0.24
        risk_reasons.append("Loan amount is too high compared to annual income")
    elif loan_to_income > 0.5:
        penalty += 0.10
        risk_reasons.append("Loan amount is on the higher side for the income level")
    elif loan_to_income <= 0.35:
        strengths.append("Loan amount is comfortably within annual income")

    if data.bank_asset_value < 20000:
        penalty += 0.14
        risk_reasons.append("Bank balance is too low for this application")
    elif data.bank_asset_value < 50000:
        penalty += 0.06
        risk_reasons.append("Bank balance is on the lower side")
    elif bank_to_loan >= 0.25:
        strengths.append("Bank balance provides good short-term coverage")

    if asset_coverage < 0.5:
        penalty += 0.10
        risk_reasons.append("Total assets provide limited support for the requested loan")
    elif asset_coverage >= 1.5:
        strengths.append("Assets provide strong collateral coverage")

    if data.no_of_dependents >= 4 and income < 600000:
        penalty += 0.08
        risk_reasons.append("High dependents count increases repayment pressure")

    if float(data.loan_term) > 36:
        penalty += 0.08
        risk_reasons.append("Long loan term increases repayment uncertainty")
    elif float(data.loan_term) <= 18:
        strengths.append("Shorter loan term lowers lender risk")

    adjusted_probability = max(0.0, min(1.0, approval_probability - penalty))
    is_approved = adjusted_probability >= 0.58

    if adjusted_probability >= 0.78:
        risk = "Low Risk"
    elif adjusted_probability >= 0.58:
        risk = "Medium Risk"
    else:
        risk = "High Risk"

    if is_approved:
        reasons = strengths[:]
        if risk_reasons:
            reasons.extend(risk_reasons[:1])
        if not reasons:
            reasons.append("Profile is within the approval range after risk adjustment")
    else:
        reasons = risk_reasons[:]
        if not reasons:
            reasons.append("Overall profile is not strong enough for approval")

    return is_approved, risk, reasons, adjusted_probability


@app.get("/")
def home():
    return {"message": "Loan Prediction API is running"}


@app.post("/predict")
def predict(data: LoanData):
    try:
        approval_label = 0

        input_dict = {
            "no_of_dependents": data.no_of_dependents,
            "education": data.education,
            "self_employed": data.self_employed,
            "income_annum": data.income_annum,
            "loan_amount": data.loan_amount,
            "loan_term": data.loan_term,
            "cibil_score": data.cibil_score,
            "residential_assets_value": data.residential_assets_value,
            "commercial_assets_value": data.commercial_assets_value,
            "luxury_assets_value": data.luxury_assets_value,
            "bank_asset_value": data.bank_asset_value,
        }

        input_df = pd.DataFrame([input_dict])[features]
        input_scaled = scaler.transform(input_df)

        raw_prediction = int(model.predict(input_scaled)[0])
        probability_vector = model.predict_proba(input_scaled)[0]
        approval_probability = float(probability_vector[approval_label])
        model_predicts_approval = raw_prediction == approval_label

        # The trained model uses class 1 for rejection, so invert SHAP values
        # to explain approval chance consistently in the UI.
        shap_values = explainer(input_scaled)
        shap_dict = {
            column: float(-value)
            for column, value in zip(input_df.columns, shap_values.values[0])
        }

        is_approved, risk, reasons, final_probability = assess_application(data, approval_probability)

        if not model_predicts_approval and is_approved:
            is_approved = False
            final_probability = min(final_probability, 0.49)
            risk = "High Risk"
            if "Model confidence is not strong enough for approval" not in reasons:
                reasons.append("Model confidence is not strong enough for approval")

        log_df = pd.DataFrame([{
            **input_dict,
            "prediction": 1 if is_approved else 0,
            "probability": final_probability,
        }])
        log_df.to_csv("logs.csv", mode="a", header=False, index=False)

        return {
            "prediction": 1 if is_approved else 0,
            "result": "Loan Approved" if is_approved else "Loan Rejected",
            "probability": round(final_probability * 100, 2),
            "risk_level": risk,
            "reasons": reasons,
            "shap_values": shap_dict,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
