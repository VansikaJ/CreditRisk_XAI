import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
import warnings

# ---------------------------
# Clean warnings
# ---------------------------
warnings.filterwarnings("ignore")

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="💳 Credit Approval Simulator",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------
# Custom CSS
# ---------------------------
st.markdown("""
<style>
body, .stApp {
    background-color: #f7f9fc;
    color: #1e293b;
    font-family: 'Segoe UI', sans-serif;
}

h1, h2, h3 {
    color: #0f172a;
    font-weight: 700;
}

.stPlotlyChart {
    border-radius: 15px;
    background-color: #ffffff;
    padding: 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

div[data-baseweb="tab-list"] {
    justify-content: center;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Load model objects
# ---------------------------
@st.cache_resource
def load_objects():
    scaler = joblib.load("scaler.pkl")
    le_income = joblib.load("le_income.pkl")
    le_education = joblib.load("le_education.pkl")
    le_marital = joblib.load("le_marital.pkl")
    model = joblib.load("credit_model.pkl")
    X_train = joblib.load("X_train.pkl")
    return scaler, le_income, le_education, le_marital, model, X_train

scaler, le_income, le_education, le_marital, model, X_train = load_objects()
st.success("✅ Model & data successfully loaded!")

# ---------------------------
# Header
# ---------------------------
st.title("💳 What-If Credit Approval Simulator")
st.markdown(
    "#### Simulate and visualize your **credit approval probability** instantly using interactive sliders and explainability tools (LIME)."
)
st.divider()

# ---------------------------
# Input Section
# ---------------------------
st.subheader("🧩 Input Your Financial Details")

col1, col2, col3 = st.columns(3)

with col1:
    employed_years = st.slider("Years Employed", 0, 50, 5)
    age = st.slider("Age", 18, 70, 30)

with col2:
    annual_income = st.number_input(
        "Annual Income (₹)", 0, 2_000_000, 500_000, step=10_000
    )
    type_income_str = st.selectbox(
        "Type of Income", le_income.classes_
    )

with col3:
    education_str = st.selectbox(
        "Education Level", le_education.classes_
    )
    marital_str = st.selectbox(
        "Marital Status", le_marital.classes_
    )

# Encode inputs
X_input = pd.DataFrame([{
    "Employed_Years": employed_years,
    "Annual_income": annual_income,
    "Age": age,
    "Type_Income": le_income.transform([type_income_str])[0],
    "Education": le_education.transform([education_str])[0],
    "Marital_status": le_marital.transform([marital_str])[0]
}])

X_input_scaled = pd.DataFrame(
    scaler.transform(X_input),
    columns=X_input.columns
)

# ============================================
# CREDIT COACHING ENGINE
# ============================================

class CreditCoach:
    def __init__(self, user_data):
        self.data = user_data
        self.suggestions = []

    def check_income(self):
        if self.data["Annual_income"] < 300000:
            self.suggestions.append(
                "Increase income or show additional sources to strengthen eligibility."
            )

    def check_employment(self):
        if self.data["Employed_Years"] < 2:
            self.suggestions.append(
                "Maintain stable employment for at least 6–12 more months."
            )

    def check_age(self):
        if self.data["Age"] < 21:
            self.suggestions.append(
                "Age is slightly low — maintaining a longer work history can help."
            )

    def check_education(self):
        if self.data["Education"] < 2:
            self.suggestions.append(
                "Higher education often increases approval chance — consider certification or skill programs."
            )

    def check_marital(self):
        if self.data["Marital_status"] == 1:
            self.suggestions.append(
                "Adding a co-applicant may strengthen your overall profile."
            )

    def generate(self):
        self.check_income()
        self.check_employment()
        self.check_age()
        self.check_education()
        self.check_marital()

        if not self.suggestions:
            self.suggestions.append(
                "Your profile is strong — small improvements can maximize approval chances."
            )

        return self.suggestions

# ---------------------------
# Tabs
# ---------------------------
tab1, tab2 = st.tabs(["🏦 Credit Approval Simulation", "📊 LIME Explainability"])

# ============================================
# TAB 1 – Prediction
# ============================================
with tab1:
    st.markdown("### 💡 Prediction Dashboard")

    try:
        pred_prob = model.predict_proba(X_input_scaled)[0][1] * 100

        # Gauge Chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=pred_prob,
            title={'text': "Approval Probability (%)", 'font': {'size': 22}},
            delta={'reference': 50},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#3b82f6"},
                'steps': [
                    {'range': [0, 40], 'color': '#fee2e2'},
                    {'range': [40, 70], 'color': '#fef3c7'},
                    {'range': [70, 100], 'color': '#dcfce7'}
                ],
            }
        ))
        fig.update_layout(height=350, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

        # Recommendation Messages
        if pred_prob > 70:
            st.success("🎉 High probability of credit approval!")
        elif pred_prob > 40:
            st.warning("⚠️ Moderate chance — consider improving income or liabilities.")
        else:
            st.error("❌ Low probability — try adjusting your financial parameters.")

        # Credit Health Block
        st.markdown("## Credit Health Score")
        credit_score = pred_prob

        if credit_score < 40:
            level, color, emoji, tip = "Bronze", "#CD7F32", "🥉", "Focus on stabilizing income & reducing liabilities."
        elif credit_score < 70:
            level, color, emoji, tip = "Silver", "#C0C0C0", "🥈", "Continue improving consistency in financial habits."
        elif credit_score < 90:
            level, color, emoji, tip = "Gold", "#FFD700", "🥇", "Excellent behavior — optimize EMI-to-income ratio."
        else:
            level, color, emoji, tip = "Platinum", "#e5e4e2", "💎", "You qualify for premium financial products."

        credit_html = f"""
        <div style="
            background-color:white;
            border-radius:15px;
            padding:20px;
            text-align:center;
            box-shadow:0 2px 8px rgba(0,0,0,0.08);
        ">
            <h2>{emoji} <b>{level} Level</b></h2>
            <p style="font-size:18px;">
                Your Credit Health Score:
                <span style="font-weight:700; color:{color};">{credit_score:.1f}</span>/100
            </p>
        </div>
        """
        st.markdown(credit_html, unsafe_allow_html=True)
        st.info(f"💡 Financial Tip: {tip}")

        # ============================================
        # PERSONALIZED CREDIT COACHING
        # ============================================
        if pred_prob < 60:
            st.markdown("## 🧠 Personalized Credit Coaching")
            st.write("Here’s how you can improve your credit profile:")

            coach = CreditCoach({
                "Employed_Years": employed_years,
                "Annual_income": annual_income,
                "Age": age,
                "Type_Income": le_income.transform([type_income_str])[0],
                "Education": le_education.transform([education_str])[0],
                "Marital_status": le_marital.transform([marital_str])[0]
            })

            suggestions = coach.generate()

            coaching_html = """
            <div style="
                background-color:white; padding:20px;
                border-radius:15px; box-shadow:0 2px 10px rgba(0,0,0,0.1);
            ">
                <h3 style="color:#0f172a;">📈 Your Credit Improvement Plan</h3>
                <ul style="font-size:16px; color:#334155;">
            """
            for s in suggestions:
                coaching_html += f"<li>{s}</li>"
            coaching_html += "</ul></div>"

            st.markdown(coaching_html, unsafe_allow_html=True)
        else:
            st.success("🎉 Excellent profile! No improvement steps required.")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ============================================
# TAB 2 – LIME Explainability
# ============================================
with tab2:
    st.markdown("### 🧠 LIME Feature Importance")

    try:
        explainer = LimeTabularExplainer(
            training_data=X_train.values,
            feature_names=X_train.columns.tolist(),
            class_names=['Rejected', 'Approved'],
            mode='classification'
        )

        exp = explainer.explain_instance(
            data_row=X_input_scaled.iloc[0].values,
            predict_fn=model.predict_proba,
            num_features=len(X_train.columns)
        )

        fig2 = exp.as_pyplot_figure()
        st.pyplot(fig2)

    except Exception as e:
        st.error(f"Error computing LIME: {e}")
