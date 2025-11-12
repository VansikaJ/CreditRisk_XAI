import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
from lime.lime_tabular import LimeTabularExplainer
from matplotlib import pyplot as plt

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="💳 Credit Approval Simulator",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    /* Main background and text */
    body, .stApp {
        background-color: #f7f9fc;
        color: #1e293b;
        font-family: 'Segoe UI', sans-serif;
    }

    /* Titles and headers */
    h1, h2, h3 {
        color: #0f172a;
        font-weight: 700;
    }

    /* Metric & gauge card look */
    .stPlotlyChart {
        border-radius: 15px;
        background-color: #ffffff;
        padding: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }

    /* Input area styling */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Tabs */
    div[data-baseweb="tab-list"] {
        justify-content: center;
    }

    /* Success message */
    .stSuccess {
        background-color: #e6fffa;
        color: #065f46;
        border-left: 4px solid #10b981;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Load model, scaler, encoders, and training data
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
st.success("✅ Model and data successfully loaded!")

# ---------------------------
# App Header
# ---------------------------
st.title("💳 What-If Credit Approval Simulator")
st.markdown("#### Simulate and visualize your **credit approval probability** instantly using interactive sliders and explainability tools (LIME).")

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
    annual_income = st.number_input("Annual Income (₹)", 0, 2_000_000, 500_000, step=10_000)
    type_income_str = st.selectbox("Type of Income", le_income.classes_)
with col3:
    education_str = st.selectbox("Education Level", le_education.classes_)
    marital_str = st.selectbox("Marital Status", le_marital.classes_)

# Encode categorical variables
type_income_code = le_income.transform([type_income_str])[0]
education_code = le_education.transform([education_str])[0]
marital_code = le_marital.transform([marital_str])[0]

# Build input DataFrame
X_input = pd.DataFrame([{
    "Employed_Years": employed_years,
    "Annual_income": annual_income,
    "Age": age,
    "Type_Income": type_income_code,
    "Education": education_code,
    "Marital_status": marital_code
}])

X_input_scaled = pd.DataFrame(scaler.transform(X_input), columns=X_input.columns)

# ---------------------------
# Tabs for Prediction & LIME
# ---------------------------
tab1, tab2 = st.tabs(["🏦 Credit Approval Simulation", "📊 LIME Explainability"])

# ---- Simulator Tab ----
with tab1:
    st.markdown("### 💡 Prediction Dashboard")
    try:
        pred_prob = model.predict_proba(X_input_scaled)[0][1] * 100

        # Gauge Chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=pred_prob,
            title={'text': "Approval Probability (%)", 'font': {'size': 22}},
            delta={'reference': 50, 'increasing': {'color': "#22c55e"}, 'decreasing': {'color': "#ef4444"}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#3b82f6", 'thickness': 0.25},
                'steps': [
                    {'range': [0, 40], 'color': '#fee2e2'},
                    {'range': [40, 70], 'color': '#fef3c7'},
                    {'range': [70, 100], 'color': '#dcfce7'}
                ],
            }
        ))
        fig.update_layout(height=400, margin=dict(t=20, b=0, l=0, r=0))
        st.plotly_chart(fig, use_container_width=True)

        # Recommendation text
        if pred_prob > 70:
            st.success("🎉 Great! You have a **high probability** of credit approval.")
        elif pred_prob > 40:
            st.warning("⚠️ Moderate chance of approval. Consider improving income or reducing liabilities.")
        else:
            st.error("❌ Low approval probability. Try adjusting your financial parameters.")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ---- LIME Tab ----
with tab2:
    st.markdown("### 🧠 LIME Feature Importance")
    st.caption("Explore how each feature contributes to your approval probability.")

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

        fig = exp.as_pyplot_figure()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error computing LIME: {e}")
