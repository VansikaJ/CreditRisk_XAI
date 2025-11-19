💳 Credit Approval Simulator with LIME Explainability

Interactive Streamlit Application for Credit Risk Prediction & Transparent Financial Decision-Making

📘 Overview

This project implements an Explainable Credit Approval Simulator built using Streamlit, Scikit-learn, LIME, Plotly, and Joblib.
The app predicts the credit approval probability of an applicant in real time and provides:

🔍 Financial What-If Analysis

🧠 LIME-based explainability

📊 Gauge visualizations and credit score levels

📝 Personalized credit improvement suggestions

The system is designed to be intuitive, visually appealing, and fully interactive for users, credit analysts, and students studying explainable AI.

🚀 Features
🔮 1. Real-Time Credit Approval Prediction

Uses a trained Machine Learning model with scaled and encoded inputs

Instant approval probability with visual gauge

🧠 2. LIME Explainability

Shows the influence of each feature

Displays positive/negative contributions to the prediction

🧩 3. Financial Input Interface

Users can adjust:

Employment years

Age

Income

Education

Marital status

Income type

📈 4. Credit Health Levels

Bronze / Silver / Gold / Platinum categories

Smart tips for better financial decisions

🧮 5. Personalized Credit Coaching Engine

The app evaluates the user’s profile and gives actionable improvement recommendations.

🛠️ Technologies Used
Area	Tool
Frontend / UI	Streamlit
Machine Learning	Scikit-learn, Joblib
Explainability (XAI)	LIME
Visualization	Plotly, Matplotlib
Data Handling	Pandas
📂 App Requirements

Make sure your directory includes the required model and preprocessing files:

scaler.pkl

le_income.pkl

le_education.pkl

le_marital.pkl

credit_model.pkl

X_train.pkl

app.py (or your chosen filename)

▶️ How to Run the Application
1️⃣ Install dependencies
pip install -r requirements.txt

2️⃣ Run Streamlit
streamlit run app.py


Your browser will open the app automatically.

📌 How the Model Works

User inputs → Encoded and scaled

Model predicts probability of approval vs rejection

LIME interprets how each feature contributed

The app generates a Credit Health Score and coaching suggestions

🧠 LIME Explainability

The LIME tab displays:

Feature importance ranking

Contribution of each variable

Visual explanation using Matplotlib

This strengthens transparency and trust in model predictions, supporting AI ethics and responsible credit scoring practices.

📦 Project Highlights

Fully deployable ML-powered Streamlit application

Combines prediction + explanation + financial coaching

Clean UI with custom CSS styling

Enables What-If simulation for financial decision support

📝 Future Enhancements

Deploying via Streamlit Cloud / AWS

Adding SHAP for deeper explainability

Including credit utilization & repayment history

Adding multi-model comparison

Building a user login dashboard for storing sessions

📄 License

This project is licensed under the MIT License.
You may use, modify, and distribute it with proper attribution.

👩‍💻 Author

Vansika Jhawar
