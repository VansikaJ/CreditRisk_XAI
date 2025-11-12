# 💳 Interactive Credit Approval Prediction and Explainability  
**An Explainable AI (XAI) Framework for Transparent Credit Risk Assessment**

![Python](https://img.shields.io/badge/Python-3.10-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red)
![ExplainableAI](https://img.shields.io/badge/Explainability-SHAP%20%7C%20LIME-orange)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

---

## 🧠 Abstract  
This project presents an explainable machine learning framework for **credit risk assessment**, integrating data preprocessing, model training, evaluation, and explainability tools such as **SHAP** and **LIME**.  
The system uses a **Random Forest classifier** to predict credit approval outcomes and provides interpretable explanations for each decision.  
An interactive **Streamlit-based interface** enables users to visualize model predictions, explore *“what-if”* scenarios, and understand feature contributions — enhancing transparency and user trust in financial systems.

---

## 📌 Problem Statement  
Traditional credit scoring systems are often **opaque**, offering applicants little insight into why their applications are approved or rejected.  
This project addresses this issue by developing an **interactive, explainable credit approval system** that:  
- Predicts credit approval probability using machine learning.  
- Explains decisions using **SHAP** and **LIME**.  
- Allows users to test *what-if* scenarios (e.g., changes in income or employment).  
- Promotes accountability, fairness, and transparency in automated credit decisions.

---

## ⚙️ System Architecture  

### 🧩 Modules
1. **Data Input & Preprocessing** – Cleans, encodes, and normalizes input data.  
2. **Model Training & Evaluation** – Implements and compares ML models (Logistic Regression, Decision Tree, Random Forest, SVM, KNN, Gradient Boosting).  
3. **Explainability Module** – Uses **SHAP** for global feature importance and **LIME** for local interpretability.  
4. **Interactive What-If Analysis** – Users modify inputs and instantly observe changes in approval probability.  
5. **Visualization Layer** – Displays feature importance plots, LIME explanations, and prediction outcomes.

---

## 🧮 Methodology  

- **Dataset:** Demographic, employment, and financial features.  
- **Preprocessing:** Missing value imputation, outlier treatment (IQR method), label encoding, and feature scaling.  
- **Feature Selection:** Chi-Square, Z-Test, and Recursive Feature Elimination (RFE).  
- **Model Training:** Six ML algorithms evaluated; **Random Forest** achieved the best performance.  
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score, ROC-AUC, Cross-Validation.  

| Model | Test Accuracy | CV Accuracy | AUC |
|:------|:--------------:|:------------:|:----:|
| Logistic Regression | 0.87 | 0.91 | 0.61 |
| Decision Tree | 0.85 | 0.88 | 0.71 |
| **Random Forest** | **0.90** | **0.93** | **0.74** |
| SVM | 0.87 | 0.91 | 0.71 |
| KNN | 0.90 | 0.89 | 0.70 |
| Gradient Boosting | 0.89 | 0.92 | 0.74 |

---

## 🧠 Explainable AI (XAI)
- **Global Explanations (SHAP):** Identify features most influencing credit approval (Income, Age, Employment Years).  
- **Local Explanations (LIME):** Show *why* a specific prediction was made, helping users identify actionable factors.  
- **Interactive Transparency:** Real-time interpretability through Streamlit’s LIME visualization tab.  

---

## 🌐 App Overview  
**Built with:** Streamlit | Scikit-learn | SHAP | LIME | Plotly | Pandas | NumPy  

**Key Features:**  
- Interactive sliders for user inputs (income, employment years, education, etc.)  
- Real-time prediction and approval probability visualization  
- *What-If* scenario testing  
- LIME plots for interpretability  
- Confidence gauge and feature importance visualization  

---

## 📈 Results & Insights  
✅ **Best Model:** Random Forest (Accuracy = 91%, AUC = 0.74)  
✅ **Explainability:** SHAP and LIME improved user understanding of ML decisions  
✅ **Deployment:** Fully functional Streamlit app for credit officers and applicants  

---

## 💡 Future Enhancements  
- Integrate **real-time financial data** and external credit bureau APIs.  
- Implement **fairness auditing** and **bias detection**.  
- Deploy as a **cloud-based API** using AWS / GCP.  
- Add **deep learning models** or **AutoML** for advanced performance.  
- Develop a **mobile-friendly dashboard** for accessibility.  

---

## ⚖️ License  
This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🚀 Getting Started  

### Installation  
```bash
git clone https://github.com/VansikaJ/CreditRisk_XAI.git
cd CreditRisk_XAI
pip install -r requirements.txt
