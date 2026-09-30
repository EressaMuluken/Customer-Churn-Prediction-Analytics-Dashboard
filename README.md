# ChurnPredict

An end-to-end **customer churn prediction and monitoring dashboard** built with Python, Scikit-learn, XGBoost, SHAP, PSI, and Streamlit.

## Dashboard

### Overview

![Overview](images/overview.png)

### Model Performance

![Model Performance](images/model_performance.png)

### Customer Lookup

![Customer Lookup](images/customer_lookup.png)

## ML Pipeline

**Data Cleaning → EDA → Preprocessing → Model Training → Hyperparameter Tuning → Threshold Optimization → Explainability → Monitoring**

- Multiple classification models: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost
- Model evaluation: Precision, Recall, F1, ROC-AUC, PR-AUC, Confusion Matrix
- **SHAP** for global and individual prediction explanations
- **PSI** for feature distribution drift monitoring
- Customer-level churn risk and CLTV analysis
- Interactive Streamlit dashboard

## Tech Stack

`Python` · `Pandas` · `Scikit-learn` · `XGBoost` · `SHAP` · `Matplotlib` · `Streamlit` · `Joblib`

## Run Locally

```bash
git clone <https://github.com/EressaMuluken/Customer-Churn-Prediction-Analytics-Dashboard>
cd <Customer-Churn-Prediction-Analytics-Dashboard>
pip install -r requirements.txt
streamlit run streamlit/app.py
```

## Project Structure

```text
├── .streamlit/
├── data/
├── evaluation/
├── images/
├── lib/
├── metadata/
├── metrics/
├── models/
├── notebooks/
├── app.py
├── customer.py
├── model_perfromance.py
├── overview.py
├── requirements.txt
└── README.md
```


