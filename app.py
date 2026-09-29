import streamlit as st 
import pandas as pd 
import joblib
import json
@st.cache_data
def get_data(path): 
  return pd.read_excel(path)
for key in st.session_state: 
  st.session_state[key] = st.session_state[key] 

if 'current_model' not in st.session_state:
  st.session_state['current_model'] = {
        'Decision Tree':joblib.load('./models/Decision Tree_pipeline.joblib'), 
        'Gradient Boosting': joblib.load('./models/Gradient Boosting_pipeline.joblib'),
        'Logistic Regression': joblib.load('./models/Logistic Regression_pipeline.joblib'),
        'Random Forest': joblib.load('./models/Random Forest_pipeline.joblib'), 
        'XGBoost': joblib.load('./models/XGBoost_pipeline.joblib')
        } 
if 'data' not in st.session_state:
  st.session_state["data"] = get_data("./data/cleaned_Telco_consumer_churn.xlsx")

st.set_page_config(layout='wide')
if 'selected_model_name' not in st.session_state:
  st.session_state['selected_model_name'] = next(iter(st.session_state['current_model']))
if 'selected_model' not in st.session_state: 
  st.session_state['selected_model'] = st.session_state['current_model'][st.session_state['selected_model_name']][st.session_state['selected_model_name']]
st.markdown("""
<style>
   .stApp {
        background-color: light-dark(#f3f4f6, #1f2937);
    }
    .block-container {
        padding-top: 3rem;
        padding-bottom: 2rem;
    }
       [data-testid="stDataFrame"] {
        background-color: light-dark(#f3f4f6, #1f2937);
    }

      [data-testid="stDataFrame"] table {
        background-color: light-dark(#f3f4f6, #1f2937) !important;
    }

    [data-testid="stDataFrame"] th {
        background-color: light-dark(#e5e7eb, #374151) !important;
    }

    [data-testid="stDataFrame"] td {
        background-color: light-dark(#f3f4f6, #1f2937) !important;
    }

</style>
""", unsafe_allow_html=True)

pg = st.navigation([
  st.Page('overview.py', title='Overview'),
  st.Page('model_perfromance.py', title='Model Performance'),
  st.Page('customer.py', title='Customer Lookup')
])

pg.run()