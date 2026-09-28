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
#print('Selected model name: ', st.session_state['selected_model_name'])
#print('selected model in startup: ', st.session_state['selected_model'])
pg = st.navigation([
  st.Page('overview.py', title='Overview'),
  st.Page('model_perfromance.py', title='Model Performance'),
  st.Page('customer.py', title='Customer Lookup')
])

pg.run()