import streamlit as st 
import pandas as pd 
import joblib
@st.cache_data
def get_data(path): 
  return pd.read_excel(path)
for key in st.session_state: 
  st.session_state[key] = st.session_state[key] 
if 'data' not in st.session_state:
  st.session_state["data"] = get_data("./data/cleaned_Telco_consumer_churn.xlsx")
  
st.set_page_config(layout='wide')

pg = st.navigation([
  st.Page('overview.py', title='Overview'),
  st.Page('model_perfromance.py', title='Model Performance'),
  st.Page('customer.py', title='Customer Lookup')
])

pg.run()