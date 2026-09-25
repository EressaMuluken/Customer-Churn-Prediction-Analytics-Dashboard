import streamlit as st 

st.set_page_config(layout='wide')

pg = st.navigation([
  st.Page('overview.py', title='Overview'),
  st.Page('model_perfromance.py', title='Model Performance'),
  st.Page('customer.py', title='Customer Lookup')
])

pg.run()