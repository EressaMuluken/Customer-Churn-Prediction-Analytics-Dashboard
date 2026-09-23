import streamlit as st 

st.set_page_config(layout='wide')

pg = st.navigation([
  st.Page('overview.py', title='Overview'),
  st.Page('risk_explorer.py', title='Risk Explorer'),
  st.Page('customer.py', title='Customer Lookup')
])

pg.run()