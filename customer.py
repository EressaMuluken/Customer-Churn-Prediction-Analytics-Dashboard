import streamlit as st 
#st.set_page_config(layout='centered')

df = st.session_state['data']
customer_list = df['CustomerID']
current_customer = st.selectbox('Customer', options=customer_list)
customer_score = df.loc[df['CustomerID'] == current_customer,'Churn Score'].iloc[0]
Churn_value = 1
customer_account = {
  'Contract': df.loc[df['CustomerID'] == current_customer,'Contract'].iloc[0], 
  'Tenure': df.loc[df['CustomerID'] == current_customer,'Tenure Months'].iloc[0],
  'Payment Method':df.loc[df['CustomerID'] == current_customer,'Payment Method'].iloc[0],
  'Paperless billing': df.loc[df['CustomerID'] == current_customer,'Paperless Billing'].iloc[0],
  'Monthly charges': df.loc[df['CustomerID'] == current_customer,'Monthly Charges'].iloc[0],
  'Total charges': df.loc[df['CustomerID'] == current_customer,'Total Charges'].iloc[0]
}
customer_services = {
  'Phone Service': df.loc[df['CustomerID'] == current_customer,'Phone Service'].iloc[0], 
  'Mulitple Lines': df.loc[df['CustomerID'] == current_customer,'Multiple Lines'].iloc[0],
  'Internet Service':df.loc[df['CustomerID'] == current_customer,'Internet Service'].iloc[0],
  'Online Security': df.loc[df['CustomerID'] == current_customer,'Online Security'].iloc[0],
  'Online Backup': df.loc[df['CustomerID'] == current_customer,'Online Backup'].iloc[0],
  'Device Protection': df.loc[df['CustomerID'] == current_customer,'Device Protection'].iloc[0],
  
  'Tech Support': df.loc[df['CustomerID'] == current_customer,'Tech Support'].iloc[0], 
  'Tenure (in month)': df.loc[df['CustomerID'] == current_customer,'Tenure Months'].iloc[0],
  'Streaming TV':df.loc[df['CustomerID'] == current_customer,'Streaming TV'].iloc[0],
  'Streaming Movies': df.loc[df['CustomerID'] == current_customer,'Paperless Billing'].iloc[0],
}
customer_churn_reason = df.loc[df['CustomerID'] == current_customer,'Churn Reason'].iloc[0]
with st.container(border=True): 
  st.markdown(f"""
              <div style="
              display:flex;
              justify-content: space-between;
              align-items: center; 
              margin-bottom: 0px;
              "> 
              <p style="margin-bottom:0px;font-weight:800; font-size:20px;">{current_customer}</p>
              <p style="              
              background-color: rgb(0,0,128);
              margin-bottom:0px;
              color:white;
              padding: 2px 8px;
              border-radius: 16px;
              background-color:{ "rgba(255, 0, 0, 1)" if Churn_value == 1 else "rgba(0, 255, 0, 1)"};
              "><b>{'Churned' if Churn_value == 1 else 'Retained'} - Score {customer_score}</b></p>
              </div>
              <div style="display:flex;gap:10px;margin-top:0px;font-size:14px;color:gray;">
              <span>{df.loc[df['CustomerID'] == current_customer,'City'].iloc[0]},</span>
              <span>{df.loc[df['CustomerID'] == current_customer,'State'].iloc[0]}</span>
              <span>{df.loc[df['CustomerID'] == current_customer,'Zip Code'].iloc[0]} - </span>
              <span>Lat {df.loc[df['CustomerID'] == current_customer,'Latitude'].iloc[0]},</span>
              <span>Long {df.loc[df['CustomerID'] == current_customer,'Longitude'].iloc[0]}</span>
              </div>
              <div style="display:flex;gap:10px;margin-bottom:20px;margin-top:0px;font-size:14px;color:gray;">
              <span style="padding:0px 8px;
              border-radius:16px;
              background-color:rgba(128,128,128,0.5);">{df.loc[df['CustomerID'] == current_customer,'Gender'].iloc[0].capitalize()}</span>
              <span style="padding:0px 8px;
              border-radius:16px;
              background-color:rgba(128,128,128,0.5);">Senior Citizen: {df.loc[df['CustomerID'] == current_customer,'Senior Citizen'].iloc[0]}</span>
              <span style="padding:0px 8px;
              border-radius:16px;
              background-color:rgba(128,128,128,0.5);"> Partner: {df.loc[df['CustomerID'] == current_customer,'Partner'].iloc[0]}</span>
              <span style="padding:0px 8px;
              border-radius:16px;
              background-color:rgba(128,128,128,0.5);">Dependents: {df.loc[df['CustomerID'] == current_customer,'Dependents'].iloc[0]},</span>
              </div>
              """, unsafe_allow_html=True)
  score = int(df.loc[df['CustomerID'] == current_customer,'Churn Score'].iloc[0])

  if score < 40:
    score_color = "#2e7d32"      
  elif score < 70:
      score_color = "#d97706"      
  else:
      score_color = "#8b0000" 
  mid_col_1, mid_col2 = st.columns([1,3])
  with mid_col_1:
    with st.container(border=True): 
        st.markdown(f"""
                <div>
                <span style="margin-bottom:0px;font-weight:800;">Churn Score</span>
                <hr style="margin-top:4px; margin-bottom:11px;"/>
                </div>
                """, unsafe_allow_html=True)
        st.markdown(f"""
            <div style="display: flex;flex-direction: column;align-items: center;justify-content: center;">
            <div style="width: 110px;height: 110px;border: 6px solid {score_color};border-radius: 50%;display: flex;
                    flex-direction: column;align-items: center;justify-content: center;box-sizing: border-box;">

            <div style="font-size: 30px;font-weight: 700;line-height: 1;">{score}</div>

            <div style="font-size:14px;font-weight:800;margin-top: 4px;color:{score_color};"> CLTV ${df.loc[df['CustomerID'] == current_customer,'CLTV'].iloc[0] / 1_000:.1f}K</div>

            </div>
            </div>
            """,unsafe_allow_html=True)
    with st.container(border=True): 
      st.markdown(f"""
                <div>
                <span style="margin-bottom:0px;font-weight:800;">Account</span>
                <hr style="margin-top:4px; margin-bottom:11px;"/>
                </div>
                """, unsafe_allow_html=True)
      for key, detail in customer_account.items():
        st.markdown(f"""
                  <div style="padding:0px 8px;">
                  <p style="display:flex;justify-content:space-between;padding:0px;margin:2px;">
                  <span style="margin-bottom:0px;">{key}</span>
                  <span style="margin-bottom:0px;">{detail}</span>
                  </p>
                  <hr style="margin-top:0px; margin-bottom:0px;padding:0px;"/>
                  </div>
                  """, unsafe_allow_html=True)
    if Churn_value == 1: 
      with st.container(border=True):
         st.markdown(f"""
                <div>
                <span style="margin-bottom:0px;font-weight:800;">Churn Reason</span>
                <hr style="margin-top:4px; margin-bottom:11px;"/>
                </div>
                """, unsafe_allow_html=True)
         st.markdown(f"""
                <div style="margin-bottom:8px;">
                <p style="padding:0px 8px;margin-bottom:0px;font-weight:600;color:rgb(128,0,0)">{customer_churn_reason}</p>
                </div>
                """, unsafe_allow_html=True)
  with mid_col2:
    with st.container(border=True): 
      st.markdown(f"""
              <div>
              <span style="margin-bottom:0px;font-weight:800;">Services</span>
              <hr style="margin-top:4px; margin-bottom:11px;"/>
              </div>
              """, unsafe_allow_html=True)
      for key, detail in customer_services.items():
        st.markdown(f"""
                    <div style="display:flex;justify-content:space-between;
                    align-items:center;padding:0px 8px;border-radius:10px;opacity:0.9;
                    border-radius:4px;background-color:rgba(1282,128,112,0.5);
                    padding:2px 8px;
                    margin-bottom:4px;
                    ">
                    <p style="padding:0px;margin:0px;">{key}</p>
                    <div style="padding:0px 8px;border-radius:16px;background-color:{'rgba(128,0,0,0.5)' if detail=='No' else 'rgba(0,128,0,0.5)'}">
                    <p style="padding:0px; margin:0px;">{detail}</p>
                    </div>
                    </div>
                    """,unsafe_allow_html=True)