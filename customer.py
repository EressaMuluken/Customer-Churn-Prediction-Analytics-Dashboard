import streamlit as st 
import shap 
from lib.constant import features, target
from sklearn.model_selection import train_test_split
import pandas as pd 
import numpy as np 
df = st.session_state['data']

X = df[features]
y = df[target]

X_train, _, _, _ = train_test_split(X,y, test_size=0.3, stratify=y) 
#st.set_page_config(layout='centered')
explainer = {
        'Decision Tree':lambda model : shap.TreeExplainer(model), 
        'Gradient Boosting': lambda model: shap.TreeExplainer(model),
        'Logistic Regression': lambda model, x : shap.LinearExplainer(model, x),
        'Random Forest': lambda model : shap.TreeExplainer(model), 
        'XGBoost': lambda model: shap.TreeExplainer(model)
        } 
def get_pred(customer_df): 
  models = ['Decision Tree', 'Gradient Boosting','Logistic Regression','Random Forest', 'XGBoost']
  result_pred = {}
  result_prob = {}
  for item in models:
      model = st.session_state['current_model'][item][item]
      result_pred[item] = model.predict(customer_df)[0]
      result_prob[item] = model.predict_proba(customer_df)[0,1]
  prediction_df = pd.DataFrame(
    list(result_pred.items()),
    columns=['Model', 'Prediction']
  )
  probabilities_df = pd.DataFrame(
    list(result_prob.items()),
    columns=['Model', 'Probabilities']
  )
  return prediction_df, probabilities_df
def calculate_shap(customer_df, X_train): 
  model = st.session_state['selected_model'].named_steps['model']
  preprocessor = st.session_state['selected_model'].named_steps['processor']
  feature_names = preprocessor.get_feature_names_out()
  customer_transformed_data = preprocessor.transform(customer_df)
  X_train = preprocessor.transform(X_train)
  
  if st.session_state['selected_model_name'] == 'Logistic Regression': 
    model_explainer = explainer[st.session_state['selected_model_name']](model, X_train)
    customer_shap_values = model_explainer.shap_values(customer_transformed_data)

  else: 
     model_explainer = explainer[st.session_state['selected_model_name']](model)
     customer_shap_values = model_explainer.shap_values(customer_transformed_data)
  churne_values = customer_shap_values[:,:,1]
  customer_shap_df = pd.DataFrame({
    'names': np.array(feature_names).flatten(),
    'values': np.array(churne_values).flatten(),
  }
  )
  return customer_shap_df

customer_list = df['CustomerID']
current_customer = st.selectbox('Customer', options=customer_list)
customer_score = df.loc[df['CustomerID'] == current_customer,'Churn Score'].iloc[0]
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

customer_df = df[df['CustomerID'] == current_customer][features]
customer_shap_values = calculate_shap(customer_df, X_train)
customer_churn_prediction, customer_churn_prob = get_pred(customer_df)
Churn_value = customer_churn_prediction['Prediction'].sum()
customer_prediction_table = customer_churn_prob.merge(customer_churn_prediction,on='Model')
customer_prediction_table['Probabilities'] = customer_prediction_table['Probabilities'] * 100
customer_prediction_table['Prediction'] = customer_prediction_table['Prediction'].map({
  0:'Retained', 
  1:'Churned'
})
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
              padding: 0px 8px;
              border-radius: 16px;
              background-color:{ "rgba(255, 0, 0, 1)" if Churn_value > 2 else "rgba(0, 255, 0, 1)"};
              "><b>Prediction - {'Churned' if Churn_value > 2 else 'Retained'}</b>
              </p>
              
              </div>
              <div style="display:flex;justify-content:space-between;gap:10px;margin-top:0px;font-size:14px;color:gray;">
              <div>
              <span>{df.loc[df['CustomerID'] == current_customer,'City'].iloc[0]},</span>
              <span>{df.loc[df['CustomerID'] == current_customer,'State'].iloc[0]}</span>
              <span>{df.loc[df['CustomerID'] == current_customer,'Zip Code'].iloc[0]} - </span>
              <span>Lat {df.loc[df['CustomerID'] == current_customer,'Latitude'].iloc[0]},</span>
              <span>Long {df.loc[df['CustomerID'] == current_customer,'Longitude'].iloc[0]}</span>
              </div>
              <div>
              <p style="padding:0px;margin:0px;">{Churn_value} / 5 models predict churn</p>
              </div>
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

  with mid_col2:
    with st.container(border=True, height=424): 
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
                    border-radius:4px;background-color:rgba(122,128,112,0.5);
                    padding:2px 8px;
                    margin-left: 8px;
                    margin-bottom:4px;
                    ">
                    <p style="padding:0px;margin:0px;">{key}</p>
                    <div style="padding:0px 8px;font-size:14px;border-radius:16px;background-color:{'rgba(128,0,0,0.5)' if detail=='No' else 'rgba(0,128,0,0.5)'}">
                    <p style="padding:0px; margin:0px;">{detail}</p>
                    </div>
                    </div>
                    """,unsafe_allow_html=True)
  with st.container(border=True):
    b_col_1, b_col_2 = st.columns(2)

    with b_col_1: 
      st.markdown(f"""
                <div>
                <span style="margin-bottom:0px;font-weight:800;">Model Predictions</span>
                <hr style="margin-top:4px; margin-bottom:11px;"/>
                </div>
                """, unsafe_allow_html=True)
      st.dataframe(customer_prediction_table, hide_index=True)
    with b_col_2: 
      top_shap_df = customer_shap_values.sort_values(by='values', ascending=False).head(5)
      top_shap_df['names'] = top_shap_df['names'].str.replace('numeric__','').str.replace('categorical__','')
      top_shap_df['values'] = top_shap_df['values'] * 100
      top_shap_df.rename(columns={'names':'Feature Names', 'values':'Contribution (in %)'}, inplace=True)
      st.markdown(f"""
                <div>
                <span style="margin-bottom:0px;font-weight:800;">Local SHAP / Top Risk Factors</span>
                <hr style="margin-top:4px; margin-bottom:11px;"/>
                </div>
                """, unsafe_allow_html=True)
      st.dataframe(top_shap_df, hide_index=True)
  if Churn_value > 2: 
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