import streamlit as st 
import pandas as pd 
import joblib 
from lib.constant import features
st.set_page_config(
    initial_sidebar_state='locked',
    layout='wide'
)
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
    st.session_state['data'] = None
    
@st.cache_data
def get_data(path): 
  return pd.read_excel(path)
@st.cache_resource
def get_bestModel(path): 
  return joblib.load()
df = get_data('./data/cleaned_Telco_consumer_churn.xlsx')
print('df shape: ', df.shape)
st.session_state['data'] = df 
print('featues :', features)
X = df[features]
      
for name, model in st.session_state['current_model'].items(): 
  df[f'{name}_predicted_Churn_probability'] = model[name].predict_proba(X)[:,1]

print(df.head())
churn_db = [
  {
    'title': 'Churn Rate', 
    'value': f"{df['Churn Value'].mean():.1%}",
    'description': f"{(df['Churn Value'] == 1).sum()} of {len(df)} - Churn Value = 1"
  },
    {
    'title': 'Avg. Churn Score', 
    'value': f"{df['Churn Score'].mean():.2f}",
    'description': f"100% of data provided"
  }, 
  {
    'title': 'CLTV at Risk', 
    'value': f"${df[(df['Churn Value'] == 1) & (df['XGBoost_predicted_Churn_probability'] > 0.7)]['CLTV'].sum() / 1_000_000:.2f}M",
    'description': f"Churned + Churn Probability > 0.7"
  }, 
    {
    'title': 'AvG. Tenure Month (Churned)', 
    'value': f"{round(df[(df['Churn Value'] == 1)]['Tenure Months'].mean())} mo",
    'description': f"Vs {df[(df['Churn Value'] == 1)]['Tenure Months'].max():.0f} mo max retained"
  }, 
     {
    'title': 'Top Reasons', 
    'value': f"{df['Churn Reason'].value_counts().idxmax().split(' ',1)[:1][0]}",
    'description': f"{df['Churn Reason'].value_counts().idxmax().split(' ',1)[1:][0]}"
  },  
]
for i, col in enumerate(st.columns(len(churn_db))):
  with col: 
    st.metric(label=churn_db[i]['title'].upper(),
            value=churn_db[i]['value'], 
            delta_description=churn_db[i]['description'],
            border=True,
            width='stretch',
            )
df_churned = df[df['Churn Value'] == 1]
col_map, col_reasons = st.columns(2)  
with col_map: 
  with st.container(border=True):
    st.markdown(f"""
                <div>
                <span style="margin-bottom:0px;">Churn by Geography</span>
                <span style="margin-left:8px;color:gray; font-size:14px;"> Latitude / Longitude</span>
                <hr style="margin-top:4px;"/>
                </div>
                """, unsafe_allow_html=True)
    #st.divider()
    st.map(df_churned,latitude='Latitude', longitude='Longitude', color='#FF0000')
with col_reasons: 
  with st.container(border=True):
    churn_reasons = df["Churn Reason"].value_counts().head(11)
    max_count = churn_reasons.max()
    total = len(df['Churn Reason'].value_counts())
    st.markdown(f"""
              <div>
              <span style="margin-bottom:0px;">Churn Reasons</span>
              <span style="margin-left:8px;color:gray; font-size:14px;"> 11 out of {total}</span>
              <hr style="margin-top:4px; margin-bottom:11px;"/>
              </div>
              """, unsafe_allow_html=True)
    for reason, count in churn_reasons.items():
      percentage = count / max_count * 100
      st.markdown(
          f"""
          <div style="margin-bottom: 12px; width:100%;">
              <div style="
                  display: flex;
                  justify-content: space-between;
                  margin-bottom: 6px;
                  font-size: 14px;
              ">
                  <span>{reason}</span>
                  <span style="font-weight: 600;">{count}</span>
              </div>
              <div style="
                  width: 100%;
                  height: 8px;
                  background-color: #e5e7eb;
                  border-radius: 5px;
              ">
                  <div style="
                      width: {percentage}%;
                      height: 8px;
                      background-color: #B23A48;
                      border-radius: 5px;
                  ">
                  </div>
              </div>
          </div>
          """, unsafe_allow_html=True)
with st.container(border=True):
  cols = ['Online Security', 'Online Backup', 'Device Protection', 'Tech Support', 'Streaming TV', 'Streaming Movies']
  churn_rates = pd.concat(
    [
        (df.groupby(service)["Churn Value"].mean() * 100).rename(service)
        for service in cols
    ],
    axis=1
    )
  churn_rates.index.name = 'Option'
  st.markdown(f"""
            <div>
            <span style="margin-bottom:0px;">Churn rate by service
            </span>
            <span style="margin-left:8px;color:gray; font-size:14px;"> Security / Backup / Device Protection / Tech Support / Streaming</span>
            <hr style="margin-top:0px;"/>
            </div>
            """, unsafe_allow_html=True)
  st.dataframe(churn_rates)
with st.container(border=True): 
  sample_cols = ['CustomerID', 'State', 'Contract', 'Tenure Months', 'Monthly Charges', 'CLTV', 'Churn Score', 'Churn Reason']
  df_sample = df[df['Churn Value'] == 1][sample_cols]
  st.markdown(f"""
              <div>
              <span style="margin-bottom:0px;">Customer Detail</span>
              <span style="margin-left:8px;color:gray; font-size:14px;">{len(df_sample)} of {len(df)} - Churn Value = 1</span>
              <hr style="margin-top:0px;"/>
              </div>
              """, unsafe_allow_html=True)
  
  st.dataframe(df_sample)