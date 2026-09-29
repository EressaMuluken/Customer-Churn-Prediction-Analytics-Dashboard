import streamlit as st 
import pandas as pd 
import joblib 
from lib.constant import features
st.set_page_config(
    initial_sidebar_state='locked',
    layout='wide'
)

@st.cache_data
def get_data(path): 
  return pd.read_excel(path)
@st.cache_resource
def get_bestModel(path): 
  return joblib.load()
#df = get_data('./data/cleaned_Telco_consumer_churn.xlsx')
df = st.session_state['data'] 
X = df[features]
      
for name, model in st.session_state['current_model'].items(): 
  df[f'{name}_predicted_Churn_probability'] = model[name].predict_proba(X)[:,1]
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
    st.markdown(
    f"""
    <div style="
        padding:16px;
        border-radius:10px;        
        border:1px solid #444;
        text-align:left;
        width:100%;
        margin-bottom:24px;
    ">
        <div style="
            font-weight:700;
            text-transform:uppercase;
            opacity:0.7;
            width:100%;
        ">
            {churn_db[i]['title']}
        </div>
        <div style="
            font-size:24px;
            font-weight:700;
            margin-top:2px;
            color:{'rgba(255,0,0,0.8)' if (churn_db[i]['title'] == 'Top Reasons' or churn_db[i]['title'] == 'CLTV at Risk') else 'rgba(255,255,255,0.8)'};
            width: 100%;
        ">
            {churn_db[i]['value']}
        </div>
         <div style="
            font-size:14px;
            margin-top:2px;
            color:gray; 
            width:100%;
        ">
            <span>{churn_db[i]['description']}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
df_churned = df[df['Churn Value'] == 1]
col_map, col_reasons = st.columns(2)  
with col_map: 
  with st.container(border=True):
    st.markdown(f"""
                <div>
                <span style="margin-bottom:0px;font-weight:800;opacity:0.5;">Churn by Geography</span>
                <span style="margin-left:8px;color:gray; font-size:14px;"> Latitude / Longitude</span>
                <hr style="margin-top:4px;"/>
                </div>
                """, unsafe_allow_html=True)
    #st.divider()
    st.map(df_churned,latitude='Latitude', longitude='Longitude', color='#FF0000', height=342, size=df_churned['Churn Score'].to_json())
with col_reasons: 
  with st.container(border=True):
    churn_reasons = df["Churn Reason"].value_counts().head(11)
    max_count = churn_reasons.max()
    total = len(df['Churn Reason'].value_counts())
    st.markdown(f"""
              <div>
              <span style="margin-bottom:0px;font-weight:800;opacity:0.5;">Churn Reasons</span>
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
                  align-items: center;
                  margin-bottom: 6px;
                  font-size: 14px;
              ">
                  <span style="width:260px;">{reason}</span>
                <div style="
                  width: 60%;
                  height: 100%;
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
                  <span style="padding-left:8px;font-weight: 600;">{count}</span>
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
            <span style="margin-bottom:0px;font-weight:800;opacity:0.5;">Churn rate by service
            </span>
            <span style="margin-left:8px;color:gray; font-size:14px;"> Security / Backup / Device Protection / Tech Support / Streaming</span>
            <hr style="margin-top:0px;"/>
            </div>
            """, unsafe_allow_html=True)
  st.dataframe(churn_rates,column_config={
        "Option": st.column_config.TextColumn(
            "Option",
            width="medium"
        ),
         "Online Security": st.column_config.NumberColumn(
            "Online Security",
            format="%.2f",
            width="medium"
        ),
          "Online Backup": st.column_config.NumberColumn(
            "Online Backup",
            format="%.2f",
            width="medium"
        ),
           "Device Protection": st.column_config.NumberColumn(
            "Device Protection",
            format="%.2f",
            width="medium"
        ),
            "Tech Support": st.column_config.NumberColumn(
            "Tech Support",
            format="%.2f",           
            width="medium"
        ),
            "Streaming TV": st.column_config.NumberColumn(
            "Streaming TV",
            format="%.2f",           
            width="medium"
        ),
            "Streaming Movies": st.column_config.NumberColumn(
            "Streaming Movies",
            format="%.2f",           
            width="medium"
        )
        })
with st.container(border=True): 
  sample_cols = ['CustomerID', 'State', 'Contract', 'Tenure Months', 'Monthly Charges', 'CLTV', 'Churn Score', 'Churn Reason']
  df_sample = df[df['Churn Value'] == 1][sample_cols]
  st.markdown(f"""
              <div>
              <span style="margin-bottom:0px;font-weight:800;opacity:0.5;">Churned Customer Detail</span>
              <span style="margin-left:8px;color:gray; font-size:14px;">{len(df_sample)} of {len(df)} - Churn Value = 1</span>
              <hr style="margin-top:0px;"/>
              </div>
              """, unsafe_allow_html=True)
  
  st.dataframe(df_sample,column_config={
        "CustomerID": st.column_config.TextColumn(
            "CustomerID",
            width="small"
        ),
         "State": st.column_config.TextColumn(
            "State",
            width="small"
        ),
          "Contract": st.column_config.TextColumn(
            "Contract",
            width="small"
        ),
           "Tenure Months": st.column_config.TextColumn(
            "Tenure Months",
            width='small'
        ),
            "Monthly Charges": st.column_config.TextColumn(
            "Monthly Charges",
            width='small'
        ),
            "CLTV": st.column_config.TextColumn(
            "CLTV",
            width='small'
        ),
            "Churn Score": st.column_config.TextColumn(
            "Churn Score",
            width="small"
        ),
            "Churn Reason": st.column_config.TextColumn(
            "Churn Reason",
            width="medium"
        )
        },hide_index=True)