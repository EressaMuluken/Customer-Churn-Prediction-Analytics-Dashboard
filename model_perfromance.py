import streamlit as st 
import json 
from itertools import islice
import pandas as pd 
import matplotlib.pyplot as plt 
def plot_roc(fpr, tpr, thresholds, roc_auc):
    fig, ax = plt.subplots(figsize=(8,4.6))
    ax.plot(fpr,tpr,linewidth=2,label=f"ROC curve (AUC = {roc_auc:.2f})")
    ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=1.2,
    alpha=0.6,
    label="Random classifier"
)
    for x,y, threshold in zip(fpr[::15], tpr[::15], thresholds[::15]):
        plt.annotate(f'{threshold:.2f}',
                     (x,y),
                     xytext=(5,5), 
                     textcoords='offset points',color='#6B7280')
    ax.set_xlabel('False positive rate',fontsize=10,color="#6B7280")
    ax.set_ylabel('True positive rate',fontsize=10,color="#6B7280")
    ax.set_title('ROC Curve', fontsize=10, fontweight='bold', color='#6B7280', pad=10)
    ax.tick_params(axis='both', labelsize=9, colors='#6B7280')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#BBBBBB")
    ax.spines["bottom"].set_color("#BBBBBB")
    ax.grid(True,linestyle="--",linewidth=0.5,alpha=0.25)
    ax.patch.set_alpha(0)
    ax.legend(fontsize=8,frameon=False,loc="lower right")
    fig.patch.set_alpha(0)
    plt.tight_layout()
    return fig 
    
models = [
        'Decision Tree', 
        'Gradient Boosting',
        'Logistic Regression',
        'Random Forest',
        'XGBoost'        
        ]
st.markdown(f""" 
            <span style="font-weight:800;text-transform:uppercase;"> Select Model</span> <span style="padding-left:8px;color:gray;font-size:14px;"> / Optimised Based on F1-Score</span>
            """, unsafe_allow_html=True)
model = st.selectbox('Select Model', options=models, label_visibility='collapsed')
with open(f'./metrics/{model}_evaluation_metrics.json', 'r') as f: 
  model_metrics = json.load(f)
cols = st.columns(len(model_metrics) - 2, width='stretch')
for col, (key, value) in zip(cols, islice(model_metrics.items(), 2, None)):
  with col: 
    st.markdown(
    f"""
    <div style="
        padding:16px;
        border-radius:10px;        border:1px solid #444;
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
            {key}
        </div>
        <div style="
            font-size:24px;
            font-weight:700;
            margin-top:2px;
            color:{'green' if key == 'roc_auc' else 'inherit'};
            width: 100%;
        ">
            {value:.2f}
        </div>
         <div style="
            font-size:14px;
            margin-top:2px;
            color:gray; 
            width:100%;
        ">
            <span>{'across all thresholds' if (key == 'roc_auc' or key == 'pr_auc') else 'at threshold &ge; 0.5'}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

mid_col1, mid_col2 = st.columns(2)
confusion_matrix = model_metrics['cm']
with mid_col2: 
    with st.container(border=True, width='stretch'): 
        st.markdown(f"""
            <div>
            <span style="margin-bottom:0px;font-weight:800;">Confusion matrix</span>
            <span style="margin-left:8px;color:gray; font-size:14px;"> threshold &ge; 0.5</span>
            <hr style="margin-top:4px;"/>
            </div>
            """, unsafe_allow_html=True)
        st.markdown(f"""
                <div style="
                display:grid;
                grid-template-columns: repeat(3, 1fr);
                width:100%;
                margin:auto;
                text-align:center;">
                   <div>
                   </div>
                   <div>
                   <p style="font-weight:800;opacity:0.75;margin-bottom:0px;">Predicted </p>
                   <p style="font-size:14px;color:gray;margin-top:0px;">No Churn </p>
                   </div>
                   <div>
                   <p style="font-weight:800;opacity:0.75;margin-bottom:0px;">Predicted </p>
                   <p style="font-size:14px;color:gray;margin-top:0px;">Churn </p>
                   </div>
                   
                   <div style="align-content:center;">
                   <p style="font-weight:800;opacity:0.75;margin-bottom:0px;">Actual</p>
                   <p style="font-size:14px;color:gray;margin-top:0px;">No Churn </p>
                   </div>
                   <div style="margin:4px;padding:20px;
                    border: 1px solid rgba(128, 0, 0, 0.25);
                    border-radius: 12px;
                    background-color: rgba(128, 0, 0, 0.08);
                    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
                    background-color:rgba(0, 255, 0, 0.25);text-weight:800;">
                   <p style="font-size:18px;margin-bottom:0px;">{confusion_matrix[0][0]}</p>
                   <p style="text-size:14px;color:gray;">true negative</p>
                   </div>
                    <div style="margin:4px;padding:20px;
                    border: 1px solid rgba(128, 0, 0, 0.25);
                    border-radius: 12px;
                    background-color: rgba(128, 0, 0, 0.08);
                    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
                    background-color:rgba(255,0, 0, 0.25);text-weight:800;"">
                    <p style="font-size:18px;margin-bottom:0px;">{confusion_matrix[0][1]}</p>
                   <p style="text-size:14px;color:gray;">false positive</p>

                   </div>
                    <div style="align-content:center;">
                   <p style="font-weight:800;opacity:0.75;margin-bottom:0px;">Actual</p>
                   <p style="font-size:14px;color:gray;margin-top:0px;">Churn </p>
                   </div>
                   <div style="margin:4px;padding:20px;
                    border: 1px solid rgba(128, 0, 0, 0.25);
                    border-radius: 12px;
                    background-color: rgba(128, 0, 0, 0.08);
                    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
                    background-color:rgba(255, 0, 0, 0.25);text-weight:800;">
                    <p style="font-size:18px;margin-bottom:0px;">{confusion_matrix[1][0]}</p>

                   <p style="text-size:14px;color:gray;">false negative</p>

                   </div>
                    <div style="margin:4px;padding:20px;
                    border: 1px solid rgba(128, 0, 0, 0.25);
                    border-radius: 12px;
                    background-color: rgba(128, 0, 0, 0.08);
                    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
                    background-color:rgba(0, 255, 0, 0.25);text-weight:800;"">
                    <p style="font-size:18px;margin-bottom:0px;">{confusion_matrix[1][1]}</p>
                   <p style="text-size:14px;color:gray;">true postive</p>

                   </div>
                
                
                </div>
                <div style="height:50px;width:100%;"></div>
                
                """, unsafe_allow_html=True)

roc = pd.read_csv(f'./evaluation/{model}_roc.csv')

with mid_col1: 
    with st.container(border=True):
        fig = plot_roc(roc['fp rate'], roc['tp rate'], roc['threshold'], model_metrics['roc_auc'])
        st.pyplot(fig)