import streamlit as st 
import json 
from itertools import islice
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
def plot_f1_threshold(thresholds, f1,f1_score): 
    n_points = 10
    indices = np.linspace(0,len(thresholds) - 1,n_points,dtype=int)
    points_thresholds = thresholds[indices]
    points_f1 = f1[indices]
    opt_index = f1.idxmax()
    opt_f1 = f1.loc[opt_index]
    fig, ax = plt.subplots(figsize=(3,2))
    ax.plot(thresholds,f1,linewidth=1,label=f"F1 (score = {f1_score:.2f})", color='#B23A48')
    ax.axvline(thresholds.loc[opt_index],linestyle="--",linewidth=1,alpha=0.6,label=f"Threshold (optimal = {thresholds.loc[opt_index]:.3f})")
    ax.plot(thresholds.loc[opt_index],opt_f1,marker='o',markersize=2)
    for x,y in zip(points_thresholds,points_f1):
        plt.annotate(f'{y:.2f}',(x,y),xytext=(5,5),fontsize=4,textcoords='offset points',color='#6B7280')
    #ax.set_xlabel('Threshold',fontsize=10,color="#6B7280")
    #ax.set_ylabel('F1 score',fontsize=10,color="#6B7280")
    ax.set_title('F1 score', fontsize=5, fontweight='bold', color='#6B7280', pad=10)
    ax.tick_params(axis='both', labelsize=4, colors='#6B7280')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#6B7280")
    ax.spines["bottom"].set_color("#6B7280")
    ax.spines['left'].set_alpha(0.2)
    ax.spines['bottom'].set_alpha(0.2)
    ax.grid(True,linestyle="--",linewidth=0.5,alpha=0.25)
    ax.patch.set_alpha(0)
    legend = ax.legend(fontsize=5,frameon=False,loc="lower left")
    plt.setp(legend.get_texts(), color="#6B7280")
    fig.patch.set_alpha(0)
    plt.tight_layout()
    return fig  
def plot_pr_auc(recall, percision, thresholds, pr_auc, churn_prop): 
    n_points = 10
    indices = np.linspace(0,len(thresholds) - 1,n_points,dtype=int)
    points_precision = percision[indices]
    points_recall = recall[indices]
    points_thresholds = thresholds[indices]
    
    fig, ax = plt.subplots(figsize=(3,2))
    ax.plot(recall,percision,linewidth=1,label=f"PR curve (AUC = {pr_auc:.2f})", color='#B23A48')
    ax.axhline(churn_prop,linestyle="--",linewidth=1,alpha=0.6,label=f"Random classifier ({churn_prop:.2f})")
    
    for x,y, threshold in zip(points_recall,points_precision, points_thresholds):
        plt.annotate(f'{threshold:.2f}',(x,y),xytext=(5,5),fontsize=4,textcoords='offset points',color='#6B7280')
    #ax.set_xlabel('Recall',fontsize=10,color="#6B7280")
    #ax.set_ylabel('Precision',fontsize=10,color="#6B7280")
    ax.set_title('PR Curve', fontsize=5, fontweight='bold', color='#6B7280', pad=10)
    ax.tick_params(axis='both', labelsize=4, colors='#6B7280')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#6B7280")
    ax.spines["bottom"].set_color("#6B7280")
    ax.spines['left'].set_alpha(0.2)
    ax.spines['bottom'].set_alpha(0.2)
    ax.grid(True,linestyle="--",linewidth=0.5,alpha=0.25)
    ax.patch.set_alpha(0)
    legend = ax.legend(fontsize=5,frameon=False,loc="upper right")
    plt.setp(legend.get_texts(), color="#6B7280")
    fig.patch.set_alpha(0)
    plt.tight_layout()
    return fig  
def plot_roc(fpr, tpr, thresholds, roc_auc):
    n_points = 10
    indices = np.linspace(0,len(thresholds) - 1,n_points,dtype=int)
    points_fpr = fpr[indices]
    points_tpr = tpr[indices]
    points_thresholds = thresholds[indices]
    
    fig, ax = plt.subplots(figsize=(3,2))
    ax.plot(fpr,tpr,linewidth=1,label=f"ROC curve (AUC = {roc_auc:.2f})", color='#B23A48')
    ax.plot([0, 1],[0, 1],linestyle="--",linewidth=1,alpha=0.6,label="Random classifier")
    for x,y, threshold in zip(points_fpr, points_tpr, points_thresholds):
        plt.annotate(f'{threshold:.2f}',(x,y),xytext=(5,5),fontsize=4,textcoords='offset points',color='#6B7280')
    #ax.set_xlabel('False positive rate',fontsize=5,color="#6B7280")
    #ax.set_ylabel('True positive rate',fontsize=5,color="#6B7280")
    ax.set_title('ROC Curve', fontsize=5, fontweight='bold', color='#6B7280', pad=10)
    ax.tick_params(axis='both', labelsize=4, colors='#6B7280')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#6B7280")
    ax.spines["bottom"].set_color("#6B7280")
    ax.spines['left'].set_alpha(0.2)
    ax.spines['bottom'].set_alpha(0.2)
    ax.grid(True,linestyle="--",linewidth=0.5,alpha=0.25)
    ax.patch.set_alpha(0)
    legend = ax.legend(fontsize=5,frameon=False,loc="lower right")
    plt.setp(legend.get_texts(), color="#6B7280")
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
            <span style="font-weight:700;text-transform:uppercase;opacity:0.8;"> Select Model</span> <span style="padding-left:8px;color:gray;font-size:14px;"> / Optimised Based on F1-Score</span>
            """, unsafe_allow_html=True)
model = st.selectbox('Select Model', options=models, label_visibility='collapsed')
with open(f'./metadata/{model}_metadata.json', 'r') as f: 
    model_metadata = json.load(f)
    
st.markdown(f"""
             <div style="display:flex;justify-content:space-between;gap:10px;padding-top:0;margin-top:0px;margin-bottom:10px;font-size:14px;font-weight:700;opacity:0.8;">
             <div style="display:flex;gap:10px;justify-content:start;width:75%;">
              <p style="
              margin-top:0px;
              margin-bottom:10px;
              padding:2px 8px;
              border-radius:16px;
              font-weight:600;
              opacity:0.9;
              background-color:#17202B;">Algorithm :-
              <span style="margin-left:8px;font-size:14px;font-weight:600;
              "> {model_metadata['model_name']}</span></p>
              <p style="
              margin-top:0px;
              margin-bottom:10px;
              padding:2px 8px;
              border-radius:16px;
              font-weight:600;
              opacity:0.9;
               background-color:#17202B;">Training data :- 
              <span style="margin-left:8px;font-size:14px;"> {model_metadata['training_date'].split(' ')[0]}</span></p>
              <p style="
              margin-top:0px;
              margin-bottom:10px;
              padding:2px 8px;
              border-radius:16px;
              font-weight:600;
              opacity:0.9;
               background-color:#17202B;">Numerical Features :-
              <span style="margin-left:8px;font-size:14px;"> {len(model_metadata['numeric_features'])}</span></p>
              <p style="
              margin-top:0px;
              margin-bottom:10px;
              padding:2px 8px;
              border-radius:16px;
              font-weight:600;
              opacity:0.9;
               background-color:#17202B;">Categorical Features :- 
              <span style="margin-left:8px;font-size:14px;"> {len(model_metadata['categorical_features'])}</span></p>
              </div>
              <div style="display:flex;gap:10px;justify-content:end;width:25%;">
              <p style="
              margin-top:0px;
              margin-bottom:10px;
              padding:2px 8px;
              border-radius:16px;
              font-weight:600;
              opacity:0.9;
               background-color:#17202B;">Threshold :- 
              <span style="margin-left:8px;font-size:14px;"> {model_metadata['threshold']}</span></p>
              <p style="
              margin-top:0px;
              margin-bottom:10px;
              padding:2px 8px;
              border-radius:16px;
              font-weight:600;
              opacity:0.9;
               background-color:#17202B;">Target :-
              <span style="margin-left:8px;font-size:14px;"> {model_metadata['target']}</span></p>
              </div>
              
              <div> 
            
            """, unsafe_allow_html=True)
st.session_state['selected_model_name'] = model
st.session_state['selected_model'] = st.session_state['current_model'][model][model]
with open(f'./metrics/{model}_evaluation_metrics.json', 'r') as f: 
  model_metrics = json.load(f)
cols = st.columns(len(model_metrics) - 2, width='stretch')
for col, (key, value) in zip(cols, islice(model_metrics.items(), 2, None)):
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
            opacity:0.8;
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
            {value:.2%}
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
            <span style="margin-bottom:0px;font-weight:700;opacity:0.8">Confusion matrix</span>
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
                <div style="height:27px;width:100%;"></div>
                
                """, unsafe_allow_html=True)

roc = pd.read_csv(f'./evaluation/{model}_roc.csv')

with mid_col1: 
    with st.container(border=True):
        fig = plot_roc(roc['fp rate'], roc['tp rate'], roc['threshold'], model_metrics['roc_auc'])
        st.pyplot(fig, width='content')
        
        
pr_auc = pd.read_csv(f'./evaluation/{model}_precision_recall_curv.csv')
churn_prop = st.session_state['data']['Churn Value'].mean()
pr_col, thr_col = st.columns(2)
with pr_col: 
    with st.container(border=True):
        fig = plot_pr_auc(pr_auc['recall'], pr_auc['precision'], pr_auc['threshold'], model_metrics['pr_auc'], churn_prop)
        st.pyplot(fig)
with thr_col: 
    with st.container(border=True): 
        fig = plot_f1_threshold(pr_auc['threshold'], pr_auc['f1'],model_metrics['f1'])
        st.pyplot(fig)
compare_array = []
for item in models: 
    with open(f'./metrics/{item}_evaluation_metrics.json', 'r') as f: 
        metrics = json.load(f)
        result = {k:v for k, v in metrics.items() if k not in ['cm']}
        compare_array.append(result)
compare_model_df = pd.DataFrame(compare_array)
cols_metrics = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc"
]
compare_model_df[cols_metrics]= compare_model_df[cols_metrics] * 100
compare_model_df.columns = compare_model_df.columns.str.upper()
with st.container(border=True): 
    st.markdown(f"""
              <div>
              <span style="margin-bottom:0px;font-weight:700;opacity:0.8;">Model Comparsion</span>
              <span style="margin-left:8px;color:gray; font-size:14px;">/ Optimised based on F1-score</span>
              <hr style="margin-top:0px;"/>
              </div>
              """, unsafe_allow_html=True)
    st.dataframe(compare_model_df, column_config={
         "MODEL": st.column_config.TextColumn(
            "MODEL",
            width='medium'
        ),
        "ACCURACY": st.column_config.ProgressColumn(
            "ACCURACY",
            format="%.1f%%",
            min_value=0,
            max_value=100
        ),
        "PRECISION": st.column_config.ProgressColumn(
            "PRECISION",
            format="%.1f%%",
            min_value=0,
            max_value=100
        ),
        "RECALL": st.column_config.ProgressColumn(
            "RECALL",
            format="%.1f%%",
            min_value=0,
            max_value=100
        ),
        "F1": st.column_config.ProgressColumn(
            "F1",
            format="%.1f%%",
            min_value=0,
            max_value=100
        ),
        "ROC_AUC": st.column_config.ProgressColumn(
            "ROC_AUC",
            format="%.1f%%",
            min_value=0,
            max_value=100
        ),
        "PR_AUC": st.column_config.ProgressColumn(
            "PR_AUC",
            format="%.1f%%",
            min_value=0,
            max_value=100
        ),
    },hide_index=True)


with st.expander('Model Configuration'):
    st.markdown(f"""
              <div>
              <p style="margin-bottom:0px;">Algorithm: 
              <span style="margin-left:8px;color:gray; font-size:14px;"> {model_metadata['model_name']}</span></p>
              <p style="margin-bottom:0px;">Training data: 
              <span style="margin-left:8px;color:gray; font-size:14px;"> {model_metadata['training_date']}</span></p>
              <p style="margin-bottom:0px;">Features: 
              <span style="margin-left:8px;color:gray; font-size:14px;"> {len(model_metadata['features'])}</span></p>
              <p style="margin-bottom:0px;">Numerical Features: 
              <span style="margin-left:8px;color:gray; font-size:14px;"> {len(model_metadata['numeric_features'])}</span></p>
              <p style="margin-bottom:0px;">Features: 
              <span style="margin-left:8px;color:gray; font-size:14px;"> {len(model_metadata['categorical_features'])}</span></p>
              <p style="margin-bottom:0px;">Target: 
              <span style="margin-left:8px;color:gray; font-size:14px;"> {model_metadata['target']}</span></p>
              <p style="margin-bottom:0px;">Classification threshold: 
              <span style="margin-left:8px;color:gray; font-size:14px;"> {model_metadata['threshold']}</span></p>
              <p style="margin-bottom:0px;">Random state: 
              <span style="margin-left:8px;color:gray; font-size:14px;"> {model_metadata['random_state']}</span></p>
              </div>
              
              """, unsafe_allow_html=True)
with st.expander('Best hyperparameters'):
    best_params = model_metadata['best_params']
    for key, value in best_params.items():
        st.markdown(f"""
                <div>
                <p style="margin-bottom:0px;">{str(key).removeprefix('model__')}: 
                <span style="margin-left:8px;color:gray; font-size:14px;"> {value}</span></p>
                </div>
                
                """, unsafe_allow_html=True)
   
    