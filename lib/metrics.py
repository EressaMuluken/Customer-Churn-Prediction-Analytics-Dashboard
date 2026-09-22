import pandas as pd 
import numpy as np 
from sklearn.metrics import accuracy_score,precision_score, recall_score,f1_score, roc_auc_score, average_precision_score, confusion_matrix

def computePSI(train_data:pd.Series,production_data:pd.Series, bins:int = 10): 
  if pd.api.types.is_numeric_dtype(train_data): 
    bins = np.quantile(
      train_data, 
      np.linspace(0,1,bins + 1)
    )
    bins = np.unique(bins)
    
    if len(bins) < 2: 
      return np.nan
    
    train_dist = pd.cut(train_data, bins=bins, include_lowest=True)
    production_dist= pd.cut(production_data, bins=bins, include_lowest=True)
    
    train_dist = train_dist.value_counts(normalize=True, sort=False)
    production_dist = production_dist.value_counts(normalize=True, sort=False)
  else: 
    categories = set(list(train_data.unique()) + list(production_data.unique()))
    train_dist = train_data.value_counts(normalize=True).reindex(index=categories, fill_value=0)
    production_dist = production_data.value_counts(normalize=True).reindex(index=categories, fill_value=0)
  
  train_dist = train_dist.clip(lower=1e-6)
  production_dist = production_dist.clip(lower=1e-6)
  
  psi = ((train_dist - production_dist) * np.log(train_dist / production_dist)).sum()
  
  return psi.round(3)

def evaluate_model(model, x, y, threshold=0.5): 
  
  predict_prob = model.predict_proba(x)[:,1]
  
  y_pred = (predict_prob >= threshold).astype(int)
  
  metrics = {
    'accuracy': float(accuracy_score(y,y_pred)),
    'precision': float(precision_score(y, y_pred, zero_division=0)),
    'recall': float(recall_score(y,y_pred, zero_division=0)), 
    'f1': float(f1_score(y,y_pred, zero_division=0)), 
    'roc_auc': float(roc_auc_score(y, predict_prob)),
    'pr_auc': float(average_precision_score(y,predict_prob))
  }
  
  cmatrix = confusion_matrix(y, y_pred)
  
  return y_pred, predict_prob, metrics, cmatrix


  