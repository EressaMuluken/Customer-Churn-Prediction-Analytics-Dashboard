import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from models import models
from constant import features, numeric_features, categorical_features, target, threshold
from preprocessor import preprocessing
from metrics import evaluate_model
from constant import param_grids
from model_tuning import tune_model
from sklearn.metrics import precision_recall_curve, roc_curve
import joblib 
import json 
from datetime import datetime
print(f'Importing data ...')
df = pd.read_excel('../data/cleaned_Telco_consumer_churn.xlsx')

print(f'Selecting features ...')
X = df[features]
y = df[target]

print(f'Splitting data for training ...')
X_train, X_val_test, y_train, y_val_test = train_test_split(X,y, test_size=0.3, stratify=y) 
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, stratify=y_val_test)
print(f'Creating pipelines ...')
preprocessor = preprocessing(numeric_features=numeric_features, categorical_features=categorical_features)
pipelines = {}
for name, model in models.items():
  pipelines[name] = Pipeline([
    ('processor', preprocessor), 
    ('model', model)
  ])
  
trained_models = {}
for name, pipeline in pipelines.items(): 
  print(f'Training a {name} model ...')
  pipeline.fit(X_train, y_train)
  trained_models[name] = pipeline
  
validation_result = []

for name, model in trained_models.items(): 
  print(f'Evaluating {name} model on validation data ...')
  y_pred, y_prob, metrics, cmatrix = evaluate_model(model, X_val, y_val, threshold=0.5)
  validation_result.append({
    'model': name, 
    **metrics
  })
df_val = pd.DataFrame(validation_result)

df_val = df_val.sort_values('pr_auc', ascending=False)

print(df_val)

print(f'Preparing for model tuning ...')
best_model = {}
final_models = {}
for name, pipeline in pipelines.items(): 
 print(f'Tuning {name} model')
 result = tune_model(model=pipeline, x=X_train, y=y_train, cv=5, param=param_grids[name], scoring='f1',n_jobs=-1)
 best_model[name] = result.best_estimator_
 best_pipepline= Pipeline([
   ('preprocessor', preprocessor), 
   ('model', result.best_estimator_)
  ])
 metadata = {
   'model_name': name, 
   'target': target, 
   'features': features, 
   'numeric_features': numeric_features, 
   'categorical_features': categorical_features, 
   'training_date': str(datetime.now()), 
   'random_state': 42, 
   'threshold': threshold, 
   'best_params': result.best_params_, 
   'f1_bestscore': float(result.best_score_)
 }
 with open(f'../metadata/{name}_metadata.json', 'w') as f: 
   json.dump(metadata, f, indent=4)
 joblib.dump(best_pipepline, f"../models/{name}_pipeline.joblib")
 
test_result = []
feature_importance = {}
for name, model in best_model.items(): 
  print(f'Evaluating tuned {name} model on test data ...')
  y_pred, y_prob, metrics, cmatrix = evaluate_model(model, X_test, y_test, threshold=0.5)
  model_precision, model_recall, model_threshold = precision_recall_curve(y_test, y_prob)
  false_postive_rate, true_positive_rate, roc_threshold = roc_curve(y_test, y_prob)
  eval_metrics = {
    'model': name, 
    **metrics
  }
  df_prc = pd.DataFrame({
    'threshold': model_threshold,
    'precision': model_precision[-1], 
    'recall': model_recall[-1]  
  })
  df_prc['f1'] = 2 * (df_prc['precision'] * df_prc['recall']) / (df_prc['precision'] + df_prc['recall'])
  df_roc = pd.DataFrame({
    'fp rate': false_postive_rate, 
    'tp rate': true_positive_rate, 
    'threshold': roc_threshold
  })
  if hasattr(model, 'feature_importances_'): 
    feature_importance[name] = model.feature_importances_
  elif hasattr(model, 'coef_'): 
    feature_importance[name] = np.abs(model.coef_[0])
    
  df_prc.to_csv(f'../evaluation/{name}_precision_recall_curv.csv')
  df_roc.to_csv(f'../evaluation/{name}_roc.csv')
  
  test_result.append(eval_metrics)
  with open(f'../metrics/{name}_evaluation_metrics.json', 'w') as f: 
    json.dump(eval_metrics, f, indent=4)
    
feature_importance['features'] = preprocessor.get_feature_names_out()
df_feature_importance = pd.DataFrame(feature_importance).to_csv('../evaluation/feature_importance.csv')

df_test = pd.DataFrame(test_result)

df_test = df_test.sort_values('pr_auc', ascending=False)
print('Model Performance on TEST data\n')
print(df_test)  
df_test.to_csv('../evaluation/model_comparsion.csv')