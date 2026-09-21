import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from models import models
from constant import features, numeric_features, categorical_features, target
from preprocessor import preprocessing
from metrics import evaluate_model
from constant import param_grids
from model_tuning import tune_model

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

for name, pipeline in pipelines.items(): 
 print(f'Tuning {name} model')
 result = tune_model(model=pipeline, x=X_train, y=y_train, cv=5, param=param_grids[name], scoring='f1',n_jobs=-1)
 best_model[name] = result.best_estimator_
 
test_result = []

for name, model in best_model.items(): 
  print(f'Evaluating tuned {name} model on test data ...')
  y_pred, y_prob, metrics, cmatrix = evaluate_model(model, X_test, y_test, threshold=0.5)
  test_result.append({
    'model': name, 
    **metrics
  })
df_test = pd.DataFrame(test_result)

df_test = df_test.sort_values('pr_auc', ascending=False)
print('Model Performance on TEST data\n')
print(df_test)  