import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from constant import features, numeric_features, categorical_features, target
from preprocessor import preprocessing
df = pd.read_excel('../data/cleaned_Telco_consumer_churn.xlsx')

X = df[features]
y = df[target]

X_train, X_val_test, y_train, y_val_test = train_test_split(X,y, test_size=0.3, stratify=y) 
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, stratify=y_val_test)

print(X_train.shape, X_val.shape, X_test.shape)
print(y_train.mean(), y_val.mean(), y_test.mean())