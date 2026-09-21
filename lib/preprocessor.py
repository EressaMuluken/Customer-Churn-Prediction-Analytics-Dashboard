from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def preprocessing(numeric_features, categorical_features, escape_numeric=True): 
  numeric_transformer = StandardScaler() if escape_numeric else 'passthrough'
  preprocessor = ColumnTransformer(transformers=[
    ('numeric', numeric_transformer, numeric_features), 
    ('categorical', OneHotEncoder(handle_unknown='ignore') , categorical_features )
  ])
  
  return preprocessor