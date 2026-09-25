features = [
    "Gender",
    "Senior Citizen",
    "Partner",
    "Dependents",
    "Tenure Months",
    "Phone Service",
    "Multiple Lines",
    "Internet Service",
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Contract",
    "Paperless Billing",
    "Payment Method",
    "Monthly Charges",
    "Total Charges",
]
target = 'Churn Value'

threshold = 0.5

numeric_features = [
  "Tenure Months",
  "Monthly Charges",
  "Total Charges",
]

categorical_features = [
    "Gender",
    "Senior Citizen",
    "Partner",
    "Dependents",
    "Phone Service",
    "Multiple Lines",
    "Internet Service",
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Contract",
    "Paperless Billing",
    "Payment Method",
]

param_grids = {

    "Logistic Regression": {
        "model__C": [0.01, 0.1, 1, 10, 100],
        "model__solver": ["lbfgs", "liblinear"],
        "model__class_weight": [None, "balanced"]
    },
    
    "Decision Tree": {
        "model__criterion": ["gini", "entropy"],
        "model__max_depth": [3, 4, 5, 6, 7, 10, None],
        "model__min_samples_split": [10, 20, 50],
        "model__min_samples_leaf": [5, 10, 20, 30, 50],
        "model__class_weight": [None, "balanced"],
        "model__max_features": [None, "sqrt", "log2"],
        "model__min_impurity_decrease": [0.0, 0.001, 0.005, 0.01]
    },

    "Random Forest": {
        "model__n_estimators": [200, 300],
        "model__max_depth": [None, 5, 10, 15, 20, 30],
        "model__min_samples_leaf": [1, 2, 5, 10, 20],
        "model__max_features": ["sqrt", "log2", None],
        "model__class_weight": [None, "balanced"],
     
    },

    "Gradient Boosting": {
        "model__n_estimators": [100, 200, 300],
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__max_depth": [2, 3, 4],
        "model__min_samples_leaf": [5, 10, 20],
        "model__subsample": [0.8, 1.0],
        "model__max_features": [None, "sqrt", "log2"]
    },

    "XGBoost": {
        "model__n_estimators": [200, 300, 500],
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__max_depth": [2, 3, 4],
        "model__min_child_weight": [1, 5, 10],
        "model__subsample": [0.8, 1.0],
        "model__colsample_bytree": [0.8, 1.0]
    }
}

