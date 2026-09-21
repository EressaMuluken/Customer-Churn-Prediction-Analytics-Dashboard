from sklearn.model_selection import GridSearchCV

def tune_model(model, x, y, cv, param, n_jobs, scoring): 
  grid_search = GridSearchCV(estimator=model, cv=cv, param_grid=param, n_jobs=n_jobs, scoring=scoring, verbose=1)
  grid_search.fit(x,y)
  
  return grid_search
  