# https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html

import pandas as pd
import shap
import sklearn
import matplotlib.pyplot as plt 
import xgboost

path = 'HousingPrices.xlsx'

data = pd.read_excel(path)

data['PRICE'].describe()


X = data.drop(['PRICE'],axis=1)
cols = list(data.drop(['PRICE'],axis=1).columns)

feature_names = list(cols)
y = data['PRICE']
y = y.to_numpy()
  
X100 = shap.utils.sample(X, 100) # 100 instances for use as the background distribution

# train XGBoost model
model_xgb = xgboost.XGBRegressor(n_estimators=100, max_depth=2).fit(X, y)

# explain the GAM model with SHAP
explainer_xgb = shap.Explainer(model_xgb, X100)
shap_values_xgb = explainer_xgb(X)

for i in range(X.shape[1]):
    if 'PRICE':
        pass
    shap.plots.scatter(shap_values_xgb[:,i], color=shap_values_xgb)
    plt.show()