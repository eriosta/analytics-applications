import pandas as pd
import numpy as np
import zipfile
from tableone import TableOne
import wget
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from scipy.stats import uniform, randint
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from imblearn.over_sampling import SMOTE



# url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip'

# wget.download(url)

zf = zipfile.ZipFile("bank.zip") 

df = pd.read_csv(zf.open('bank-full.csv'), sep=';')

cols = list(df.drop(['y'],axis=1).columns)
groupby = ['y']

categorical = [
    'job','marital','education','default','loan','housing','contact','month','poutcome'
]

mytable = TableOne(
    df,
    columns=cols,
    categorical=categorical,
    groupby=groupby,
    pval=True)

print(mytable.tabulate(tablefmt = "fancy_grid"))

mytable.to_excel('bank_marketing/summary.xlsx')

df['y'] = np.where(df['y']=="yes",1,0)
import statsmodels.formula.api as smf
# maxiter = 35
# model = smf.logit("isNASH ~ Age + C(Gender) + C(Race) + C(isHighBMI) + C(isHighWC) + C(isHighGluc) + C(isHighTG) + C(isLowHDL) + C(isHighHOMAIR) + C(isPreDM) + C(isDM) + C(isHighALT)", data = data).fit(maxiter=maxiter)
model = smf.logit("y ~ age + C(job) + C(marital) + C(education) + C(default) + balance + C(housing) + C(loan) + C(contact) + day + C(month) + campaign + pdays + previous + C(poutcome)", data = df).fit()
model.summary()
model_odds = pd.DataFrame(np.exp(model.params), columns= ['OR'])
model_odds['z-value']= model.pvalues
model_odds[['2.5%', '97.5%']] = np.exp(model.conf_int())
model_odds.sort_values(by='OR', ascending=False).to_csv('bank_marketing/logit.csv')

data = df

X = data.drop(['y','duration'],axis=1)
y = data['y'].to_numpy()
feature_names =  list(data.drop(['y','duration'],axis=1).columns)

sum(X.pdays == -1) # is this the 999?

X['pdays'] = np.where(X['pdays']==-1,0,X['pdays'])

cats = [col for col in X if X[col].dtype != np.dtype('int64')]

def transform_one_hot(data, features_to_encode):
    for feature in features_to_encode:
        one_hot = (pd.get_dummies(data[feature])).add_prefix(feature + '_')
        data = data.join(one_hot)
        data = data.drop(feature,axis=1)
    return data

X = transform_one_hot(X,cats).to_numpy()

smote = SMOTE()
X, y = smote.fit_resample(X,y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

# import xgboost as xgb

# def report_best_scores(results, n_top=3):
#     for i in range(1, n_top + 1):
#         candidates = np.flatnonzero(results['rank_test_score'] == i)
#         for candidate in candidates:
#             print("Model with rank: {0}".format(i))
#             print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
#                   results['mean_test_score'][candidate],
#                   results['std_test_score'][candidate]))
#             print("Parameters: {0}".format(results['params'][candidate]))
#             print("")

# # hyperparameter searching
# xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)

# params = {
#     "colsample_bytree": uniform(0.7, 0.3),
#     "gamma": uniform(0, 0.5),
#     "learning_rate": uniform(0.03, 0.3), # default 0.1 
#     "max_depth": randint(3, 6), # default 3
#     "n_estimators": randint(100, 150), # default 100
#     "subsample": uniform(0.6, 0.4)
# }

# search = RandomizedSearchCV(xgb_model, param_distributions=params, random_state=42, n_iter=200, cv=3, verbose=1, n_jobs=1, return_train_score=True, scoring='roc_auc')

# search.fit(X_train, y_train)

# report_best_scores(search.cv_results_, 1)

# param = {'colsample_bytree': 0.944979831841473, 'gamma': 0.2195674542851092, 'learning_rate': 0.14308332882747227, 'max_depth': 5, 'n_estimators': 137, 'subsample': 0.8159124247658532}
# eval_set = [(X_train, y_train), (X_test, y_test)]


# xgb_model = xgb.XGBClassifier(objective="binary:logistic",
#                               random_state=42,
#                               colsample_bytree=param['colsample_bytree'],
#                               gamma=param['gamma'],
#                               learning_rate=param['learning_rate'],
#                               max_depth=param['max_depth'],
#                               n_estimators=param['n_estimators'],
#                               subsample=param['subsample'],
#                             #   scale_pos_weight=c1*100,
#                               booster='gbtree' # gbtree, dart, gblinear
#                               )


# xgb_model.fit(X_train, y_train,
#             eval_metric=["error", "logloss"],
#               eval_set=eval_set,
#               verbose=False)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(random_state=0, max_iter=1000).fit(X_train, y_train)

model.score(X, y)

y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]

# results = xgb_model.evals_result()
# epochs = len(results["validation_0"]["error"])
# x_axis = range(0, epochs)

# # plot log loss
# fig, ax = plt.subplots(figsize=(5,5))
# ax.plot(x_axis, results["validation_0"]["logloss"], label="Train")
# ax.plot(x_axis, results["validation_1"]["logloss"], label="Test")
# ax.legend()
# plt.ylabel("Log Loss")
# plt.title("XGBoost Log Loss")
# plt.show()
# # plot classification error
# fig, ax = plt.subplots(figsize=(5,5))
# ax.plot(x_axis, results["validation_0"]["error"], label="Train")
# ax.plot(x_axis, results["validation_1"]["error"], label="Test")
# ax.legend()
# plt.ylabel("Classification Error")
# plt.title("XGBoost Classification Error")
# plt.show()

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

from numpy import mean
# scores = cross_val_score(xgb_model, train_features, train_labels, scoring='roc_auc', cv=cv, n_jobs=-1)
# print("Mean ROC AUC for Training Set: %.5f" % mean(scores))

# scores = cross_val_score(xgb_model, test_features, test_labels, scoring='roc_auc', cv=cv, n_jobs=-1)
# print("Mean ROC AUC for Test Set: %.5f" % mean(scores))


import sklearn.metrics
import math
def matrix_metrix(real_values,pred_values,beta):
   CM = confusion_matrix(real_values,pred_values,)
   TN = CM[0][0]
   FN = CM[1][0] 
   TP = CM[1][1]
   FP = CM[0][1]
   Population = TN+FN+TP+FP
   Prevalence = round( (TP+FP) / Population,2)
   Accuracy   = round( (TP+TN) / Population,4)
   Precision  = round( TP / (TP+FP),4 )
   NPV        = round( TN / (TN+FN),4 )
   PPV        = round( TP / (TP + FP),4)
   FDR        = round( FP / (TP+FP),4 )
   FOR        = round( FN / (TN+FN),4 ) 
   check_Pos  = Precision + FDR
   check_Neg  = NPV + FOR
   Recall     = round( TP / (TP+FN),4 ) # Sensitivity, True Positive Rate
   FPR        = round( FP / (TN+FP),4 )
   FNR        = round( FN / (TP+FN),4 )
   TNR        = round( TN / (TN+FP),4 ) # Specificity, True Negative Rate (1 - False Positive Rate)
   check_Pos2 = Recall + FNR
   check_Neg2 = FPR + TNR
   LRPos      = round( Recall/FPR,4 ) 
   LRNeg      = round( FNR / TNR ,4 )
   DOR        = round( LRPos/LRNeg)
   F1         = round ( 2 * ((Precision*Recall)/(Precision+Recall)),4)
   FBeta      = round ( (1+beta**2)*((Precision*Recall)/((beta**2 * Precision)+ Recall)) ,4)
   MCC        = round ( ((TP*TN)-(FP*FN))/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))  ,4)
   BM         = Recall+TNR-1
   MK         = Precision+NPV-1
   mat_met = pd.DataFrame({
'Metric':['TP','TN','FP','FN','Prevalence','Accuracy','Precision','NPV','PPV','False Discovery Rate (FDR)','False Omission Rate (FOR)','check_Pos','check_Neg','Sensitivity (TPR, Recall)','False Positive Ratio (FPR)','False Negative Ratio (FNR)','Specificity (TNR)','check_Pos2','check_Neg2','LR+','LR-','Odds Ratio','F1','FBeta','MCC','BM','MK'],     'Value':[TP,TN,FP,FN,Prevalence,Accuracy,Precision,NPV,PPV,FDR,FOR,check_Pos,check_Neg,Recall,FPR,FNR,TNR,check_Pos2,check_Neg2,LRPos,LRNeg,DOR,F1,FBeta,MCC,BM,MK]})
   return (mat_met)


matrix_metrix(y_test, y_pred, 0.4)


from sklearn.metrics import plot_roc_curve, plot_confusion_matrix
ax = plt.gca()
rfc_disp = plot_roc_curve(xgb_model, X_test, y_test, ax=ax, alpha=0.8)
plt.show()



import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay

np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
titles_options = [
    ("Confusion matrix, without normalization", None),
    ("Normalized confusion matrix", "true"),
]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(
        xgb_model,
        X_test, 
        y_test,
        display_labels=['No','Yes'],
        cmap=plt.cm.Blues,
        normalize=normalize,
    )
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

plt.show()

# SHAP
import time
import shap
X_sampled = pd.DataFrame(X_test)
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_sampled)

# # summarize the effects of all the features
# shap.initjs()
shap.summary_plot(shap_values, X_sampled,plot_size=[8,5])


# for i, col in enumerate(X_sampled.columns):
#     print(col,i)
#     shap.dependence_plot(col, shap_values, X_sampled, show=False)
# plt.show()


