import pandas as pd
import shap
import sklearn
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.stats import uniform, randint
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error, log_loss, accuracy_score
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split
import sklearn.metrics
import math
from imblearn.over_sampling import SMOTE

print(
    "shap ==",shap.__version__,
    "sklearn ==",sklearn.__version__,
    "xgboost ==",xgb.__version__
)



add = [
'Age',
'Income',
# 'Gender',
'PercWhite',
'PercBlack',
'MedAge',
'AvgAge',
'PercElder', 
'PercMarr', 
'PercSingle', 
'MedHomeVal',
'AvgHomeVal',
'MedRent', 
'AvgRent',
'MedHousInc', 
'MedFamInc', 
'AvgHousInc',
'AvgFamInc', 
'PerCapInc', 
'PercProf',
'PercManage', 
'PercSales', 
'PercMech', 
'PercLab',
'PercStateBr', 
'LifeGiftDol', 
'LifeGiftNum', 
'LifeGiftPromNum', 
'MinDol', 
'MaxDol', 
'LastDol ',
'AvgDol', 
'AmDonated',
'Donation'
]

path = 'donation_prediction/Customer_Analytics_TrainTest.csv'
data = pd.read_csv(path)[add] # .drop(drop,axis=1)

data = data.query('Age >= 18 and (AmDonated >= 20 or AmDonated == 0)').dropna().drop('AmDonated',axis=1)

sum(data['Donation'] == 1)/len(data)

dat = data.drop(['Donation'],axis=1)
cols = list(data.drop(['Donation'],axis=1).columns)

feature_names = list(cols)
labels = data['Donation']
labels = labels.to_numpy()

smote = SMOTE()
dat, labels = smote.fit_resample(dat,labels)

train_features, test_features = train_test_split(dat, test_size=0.5,random_state=42)
train_labels, test_labels = train_test_split(labels, test_size=0.5,random_state=42)

# weights
c1 = 1 - sum(data['Donation'] == 1)/len(data)
c0 = 1 - sum(data['Donation'] == 0)/len(data)

def report_best_scores(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

# hyperparameter searching
xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42, scale_pos_weight=c1*100)

params = {
    "colsample_bytree": uniform(0.7, 0.3),
    "gamma": uniform(0, 0.5),
    "learning_rate": uniform(0.03, 0.3), # default 0.1 
    "max_depth": randint(3, 6), # default 3
    "n_estimators": randint(100, 150), # default 100
    "subsample": uniform(0.6, 0.4)
}

search = RandomizedSearchCV(xgb_model, param_distributions=params, random_state=42, n_iter=200, cv=3, verbose=1, n_jobs=-1, return_train_score=True, scoring='roc_auc')

# search.fit(train_features, train_labels)

# report_best_scores(search.cv_results_, 1)

param =   {'colsample_bytree': 0.9751940726386727, 'gamma': 0.0684093154594807, 'learning_rate': 0.31507120614624073, 'max_depth': 5, 'n_estimators': 148, 'subsample': 0.6740531715354479}
eval_set = [(train_features, train_labels), (test_features, test_labels)]

xgb_model = xgb.XGBClassifier(objective="binary:logistic",
                              random_state=42,
                              colsample_bytree=param['colsample_bytree'],
                              gamma=param['gamma'],
                              learning_rate=param['learning_rate'],
                              max_depth=param['max_depth'],
                              n_estimators=param['n_estimators'],
                              subsample=param['subsample'],
                              #scale_pos_weight=c1*100,
                              booster='gbtree' # gbtree, dart, gblinear
                              )


xgb_model.fit(train_features, train_labels,
            eval_metric=["error", "logloss"],
              eval_set=eval_set,
              verbose=False)

y_pred = xgb_model.predict(test_features)
y_pred_prob = xgb_model.predict_proba(test_features)[:, 1]

results = xgb_model.evals_result()
epochs = len(results["validation_0"]["error"])
x_axis = range(0, epochs)

# plot log loss
fig, ax = plt.subplots(figsize=(5,5))
ax.plot(x_axis, results["validation_0"]["logloss"], label="Train")
ax.plot(x_axis, results["validation_1"]["logloss"], label="Test")
ax.legend()
plt.ylabel("Log Loss")
plt.title("XGBoost Log Loss")
plt.show()
# plot classification error
fig, ax = plt.subplots(figsize=(5,5))
ax.plot(x_axis, results["validation_0"]["error"], label="Train")
ax.plot(x_axis, results["validation_1"]["error"], label="Test")
ax.legend()
plt.ylabel("Classification Error")
plt.title("XGBoost Classification Error")
plt.show()

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

from numpy import mean
# scores = cross_val_score(xgb_model, train_features, train_labels, scoring='roc_auc', cv=cv, n_jobs=-1)
# print("Mean ROC AUC for Training Set: %.5f" % mean(scores))

# scores = cross_val_score(xgb_model, test_features, test_labels, scoring='roc_auc', cv=cv, n_jobs=-1)
# print("Mean ROC AUC for Test Set: %.5f" % mean(scores))


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


matrix_metrix(test_labels, y_pred, 0.4)


from sklearn.metrics import plot_roc_curve, plot_confusion_matrix
ax = plt.gca()
rfc_disp = plot_roc_curve(xgb_model, test_features, test_labels, ax=ax, alpha=0.8)
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
        test_features, 
        test_labels,
        display_labels=['No Donation','Donation'],
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
X_sampled = pd.DataFrame(test_features, columns=feature_names)
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_sampled)

# # summarize the effects of all the features
# shap.initjs()
shap.summary_plot(shap_values, X_sampled,plot_size=[8,5])


for i, col in enumerate(X_sampled.columns):
    print(col,i)
    shap.dependence_plot(col, shap_values, X_sampled, show=False)
plt.show()
