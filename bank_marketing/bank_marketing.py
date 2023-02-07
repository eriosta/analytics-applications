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

# zf = zipfile.ZipFile("bank.zip") 

# df = pd.read_csv(zf.open('bank-full.csv'), sep=';')

df = pd.read_csv('bank_marketing/bank-additional.csv', sep=';')

df.rename(columns={
    'emp.var.rate':'empvarrate',
    'cons.price.idx':'conspriceidx',
    'cons.conf.idx':'consconfidx',
    'euribor3m':'euribor3m',
    'nr.employed':'nremployed'
},inplace=True)

cols = list(df.drop(['y'],axis=1).columns)
groupby = ['y']

categorical = [
    'job','marital','education','default','loan','housing','contact','day_of_week','month','poutcome'
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

sum(df['y']==1)
import statsmodels.formula.api as smf
# maxiter = 35
model = smf.logit("y ~ age + C(job) + C(marital) + C(education) + C(housing) + C(loan) + C(contact) + C(day_of_week) + C(month) + campaign + pdays + previous + C(poutcome) + empvarrate + conspriceidx + consconfidx + euribor3m + nremployed", data = df).fit()

model.summary()
model_odds = pd.DataFrame(np.exp(model.params), columns= ['OR'])
model_odds['z-value']= model.pvalues
model_odds[['2.5%', '97.5%']] = np.exp(model.conf_int())
model_odds.sort_values(by='OR', ascending=False).to_csv('bank_marketing/logit.csv')

data = df

X = data.drop(['y','duration'],axis=1)
y = data['y'].to_numpy()

# sum(X.pdays == -1) # is this the 999?

X['pdays'] = np.where(X['pdays']==999,0,X['pdays'])

cats = [col for col in X if X[col].dtype == np.dtype('object')]

def transform_one_hot(data, features_to_encode):
    for feature in features_to_encode:
        one_hot = (pd.get_dummies(data[feature])).add_prefix(feature + '_')
        data = data.join(one_hot)
        data = data.drop(feature,axis=1)
    return data

X = transform_one_hot(X,cats)

feature_names =  list(X.columns)

X=X.to_numpy()

smote = SMOTE()
X, y = smote.fit_resample(X,y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

from sklearn.linear_model import LogisticRegression

weight = sum(y==1) / (2 * np.bincount(y))[0]

model = LogisticRegression(random_state=0, max_iter=1000).fit(X_train, y_train)

model.score(X, y)

y_pred = model.predict(X_test)

y_pred_prob = model.predict_proba(X_test)[:, 1]


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
rfc_disp = plot_roc_curve(model, X_test, y_test, ax=ax, alpha=0.8)
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
        model,
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
explainer = shap.LinearExplainer(model, X_sampled, feature_perturbation="interventional")
shap_values = explainer.shap_values(X_sampled)

# # summarize the effects of all the features
# shap.initjs()
shap.summary_plot(shap_values, X_sampled,plot_size=[8,5],feature_names=feature_names)

for i, col in enumerate(X_sampled.columns):
    print(col,i)
    shap.dependence_plot(col, shap_values, X_sampled, show=False,feature_names=feature_names)
plt.show()


