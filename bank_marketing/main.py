import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from tableone import TableOne
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import math


def clean_bank_data(filepath):
    """
    Reads the bank-additional.csv file, renames its columns, and replaces the 999 value in the "pdays" column with 0.

    Parameters:
    filepath (str): The file path of the csv file.

    Returns:
    pandas.DataFrame: A cleaned DataFrame of the bank-additional data.
    """
    df = pd.read_csv(filepath, sep=';')

    df.rename(columns={
        'emp.var.rate':'empvarrate',
        'cons.price.idx':'conspriceidx',
        'cons.conf.idx':'consconfidx',
        'euribor3m':'euribor3m',
        'nr.employed':'nremployed'
    }, inplace=True)

    df['pdays'] = np.where(df['pdays']==999,0,df['pdays'])

    return df


def drop_unknown_rows(df):
    """
    Drops rows in the dataframe that contain the value "unknown".

    Parameters:
    df (pandas.DataFrame): The input dataframe

    Returns:
    pandas.DataFrame: The filtered dataframe with rows containing "unknown" removed.
    """
    df = df[df.applymap(lambda x: x != "unknown").all(1)]
    return df


def generate_summary_table(df, y_column, categorical_columns):
    """
    Generate summary statistics of the given DataFrame.

    Parameters:
    - df: DataFrame
        The input DataFrame.
    - y_column: str
        The target column.
    - categorical_columns: list
        List of categorical columns in the DataFrame.

    Returns:
    - mytable: TableOne object
        The summary statistics.
    """

    # Get the column names, excluding the target column
    cols = list(df.drop([y_column], axis=1).columns)
    groupby = [y_column]

    # Create the summary table
    mytable = TableOne(
        df,
        columns=cols,
        categorical=categorical_columns,
        groupby=groupby,
        pval=True
    )

    # Print the summary table in a grid format
    print(mytable.tabulate(tablefmt="fancy_grid"))

    # Save the summary table to an Excel file
    mytable.to_excel('summary.xlsx')
    mytable.to_html('summary.html')

    return mytable


def logit_regression(df, y, predictors):
    """
    Perform a logistic regression and return the odds ratios and other summary statistics.

    Parameters:
    df (pandas.DataFrame): DataFrame with the data to be used for regression
    y (str): The dependent variable in the regression
    predictors (str): The independent variables in the regression, formatted as a string for input into statsmodels

    Returns:
    pandas.DataFrame: DataFrame containing odds ratios, z-values, and confidence intervals for the regression variables
    """
    model = smf.logit(f"{y} ~ {predictors}", data = df).fit()
    model_odds = pd.DataFrame(np.exp(model.params), columns= ['OR'])
    model_odds['z-value']= model.pvalues
    model_odds[['2.5%', '97.5%']] = np.exp(model.conf_int())
    model_odds = model_odds.sort_values(by='OR', ascending=False)
    model_odds = model_odds[model_odds.index != 'Intercept']

    model_odds = round(model_odds,4)
    model_odds.to_excel("logit.xlsx")

    return model_odds

def transform_one_hot(data, features_to_encode):
    """
    Transform categorical features in a pandas DataFrame into one-hot encoded features.

    Parameters:
    data (pandas.DataFrame): The input DataFrame to be transformed.
    features_to_encode (list): A list of the names of the categorical features to be transformed.

    Returns:
    pandas.DataFrame: The transformed DataFrame with one-hot encoded features.
    """
    for feature in features_to_encode:
        one_hot = (pd.get_dummies(data[feature])).add_prefix(feature + '_')
        data = data.join(one_hot)
        data = data.drop(feature,axis=1)
    return data


def train_and_evaluate_logistic_regression(X_train, X_test, y_train, y_test):
    """
    Train a logistic regression model on input data and evaluate its performance.

    Parameters:
    X (numpy.ndarray or pandas.DataFrame): The feature matrix.
    y (numpy.ndarray or pandas.Series): The target vector.

    Returns:
    float: The accuracy score of the model on the test data.
    numpy.ndarray: The predicted target values of the model on the test data.
    numpy.ndarray: The predicted probabilities of the positive class for the test data.
    """

    model = LogisticRegression(random_state=0, max_iter=1000).fit(X_train, y_train)

    y_pred = model.predict(X_test)

    y_pred_prob = model.predict_proba(X_test)[:, 1]

    return model, y_pred, y_pred_prob


def matrix_metrics(real_values,pred_values,beta=0.4):
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

def visualize_model_metrics(model, X_test, y_test):
    """
    A function to visualize the performance of a machine learning model.
    
    Parameters
    ----------
    model : object
        A trained machine learning model.
    X_test : array-like
        A test data set.
    y_test : array-like
        True labels for the test data set.
        
    Returns
    -------
    None
        The function visualizes the ROC curve and confusion matrix of the model.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import plot_roc_curve, plot_confusion_matrix
    
    ax = plt.gca()
    rfc_disp = plot_roc_curve(model, X_test, y_test, ax=ax, alpha=0.8)
    plt.show()

    np.set_printoptions(precision=2)

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

def plot_shap_values(model, X_test, feature_names):
    """
    Plot SHAP values for a given model and test data.
    
    Parameters:
        model (object): a scikit-learn compatible model object
        X_test (pandas dataframe): test data with features
        feature_names (list): list of feature names
        
    Returns:
        None
        
    Side Effects:
        Creates a summary plot of SHAP values, dependence plots for each feature, and prints the index of each feature
    """
    import shap
    import matplotlib.pyplot as plt

    X_sampled = pd.DataFrame(X_test)
    explainer = shap.LinearExplainer(model, X_sampled, feature_perturbation="interventional")
    shap_values = explainer.shap_values(X_sampled)

    shap.summary_plot(shap_values, X_sampled,plot_size=[8,5],feature_names=feature_names)

    for i, col in enumerate(X_sampled.columns):
        print(col,i)
        shap.dependence_plot(col, shap_values, X_sampled, show=False,feature_names=feature_names)
    plt.show()
