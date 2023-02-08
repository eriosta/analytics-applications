# Bank Marketing Case Study

## Background

## Goal
Predict if the client will subscribe (yes/no) to a term deposit (y)

## Methodology
### Variables
Identification, classification and operationalization of variables.

#### Bank client data
1. `age` : (numeric)
2. `job` : type of job (categorical: "admin.","blue-collar","entrepreneur","housemaid","management","retired","self-employed","services","student","technician","unemployed","unknown")
3. `marital` : marital status (categorical: "divorced","married","single","unknown"; note: "divorced" means divorced or widowed)
4. `education` : (categorical: "basic.4y","basic.6y","basic.9y","high.school","illiterate","professional.course","university.degree","unknown")
5. `default` : has credit in default? (categorical: "no","yes","unknown")
6. `housing` : has housing loan? (categorical: "no","yes","unknown")
7. `loan` : has personal loan? (categorical: "no","yes","unknown")
   
#### Data related with the last contact of the current campaign
8. `contact` : contact communication type (categorical: "cellular","telephone") 
9. `month` : last contact month of year (categorical: "jan", "feb", "mar", ..., "nov", "dec")
10. `day_of_week` : last contact day of the week (categorical: "mon","tue","wed","thu","fri")
11. `duration` : last contact duration, in seconds (numeric). Important note:  this attribute highly affects the output target (e.g., if duration=0 then y="no"). Yet, the duration is not known before a call is performed. Also, after the end of the call, `y` is obviously known. **Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.**
    
#### Other attributes
1.  `campaign` : number of contacts performed during this campaign and for this client (numeric, includes last contact)
2.  `pdays` : number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
3.  `previous` : number of contacts performed before this campaign and for this client (numeric)
4.  `poutcome` : outcome of the previous marketing campaign (categorical: "failure","nonexistent","success")
    
#### Social and economic context attributes
16. `emp.var.rate` : employment variation rate - quarterly indicator (numeric)
17. `cons.price.idx` : consumer price index - monthly indicator (numeric)     
18. `cons.conf.idx` : consumer confidence index - monthly indicator (numeric)     
19. `euribor3m` : euribor 3 month rate - daily indicator (numeric)
20. `nr.employed` : number of employees - quarterly indicator (numeric)

#### Output variable (desired target)
21. `y` : has the client subscribed a term deposit? (binary: "yes","no")


### Aims and hypotheses
Statements of hypotheses being tested and/or models being developed.
### Sampling
Sampling techniques, if full data is not being used.
### Data collection
Data collection process, including data sources, data size, etc. Primary/secondary?
### Data modeling
Modeling analysis/techniques used
### Assumptions and limitations
Methodological assumptions and limitations. 


---
```Data Import: The data was imported from a .csv file named "bank-additional.csv" using the Pandas library (pd.read_csv). The separator was ';'. The data was then renamed for some columns.

Summary Statistics: The summary statistics of the data were calculated using the TableOne package and stored in a fancy_grid format and .xlsx file.

Logistic Regression: The logistic regression was fitted using the Statsmodels formula API (smf.logit). The predictors used in the model are age, job, marital, education, housing, loan, contact, day_of_week, month, campaign, pdays, previous, poutcome, empvarrate, conspriceidx, consconfidx, euribor3m, and nremployed. The logistic regression results were saved in a .csv file named 'logit.csv'.

Data Transformation: The data was transformed by creating one-hot encoding for categorical variables. The function "transform_one_hot" was used for this purpose.

SMOTE (Synthetic Minority Over-sampling Technique): The data was oversampled using the SMOTE algorithm from the imblearn library to balance the classes in the target variable.

Train-Test Split: The transformed data was split into a training set (67%) and a test set (33%) using the train_test_split function from the sklearn library.

Logistic Regression Model: A logistic regression model was fitted on the training data using the LogisticRegression class from the sklearn library. The random state was set to 0, and the maximum iteration was set to 1000. The model's accuracy was calculated using the score method.

Model Prediction: The model was used to predict the target variable on the test data and stored in y_pred. The probabilities of the predictions were stored in y_pred_prob.

Model Evaluation: The model's performance was evaluated using the confusion matrix, which was calculated using the confusion_matrix function from the sklearn library.
```

---


## Data 

## Approach

## Results

## Next steps