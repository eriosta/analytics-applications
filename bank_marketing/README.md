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
SMOTE (Synthetic Minority Over-sampling Technique): The data was oversampled using the SMOTE algorithm from the imblearn library to balance the classes in the target variable.
### Data collection
Data collection process, including data sources, data size, etc. Primary/secondary?

### Data modeling
#### `stasmodels`
Logistic Regression: The logistic regression was fitted using the Statsmodels formula API (smf.logit). The predictors used in the model are age, job, marital, education, housing, loan, contact, day_of_week, month, campaign, pdays, previous, poutcome, empvarrate, conspriceidx, consconfidx, euribor3m, and nremployed. The logistic regression results were saved in a .csv file named 'logit.csv'.
#### `scikit-learn`
Train-Test Split: The transformed data was split into a training set (67%) and a test set (33%) using the train_test_split function from the sklearn library.

Logistic Regression Model: A logistic regression model was fitted on the training data using the LogisticRegression class from the sklearn library. The random state was set to 0, and the maximum iteration was set to 1000. The model's accuracy was calculated using the score method.

Model Prediction: The model was used to predict the target variable on the test data and stored in y_pred. The probabilities of the predictions were stored in y_pred_prob.

Model Evaluation: The model's performance was evaluated using the confusion matrix, which was calculated using the confusion_matrix function from the sklearn library.

[Ref](https://medium.com/@hsrinivasan2/linear-regression-in-scikit-learn-vs-statsmodels-568b60792991)

### Assumptions and limitations
Methodological assumptions and limitations. 

## Data
### Data cleaning
Data Import: The data was imported from a .csv file named `bank-additional.csv` using the Pandas library (pd.read_csv). The separator was `';'`. The data was then renamed for some columns.

### Data preprocessing

Data Transformation: The data was transformed by creating one-hot encoding for categorical variables. The function "transform_one_hot" was used for this purpose.

### Data limitations
[][][]

## Findings

### Summary statistics

<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th></th>
      <th colspan="5" halign="left">Client has subscribed to a term deposit?</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th>Missing</th>
      <th>Overall</th>
      <th>No</th>
      <th>Yes</th>
      <th>p-value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>n</th>
      <th></th>
      <td></td>
      <td>4119</td>
      <td>3668</td>
      <td>451</td>
      <td></td>
    </tr>
    <tr>
      <th>age, mean (SD)</th>
      <th></th>
      <td>0</td>
      <td>40.1 (10.3)</td>
      <td>39.9 (9.9)</td>
      <td>41.9 (13.3)</td>
      <td>0.002</td>
    </tr>
    <tr>
      <th rowspan="12" valign="top">job, n (%)</th>
      <th>admin.</th>
      <td>0</td>
      <td>1012 (24.6)</td>
      <td>879 (24.0)</td>
      <td>133 (29.5)</td>
      <td>&lt;0.001</td>
    </tr>
    <tr>
      <th>blue-collar</th>
      <td></td>
      <td>884 (21.5)</td>
      <td>823 (22.4)</td>
      <td>61 (13.5)</td>
      <td></td>
    </tr>
    <tr>
      <th>entrepreneur</th>
      <td></td>
      <td>148 (3.6)</td>
      <td>140 (3.8)</td>
      <td>8 (1.8)</td>
      <td></td>
    </tr>
    <tr>
      <th>housemaid</th>
      <td></td>
      <td>110 (2.7)</td>
      <td>99 (2.7)</td>
      <td>11 (2.4)</td>
      <td></td>
    </tr>
    <tr>
      <th>management</th>
      <td></td>
      <td>324 (7.9)</td>
      <td>294 (8.0)</td>
      <td>30 (6.7)</td>
      <td></td>
    </tr>
    <tr>
      <th>retired</th>
      <td></td>
      <td>166 (4.0)</td>
      <td>128 (3.5)</td>
      <td>38 (8.4)</td>
      <td></td>
    </tr>
    <tr>
      <th>self-employed</th>
      <td></td>
      <td>159 (3.9)</td>
      <td>146 (4.0)</td>
      <td>13 (2.9)</td>
      <td></td>
    </tr>
    <tr>
      <th>services</th>
      <td></td>
      <td>393 (9.5)</td>
      <td>358 (9.8)</td>
      <td>35 (7.8)</td>
      <td></td>
    </tr>
    <tr>
      <th>student</th>
      <td></td>
      <td>82 (2.0)</td>
      <td>63 (1.7)</td>
      <td>19 (4.2)</td>
      <td></td>
    </tr>
    <tr>
      <th>technician</th>
      <td></td>
      <td>691 (16.8)</td>
      <td>611 (16.7)</td>
      <td>80 (17.7)</td>
      <td></td>
    </tr>
    <tr>
      <th>unemployed</th>
      <td></td>
      <td>111 (2.7)</td>
      <td>92 (2.5)</td>
      <td>19 (4.2)</td>
      <td></td>
    </tr>
    <tr>
      <th>unknown</th>
      <td></td>
      <td>39 (0.9)</td>
      <td>35 (1.0)</td>
      <td>4 (0.9)</td>
      <td></td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">marital, n (%)</th>
      <th>divorced</th>
      <td>0</td>
      <td>446 (10.8)</td>
      <td>403 (11.0)</td>
      <td>43 (9.5)</td>
      <td>0.016</td>
    </tr>
    <tr>
      <th>married</th>
      <td></td>
      <td>2509 (60.9)</td>
      <td>2257 (61.5)</td>
      <td>252 (55.9)</td>
      <td></td>
    </tr>
    <tr>
      <th>single</th>
      <td></td>
      <td>1153 (28.0)</td>
      <td>998 (27.2)</td>
      <td>155 (34.4)</td>
      <td></td>
    </tr>
    <tr>
      <th>unknown</th>
      <td></td>
      <td>11 (0.3)</td>
      <td>10 (0.3)</td>
      <td>1 (0.2)</td>
      <td></td>
    </tr>
    <tr>
      <th rowspan="8" valign="top">education, n (%)</th>
      <th>basic.4y</th>
      <td>0</td>
      <td>429 (10.4)</td>
      <td>391 (10.7)</td>
      <td>38 (8.4)</td>
      <td>0.002</td>
    </tr>
    <tr>
      <th>basic.6y</th>
      <td></td>
      <td>228 (5.5)</td>
      <td>211 (5.8)</td>
      <td>17 (3.8)</td>
      <td></td>
    </tr>
    <tr>
      <th>basic.9y</th>
      <td></td>
      <td>574 (13.9)</td>
      <td>531 (14.5)</td>
      <td>43 (9.5)</td>
      <td></td>
    </tr>
    <tr>
      <th>high.school</th>
      <td></td>
      <td>921 (22.4)</td>
      <td>824 (22.5)</td>
      <td>97 (21.5)</td>
      <td></td>
    </tr>
    <tr>
      <th>illiterate</th>
      <td></td>
      <td>1 (0.0)</td>
      <td>1 (0.0)</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>professional.course</th>
      <td></td>
      <td>535 (13.0)</td>
      <td>470 (12.8)</td>
      <td>65 (14.4)</td>
      <td></td>
    </tr>
    <tr>
      <th>university.degree</th>
      <td></td>
      <td>1264 (30.7)</td>
      <td>1099 (30.0)</td>
      <td>165 (36.6)</td>
      <td></td>
    </tr>
    <tr>
      <th>unknown</th>
      <td></td>
      <td>167 (4.1)</td>
      <td>141 (3.8)</td>
      <td>26 (5.8)</td>
      <td></td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">default, n (%)</th>
      <th>no</th>
      <td>0</td>
      <td>3315 (80.5)</td>
      <td>2913 (79.4)</td>
      <td>402 (89.1)</td>
      <td>&lt;0.001</td>
    </tr>
    <tr>
      <th>unknown</th>
      <td></td>
      <td>803 (19.5)</td>
      <td>754 (20.6)</td>
      <td>49 (10.9)</td>
      <td></td>
    </tr>
    <tr>
      <th>yes</th>
      <td></td>
      <td>1 (0.0)</td>
      <td>1 (0.0)</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">housing, n (%)</th>
      <th>no</th>
      <td>0</td>
      <td>1839 (44.6)</td>
      <td>1637 (44.6)</td>
      <td>202 (44.8)</td>
      <td>0.731</td>
    </tr>
    <tr>
      <th>unknown</th>
      <td></td>
      <td>105 (2.5)</td>
      <td>96 (2.6)</td>
      <td>9 (2.0)</td>
      <td></td>
    </tr>
    <tr>
      <th>yes</th>
      <td></td>
      <td>2175 (52.8)</td>
      <td>1935 (52.8)</td>
      <td>240 (53.2)</td>
      <td></td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">loan, n (%)</th>
      <th>no</th>
      <td>0</td>
      <td>3349 (81.3)</td>
      <td>2975 (81.1)</td>
      <td>374 (82.9)</td>
      <td>0.568</td>
    </tr>
    <tr>
      <th>unknown</th>
      <td></td>
      <td>105 (2.5)</td>
      <td>96 (2.6)</td>
      <td>9 (2.0)</td>
      <td></td>
    </tr>
    <tr>
      <th>yes</th>
      <td></td>
      <td>665 (16.1)</td>
      <td>597 (16.3)</td>
      <td>68 (15.1)</td>
      <td></td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">contact, n (%)</th>
      <th>cellular</th>
      <td>0</td>
      <td>2652 (64.4)</td>
      <td>2277 (62.1)</td>
      <td>375 (83.1)</td>
      <td>&lt;0.001</td>
    </tr>
    <tr>
      <th>telephone</th>
      <td></td>
      <td>1467 (35.6)</td>
      <td>1391 (37.9)</td>
      <td>76 (16.9)</td>
      <td></td>
    </tr>
    <tr>
      <th rowspan="10" valign="top">month, n (%)</th>
      <th>apr</th>
      <td>0</td>
      <td>215 (5.2)</td>
      <td>179 (4.9)</td>
      <td>36 (8.0)</td>
      <td>&lt;0.001</td>
    </tr>
    <tr>
      <th>aug</th>
      <td></td>
      <td>636 (15.4)</td>
      <td>572 (15.6)</td>
      <td>64 (14.2)</td>
      <td></td>
    </tr>
    <tr>
      <th>dec</th>
      <td></td>
      <td>22 (0.5)</td>
      <td>10 (0.3)</td>
      <td>12 (2.7)</td>
      <td></td>
    </tr>
    <tr>
      <th>jul</th>
      <td></td>
      <td>711 (17.3)</td>
      <td>652 (17.8)</td>
      <td>59 (13.1)</td>
      <td></td>
    </tr>
    <tr>
      <th>jun</th>
      <td></td>
      <td>530 (12.9)</td>
      <td>462 (12.6)</td>
      <td>68 (15.1)</td>
      <td></td>
    </tr>
    <tr>
      <th>mar</th>
      <td></td>
      <td>48 (1.2)</td>
      <td>20 (0.5)</td>
      <td>28 (6.2)</td>
      <td></td>
    </tr>
    <tr>
      <th>may</th>
      <td></td>
      <td>1378 (33.5)</td>
      <td>1288 (35.1)</td>
      <td>90 (20.0)</td>
      <td></td>
    </tr>
    <tr>
      <th>nov</th>
      <td></td>
      <td>446 (10.8)</td>
      <td>403 (11.0)</td>
      <td>43 (9.5)</td>
      <td></td>
    </tr>
    <tr>
      <th>oct</th>
      <td></td>
      <td>69 (1.7)</td>
      <td>44 (1.2)</td>
      <td>25 (5.5)</td>
      <td></td>
    </tr>
    <tr>
      <th>sep</th>
      <td></td>
      <td>64 (1.6)</td>
      <td>38 (1.0)</td>
      <td>26 (5.8)</td>
      <td></td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">day_of_week, n (%)</th>
      <th>fri</th>
      <td>0</td>
      <td>768 (18.6)</td>
      <td>685 (18.7)</td>
      <td>83 (18.4)</td>
      <td>0.972</td>
    </tr>
    <tr>
      <th>mon</th>
      <td></td>
      <td>855 (20.8)</td>
      <td>757 (20.6)</td>
      <td>98 (21.7)</td>
      <td></td>
    </tr>
    <tr>
      <th>thu</th>
      <td></td>
      <td>860 (20.9)</td>
      <td>764 (20.8)</td>
      <td>96 (21.3)</td>
      <td></td>
    </tr>
    <tr>
      <th>tue</th>
      <td></td>
      <td>841 (20.4)</td>
      <td>750 (20.4)</td>
      <td>91 (20.2)</td>
      <td></td>
    </tr>
    <tr>
      <th>wed</th>
      <td></td>
      <td>795 (19.3)</td>
      <td>712 (19.4)</td>
      <td>83 (18.4)</td>
      <td></td>
    </tr>
    <tr>
      <th>duration, mean (SD)</th>
      <th></th>
      <td>0</td>
      <td>256.8 (254.7)</td>
      <td>219.4 (198.3)</td>
      <td>560.8 (411.5)</td>
      <td>&lt;0.001</td>
    </tr>
    <tr>
      <th>campaign, mean (SD)</th>
      <th></th>
      <td>0</td>
      <td>2.5 (2.6)</td>
      <td>2.6 (2.7)</td>
      <td>2.0 (1.4)</td>
      <td>&lt;0.001</td>
    </tr>
    <tr>
      <th>pdays, mean (SD)</th>
      <th></th>
      <td>0</td>
      <td>960.4 (191.9)</td>
      <td>982.8 (125.9)</td>
      <td>778.7 (413.2)</td>
      <td>&lt;0.001</td>
    </tr>
    <tr>
      <th>previous, mean (SD)</th>
      <th></th>
      <td>0</td>
      <td>0.2 (0.5)</td>
      <td>0.1 (0.4)</td>
      <td>0.6 (1.0)</td>
      <td>&lt;0.001</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">poutcome, n (%)</th>
      <th>failure</th>
      <td>0</td>
      <td>454 (11.0)</td>
      <td>387 (10.6)</td>
      <td>67 (14.9)</td>
      <td>&lt;0.001</td>
    </tr>
    <tr>
      <th>nonexistent</th>
      <td></td>
      <td>3523 (85.5)</td>
      <td>3231 (88.1)</td>
      <td>292 (64.7)</td>
      <td></td>
    </tr>
    <tr>
      <th>success</th>
      <td></td>
      <td>142 (3.4)</td>
      <td>50 (1.4)</td>
      <td>92 (20.4)</td>
      <td></td>
    </tr>
    <tr>
      <th>empvarrate, mean (SD)</th>
      <th></th>
      <td>0</td>
      <td>0.1 (1.6)</td>
      <td>0.2 (1.5)</td>
      <td>-1.2 (1.6)</td>
      <td>&lt;0.001</td>
    </tr>
    <tr>
      <th>conspriceidx, mean (SD)</th>
      <th></th>
      <td>0</td>
      <td>93.6 (0.6)</td>
      <td>93.6 (0.6)</td>
      <td>93.4 (0.7)</td>
      <td>&lt;0.001</td>
    </tr>
    <tr>
      <th>consconfidx, mean (SD)</th>
      <th></th>
      <td>0</td>
      <td>-40.5 (4.6)</td>
      <td>-40.6 (4.4)</td>
      <td>-39.8 (5.9)</td>
      <td>0.006</td>
    </tr>
    <tr>
      <th>euribor3m, mean (SD)</th>
      <th></th>
      <td>0</td>
      <td>3.6 (1.7)</td>
      <td>3.8 (1.6)</td>
      <td>2.1 (1.8)</td>
      <td>&lt;0.001</td>
    </tr>
    <tr>
      <th>nremployed, mean (SD)</th>
      <th></th>
      <td>0</td>
      <td>5166.5 (73.7)</td>
      <td>5175.5 (65.9)</td>
      <td>5093.1 (90.6)</td>
      <td>&lt;0.001</td>
    </tr>
  </tbody>
</table>

### Odds ratios

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>OR</th>
      <th>z-value</th>
      <th>2.5%</th>
      <th>97.5%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>C(month)[T.mar]</th>
      <td>6.316210e+00</td>
      <td>0.000043</td>
      <td>2.613317e+00</td>
      <td>1.526585e+01</td>
    </tr>
    <tr>
      <th>conspriceidx</th>
      <td>5.393726e+00</td>
      <td>0.015346</td>
      <td>1.380805e+00</td>
      <td>2.106908e+01</td>
    </tr>
    <tr>
      <th>C(poutcome)[T.success]</th>
      <td>3.673955e+00</td>
      <td>0.024471</td>
      <td>1.182418e+00</td>
      <td>1.141555e+01</td>
    </tr>
    <tr>
      <th>C(month)[T.dec]</th>
      <td>2.411551e+00</td>
      <td>0.131652</td>
      <td>7.678846e-01</td>
      <td>7.573507e+00</td>
    </tr>
    <tr>
      <th>C(poutcome)[T.nonexistent]</th>
      <td>1.897657e+00</td>
      <td>0.017504</td>
      <td>1.118692e+00</td>
      <td>3.219029e+00</td>
    </tr>
    <tr>
      <th>C(marital)[T.single]</th>
      <td>1.327609e+00</td>
      <td>0.219904</td>
      <td>8.442049e-01</td>
      <td>2.087818e+00</td>
    </tr>
    <tr>
      <th>C(education)[T.basic.6y]</th>
      <td>1.277890e+00</td>
      <td>0.471070</td>
      <td>6.559893e-01</td>
      <td>2.489376e+00</td>
    </tr>
    <tr>
      <th>C(education)[T.professional.course]</th>
      <td>1.242433e+00</td>
      <td>0.443853</td>
      <td>7.127930e-01</td>
      <td>2.165621e+00</td>
    </tr>
    <tr>
      <th>C(education)[T.university.degree]</th>
      <td>1.218997e+00</td>
      <td>0.449504</td>
      <td>7.296403e-01</td>
      <td>2.036555e+00</td>
    </tr>
    <tr>
      <th>C(education)[T.unknown]</th>
      <td>1.215929e+00</td>
      <td>0.570510</td>
      <td>6.188128e-01</td>
      <td>2.389224e+00</td>
    </tr>
    <tr>
      <th>C(marital)[T.married]</th>
      <td>1.188594e+00</td>
      <td>0.391503</td>
      <td>8.005897e-01</td>
      <td>1.764645e+00</td>
    </tr>
    <tr>
      <th>C(day_of_week)[T.wed]</th>
      <td>1.172725e+00</td>
      <td>0.396528</td>
      <td>8.113983e-01</td>
      <td>1.694955e+00</td>
    </tr>
    <tr>
      <th>previous</th>
      <td>1.156334e+00</td>
      <td>0.369697</td>
      <td>8.418819e-01</td>
      <td>1.588238e+00</td>
    </tr>
    <tr>
      <th>C(education)[T.basic.9y]</th>
      <td>1.146696e+00</td>
      <td>0.614616</td>
      <td>6.730284e-01</td>
      <td>1.953723e+00</td>
    </tr>
    <tr>
      <th>C(education)[T.high.school]</th>
      <td>1.128289e+00</td>
      <td>0.643839</td>
      <td>6.763917e-01</td>
      <td>1.882097e+00</td>
    </tr>
    <tr>
      <th>C(job)[T.unemployed]</th>
      <td>1.091667e+00</td>
      <td>0.792854</td>
      <td>5.672862e-01</td>
      <td>2.100770e+00</td>
    </tr>
    <tr>
      <th>C(marital)[T.unknown]</th>
      <td>1.087086e+00</td>
      <td>0.943721</td>
      <td>1.070109e-01</td>
      <td>1.104333e+01</td>
    </tr>
    <tr>
      <th>consconfidx</th>
      <td>1.054225e+00</td>
      <td>0.022080</td>
      <td>1.007619e+00</td>
      <td>1.102987e+00</td>
    </tr>
    <tr>
      <th>euribor3m</th>
      <td>1.034505e+00</td>
      <td>0.925260</td>
      <td>5.092478e-01</td>
      <td>2.101531e+00</td>
    </tr>
    <tr>
      <th>age</th>
      <td>1.016063e+00</td>
      <td>0.019670</td>
      <td>1.002549e+00</td>
      <td>1.029759e+00</td>
    </tr>
    <tr>
      <th>C(day_of_week)[T.thu]</th>
      <td>1.010968e+00</td>
      <td>0.952572</td>
      <td>7.057056e-01</td>
      <td>1.448276e+00</td>
    </tr>
    <tr>
      <th>nremployed</th>
      <td>1.004316e+00</td>
      <td>0.612756</td>
      <td>9.877071e-01</td>
      <td>1.021203e+00</td>
    </tr>
    <tr>
      <th>pdays</th>
      <td>9.995361e-01</td>
      <td>0.426181</td>
      <td>9.983945e-01</td>
      <td>1.000679e+00</td>
    </tr>
    <tr>
      <th>C(day_of_week)[T.tue]</th>
      <td>9.809987e-01</td>
      <td>0.918352</td>
      <td>6.797838e-01</td>
      <td>1.415683e+00</td>
    </tr>
    <tr>
      <th>C(job)[T.student]</th>
      <td>9.772091e-01</td>
      <td>0.947461</td>
      <td>4.922430e-01</td>
      <td>1.939972e+00</td>
    </tr>
    <tr>
      <th>C(month)[T.aug]</th>
      <td>9.767738e-01</td>
      <td>0.947607</td>
      <td>4.846033e-01</td>
      <td>1.968800e+00</td>
    </tr>
    <tr>
      <th>C(day_of_week)[T.mon]</th>
      <td>9.686869e-01</td>
      <td>0.861418</td>
      <td>6.777331e-01</td>
      <td>1.384548e+00</td>
    </tr>
    <tr>
      <th>C(month)[T.jun]</th>
      <td>9.662659e-01</td>
      <td>0.926150</td>
      <td>4.676933e-01</td>
      <td>1.996329e+00</td>
    </tr>
    <tr>
      <th>C(job)[T.technician]</th>
      <td>9.582852e-01</td>
      <td>0.824361</td>
      <td>6.577695e-01</td>
      <td>1.396098e+00</td>
    </tr>
    <tr>
      <th>C(month)[T.sep]</th>
      <td>9.390376e-01</td>
      <td>0.902637</td>
      <td>3.427790e-01</td>
      <td>2.572479e+00</td>
    </tr>
    <tr>
      <th>C(loan)[T.yes]</th>
      <td>9.280807e-01</td>
      <td>0.639360</td>
      <td>6.792165e-01</td>
      <td>1.268128e+00</td>
    </tr>
    <tr>
      <th>campaign</th>
      <td>9.260809e-01</td>
      <td>0.024783</td>
      <td>8.660225e-01</td>
      <td>9.903043e-01</td>
    </tr>
    <tr>
      <th>C(month)[T.jul]</th>
      <td>9.149949e-01</td>
      <td>0.767768</td>
      <td>5.073913e-01</td>
      <td>1.650039e+00</td>
    </tr>
    <tr>
      <th>C(housing)[T.yes]</th>
      <td>9.038962e-01</td>
      <td>0.388999</td>
      <td>7.182541e-01</td>
      <td>1.137520e+00</td>
    </tr>
    <tr>
      <th>C(month)[T.oct]</th>
      <td>8.438026e-01</td>
      <td>0.704985</td>
      <td>3.502665e-01</td>
      <td>2.032746e+00</td>
    </tr>
    <tr>
      <th>C(loan)[T.unknown]</th>
      <td>8.402975e-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>C(housing)[T.unknown]</th>
      <td>8.402974e-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>C(job)[T.housemaid]</th>
      <td>8.288149e-01</td>
      <td>0.636941</td>
      <td>3.800500e-01</td>
      <td>1.807484e+00</td>
    </tr>
    <tr>
      <th>C(job)[T.services]</th>
      <td>8.165901e-01</td>
      <td>0.396468</td>
      <td>5.112204e-01</td>
      <td>1.304368e+00</td>
    </tr>
    <tr>
      <th>C(job)[T.retired]</th>
      <td>7.514464e-01</td>
      <td>0.346544</td>
      <td>4.144748e-01</td>
      <td>1.362379e+00</td>
    </tr>
    <tr>
      <th>C(month)[T.may]</th>
      <td>7.338506e-01</td>
      <td>0.220240</td>
      <td>4.474452e-01</td>
      <td>1.203581e+00</td>
    </tr>
    <tr>
      <th>C(job)[T.blue-collar]</th>
      <td>7.274123e-01</td>
      <td>0.157035</td>
      <td>4.681053e-01</td>
      <td>1.130362e+00</td>
    </tr>
    <tr>
      <th>C(job)[T.management]</th>
      <td>6.179615e-01</td>
      <td>0.054195</td>
      <td>3.785788e-01</td>
      <td>1.008710e+00</td>
    </tr>
    <tr>
      <th>C(job)[T.unknown]</th>
      <td>5.970944e-01</td>
      <td>0.419808</td>
      <td>1.705895e-01</td>
      <td>2.089939e+00</td>
    </tr>
    <tr>
      <th>C(month)[T.nov]</th>
      <td>5.909971e-01</td>
      <td>0.134749</td>
      <td>2.966591e-01</td>
      <td>1.177370e+00</td>
    </tr>
    <tr>
      <th>C(job)[T.entrepreneur]</th>
      <td>5.781740e-01</td>
      <td>0.169276</td>
      <td>2.647128e-01</td>
      <td>1.262822e+00</td>
    </tr>
    <tr>
      <th>C(job)[T.self-employed]</th>
      <td>5.577747e-01</td>
      <td>0.092307</td>
      <td>2.826542e-01</td>
      <td>1.100683e+00</td>
    </tr>
    <tr>
      <th>C(contact)[T.telephone]</th>
      <td>3.576971e-01</td>
      <td>0.000014</td>
      <td>2.249851e-01</td>
      <td>5.686920e-01</td>
    </tr>
    <tr>
      <th>empvarrate</th>
      <td>3.561491e-01</td>
      <td>0.009215</td>
      <td>1.637397e-01</td>
      <td>7.746571e-01</td>
    </tr>
    <tr>
      <th>C(education)[T.illiterate]</th>
      <td>5.011785e-59</td>
      <td>1.000000</td>
      <td>0.000000e+00</td>
      <td>inf</td>
    </tr>
  </tbody>
</table>

### Results presented in tables or charts when appropriate
### Results reported with respect to hypotheses/models.
### Factual information kept separate from interpretation, inference and evaluation.

## Conclusions and Recommendations
Discuss alternative methodologies






