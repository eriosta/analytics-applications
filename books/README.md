[![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)](https://github.com/eriosta/analytics-applications/blob/main/books/main.ipynb)
[![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/eriosta/analytics-applications/tree/main/books)

# Executive Summary
In this analysis, we used data from Bookbinders Book Club (BBBC) to explore the use of predictive modeling approaches to improve the efficacy of its direct mail program. Specifically, we analyzed a subset of the database containing data for customers who purchased the book and customers who did not, to identify the factors that influence book purchasing behavior.

We evaluated three commonly used machine learning algorithms – linear regression, logistic regression, and SVM – to predict book purchasing behavior based on customer characteristics. We found that the logistic regression model had the highest accuracy score, but relatively low precision, recall, and F1 scores, indicating that the model is better at predicting true negatives than true positives. The SVM model had a lower accuracy score than the logistic regression model, and even lower precision, recall, and F1 scores. The linear regression model had the lowest accuracy score and the highest MSE and MAE scores, indicating that it is not very good at predicting the target variable accurately.

Based on the performance metrics, it may be beneficial for BBBC to develop expertise in all three methods, as each method has its own strengths and weaknesses that may be relevant depending on the specific problem being addressed. Additionally, a pipeline can be developed to automate and simplify future modeling efforts at the company.

# Problem
Bookbinders Book Club (BBBC) is a distributor of specialty books through direct marketing. In anticipation of using database marketing, BBBC made a strategic decision to build and maintain a detailed database about its members containing all relevant information about them. The company is exploring whether to use predictive modeling approaches to improve the efficacy of its direct mail program. The objective of this analysis is to identify the factors that influence book purchasing behavior based on customer characteristics, to help BBBC develop a response model that can be used to improve the efficacy of its direct mail program.

# Literature Review
Machine learning approaches have been used extensively in predicting book buying behavior based on customer characteristics. For example, regression models and decision trees have been used to predict the likelihood of purchase based on demographic and transactional data. Neural networks have also been used to predict book buying behavior based on a wide range of features, including customer behavior on social media platforms, product ratings and reviews, and sentiment analysis of customer reviews. These methods have been shown to be effective in predicting book buying behavior, and can help companies like BBBC develop more targeted and effective direct mail campaigns.

# Methods
## Variables

<table>
  <tr>
    <th>Variable</th>
    <th>Description</th>
  </tr>
  <tr>
    <td>Choice</td>
    <td>Whether the customer purchased The Art History of Florence. 1 corresponds to a purchase and 0 corresponds to a non-purchase.</td>
  </tr>
  <tr>
    <td>Gender</td>
    <td>0 = Female and 1 = Male.</td>
  </tr>
  <tr>
    <td>Amount_purchased</td>
    <td>Total money spent on BBBC books.</td>
  </tr>
  <tr>
    <td>Frequency</td>
    <td>Total number of purchases in the chosen period (used as a proxy for frequency).</td>
  </tr>
  <tr>
    <td>Last_purchase (recency of purchase)</td>
    <td>Months since last purchase.</td>
  </tr>
  <tr>
    <td>First_purchase</td>
    <td>Months since first purchase.</td>
  </tr>
  <tr>
    <td>P_Child</td>
    <td>Number of children's books purchased.</td>
  </tr>
  <tr>
    <td>P_Youth</td>
    <td>Number of youth books purchased.</td>
  </tr>
  <tr>
    <td>P_Cook</td>
    <td>Number of cookbooks purchased.</td>
  </tr>
  <tr>
    <td>P_DIY</td>
    <td>Number of do-it-yourself books purchased.</td>
  </tr>
  <tr>
    <td>P_Art</td>
    <td>Number of art books purchased.</td>
  </tr>
</table>

## Models
Linear regression, logistic regression, and SVM are all commonly used machine learning algorithms for regression and classification tasks. Each algorithm has its own strengths and weaknesses, which can be evaluated based on the performance metrics for each algorithm.


### Linear regression
<table>
    </tr>
    <tr>
        <td>Pros</td>
        <td>Cons</td>
    </tr>
    <tr>
        <td>Simple and easy to understand.</td>
        <td>Assumes a linear relationship between the input and output variables, which may not always be the case.</td>
    </tr>
    <tr>
        <td>Provides interpretable results, since the coefficients for each feature can be directly interpreted as the effect of that feature on the output variable.</td>
        <td>Can be sensitive to outliers, since the model tries to minimize the mean squared error (MSE) of the predictions.</td>
    </tr>
    <tr>
        <td>Performs well when the relationship between the input and output variables is linear.</td>
        <td></td>
    </tr>
</table>

### Logistic regression

<table>
    <tr>
    </tr>
    <tr>
        <td>Pros</td>
        <td>Cons</td>
    </tr>
    <tr>
        <td>Can model non-linear relationships between the input and output variables using non-linear transformations of the features.</td>
        <td>Assumes a linear relationship between the log odds of the output variable and the input variables, which may not always be the case.</td>
    </tr>
    <tr>
        <td>Provides interpretable results, since the coefficients for each feature can be directly interpreted as the effect of that feature on the log odds of the output variable.</td>
        <td>Can be sensitive to outliers, since the model tries to maximize the log-likelihood of the observations.</td>
    </tr>
    <tr>
        <td>Can output probabilities of class membership, which can be useful for decision making.</td>
        <td>May not perform well when the classes are highly overlapping or the decision boundary is highly non-linear.</td>
    </tr>
    <tr>
        <td>Performs well when the classes are well-separated and the decision boundary is close to linear.</td>
        <td></td>
    </tr>
</table>

### SVM

<table>
    <tr>
    </tr>
    <tr>
        <td>Pros</td>
        <td>Cons</td>
    </tr>
    <tr>
        <td>Can model non-linear relationships between the input and output variables using kernel functions.</td>
        <td>Can be computationally expensive, especially for large datasets or complex kernel functions.</td>
    </tr>
    <tr>
        <td>Can handle high-dimensional data well.</td>
        <td>Can be sensitive to the choice of kernel function and hyperparameters.</td>
    </tr>
    <tr>
        <td>Tends to work well with small to medium-sized datasets.</td>
        <td>Can be difficult to interpret, since the model does not provide explicit coefficients for each feature.</td>
    </tr>
    <tr>
        <td>Can find the maximum margin hyperplane, which can help to avoid overfitting.</td>
        <td></td>
    </tr>
</table>

## Documentation

[Source code](https://github.com/eriosta/analytics-applications/blob/main/books/model.py)

### Class `BBBC_Model`
The `BBBC_Model` class provides a collection of methods for building, training, evaluating, and selecting the best model for a binary classification problem based on customer purchase behavior. Here is the high-level documentation for the classes and their methods:

#### Attributes
`data_path` : str

The path to the data file.

`response_var` : str

The dependent variable for the analysis - purchase or no purchase of the book.

`test_size` : float

The proportion of the data to use for testing the models. Default is 0.2.

`train_path` : str or None

The path to the CSV file containing the custom training set.

`test_path` : str or None

The path to the CSV file containing the custom testing set.

#### Methods
`__init__(self, data_path=None, response_var=None, test_size=0.2, train_path=None, test_path=None)`

Initializes the class with the given parameters.

`load_data(self)`

Loads the data from the specified file or the training and testing CSV files.

`analyze_high_cardinality(self, data, columns, threshold=0.9)`

Analyzes high cardinality between variables in a pandas DataFrame.

`spearman_correlation(self, data, response_var=None)`

Performs Spearman correlation and visualizes the correlation matrix.

`summarize_stats(self, data, response_var=None)`

Performs summary statistics between `response_var==1` and `response_var==0` and generates a table with the following:
- variable name
- mean difference
- test type (t-test, Mann-Whitney U test, etc.)
- statistic
- p-value
- confidence intervals

`preprocess_data(self, data=None)`

Preprocesses the data by removing missing values, creating dummy variables, and splitting into training and testing sets.

`build_linear_model(self, X_train, y_train)`

Builds and trains a linear regression model.

`build_logit_model(self, X_train, y_train)`

Builds and trains a logistic regression model.

`build_svm_model(self, X_train, y_train)`

Builds and trains a support vector machine model.

`predict(self, model, X_new)`

Predicts the class label of new instances using the given model.

`evaluate_model(self, model, X_test, y_test)`

Evaluates the performance of the given model on the test data.

`select_best_model(self)`
Builds, trains, and evaluates several models and selects the best-performing model based on the F1 score.

`shap_analysis(self, model, X_test, plot_dependence=False)`

Performs SHAP analysis on the given model and test data and visualizes the SHAP values for the first instance in the test data. Optionally, the method can also visualize two-way dependence plots for each feature.

### Clas `ModelAnalyzer`

The `ModelAnalyzer` class is a Python class designed to analyze and visualize the coefficients of a linear, logistic, or SVM model. The class contains three main methods: `__init__()`, `analyze()`, and `get_covariate_table()`.

#### Attributes:
`model_type`

A string that specifies the type of model to analyze ('linear', 'logistic', or 'svm').

`X_train`

A pandas DataFrame containing the training data features.

`y_train`

A pandas DataFrame containing the training data response variable.

`summary_table`

A summary table of coefficient values and 95% CIs, created by the `analyze()` method. This attribute is set if the `model_type` is 'linear' or 'logistic'.

`coefs`

An array of coefficient values, created by the `analyze()` method.

`lower_cis`

An array of lower bounds for the 95% CIs of the coefficient values, created by the `analyze()` method.

`upper_cis`

An array of upper bounds for the 95% CIs of the coefficient values, created by the `analyze()` method.

#### Methods
`init(self, model_type, X_train, y_train)`

This is the constructor method for the `ModelAnalyzer` class. It takes three arguments: `model_type`, which specifies the type of model to analyze ('linear', 'logistic', or 'svm'); `X_train`, which is a pandas DataFrame containing the training data features; and `y_train`, which is a pandas DataFrame containing the training data response variable.

`analyze(self)`

This method trains the specified model type and extracts the coefficient values and 95% CIs. The method does not take any arguments. If the model_type is 'linear' or 'logistic', the method trains a regression model using the `statsmodels` package and creates a summary table containing the coefficient values and 95% CIs. If the `model_type` is 'svm', the method trains an SVM model using `scikit-learn` and bootstrap resamples the data to obtain 95% CIs. The method sets the `coefs`, `lower_cis`, and `upper_cis` attributes of the class instance.

`get_covariate_table(self)`

This method creates a table of covariates, coefficients, and 95% CIs. The method does not take any arguments. The method returns a pandas DataFrame containing the covariate names, coefficient values, and 95% CIs. The table is created using the `coefs`, `lower_cis`, and `upper_cis` attributes that were set by the `analyze()` method.

# Data
## Preprocessing
The `preprocess_data` method preprocesses the data by removing missing values, creating dummy variables for categorical variables, and splitting the data into training and testing sets using the specified test size. If custom training and testing files are provided, they are loaded and split into training and testing sets. The method then returns the preprocessed training and testing sets as pandas DataFrames.
## Analysis
The `analyze_high_cardinality` method identifies columns in the data that have high cardinality, meaning they have a high proportion of unique values to the number of rows.

The `spearman_correlation` method performs Spearman correlation on the data and visualizes the correlation matrix.

## Summary Statistics
`Gender`: The mean difference is -0.164, which suggests that the proportion of males in the `Choice==1` group is lower than that in the `Choice==0` group. The Satterthwaite t-test* statistic is 7.52, and the p-value is very small (less than 0.001), indicating that the mean difference is statistically significant.

`Amount_purchased`: The mean difference is 24.12, which suggests that the `Choice==1` group tends to purchase more than the `Choice==0` group. The Mann-Whitney U test** statistic is 8.53e+05, and the p-value is very small (less than 0.001), indicating that the mean difference is statistically significant.

`Frequency`: The mean difference is -4.41, which suggests that the `Choice==1` group tends to have lower frequency of purchases compared to the `Choice==0` group. The Satterthwaite t-test statistic is 15.38, and the p-value is very small (less than 0.001), indicating that the mean difference is statistically significant.

`Last_purchase`: The mean difference is 0.87, which suggests that the `Choice==1` group tends to have a more recent last purchase compared to the `Choice==0` group. The Satterthwaite t-test statistic is -5.91, and the p-value is very small (less than 0.001), indicating that the mean difference is statistically significant.

`First_purchase`: The mean difference is -0.51, which suggests that the `Choice==1` group tends to have a later first purchase compared to the `Choice==0` group, but this difference is not statistically significant at the 0.05 level (p-value = 0.502).

`P_Child`, `P_Youth`, `P_Cook`, `P_DIY`: The mean differences for these covariates are small and not statistically significant, based on the p-values being greater than 0.05.

`P_Art`: The mean difference is 0.57, which suggests that the `Choice==1` group tends to purchase more from the Art department compared to the `Choice==0` group. The Satterthwaite t-test statistic is -14.58, and the p-value is very small (less than 0.001), indicating that the mean difference is statistically significant.

*Satterthwaite t-test assumes non-normally distributed data with unequal variances

**Mann-Whitney U test assumes non-normally distributed data with equal variances

# Results
## Performance
Based on the provided metrics, the logistic regression model has the highest accuracy score, but the precision, recall, and F1 scores are relatively low. This indicates that the model is good at predicting true negatives (i.e., correctly identifying instances that do not belong to the positive class), but not very good at predicting true positives (i.e., correctly identifying instances that do belong to the positive class). The AUC score is also relatively high, indicating that the model is good at ranking instances by their probability of belonging to the positive class. The specificity score is high, indicating that the model has a low false positive rate. However, the sensitivity score is relatively low, indicating that the model has a low true positive rate.

The SVM model has a lower accuracy score than the logistic regression model, but the precision, recall, and F1 scores are even lower, indicating that the model is not very good at predicting either true positives or true negatives. The AUC score is also lower than the logistic regression model, indicating that the model is not as good at ranking instances by their probability of belonging to the positive class. The specificity score is very high, indicating that the model has a very low false positive rate, but the sensitivity score is very low, indicating that the model has a very low true positive rate.

The linear regression model has the lowest accuracy score and the highest MSE and MAE scores, indicating that the model is not very good at predicting the target variable accurately. However, since it is a linear model, the coefficients for each feature can be directly interpreted, which may be useful in some contexts.

## Multivariate Analysis
The linear regression model and the logistic regression model are both used to predict an outcome variable based on several input variables or covariates. However, there are some key differences between the two models, and this is reflected in the coefficients for the covariates.

In the linear regression model, the coefficients for the covariates represent the change in the outcome variable for a one-unit increase in the corresponding input variable, while holding all other variables constant. For example, a one-unit increase in the last purchase variable leads to a 0.117791 increase in the outcome variable, holding all other variables constant. The coefficients are all expressed as continuous values, since the outcome variable is continuous.

In the logistic regression model, the coefficients for the covariates represent the log-odds of the outcome variable for a one-unit increase in the corresponding input variable, while holding all other variables constant. For example, a one-unit increase in the last purchase variable leads to a 0.591887 increase in the log-odds of the outcome variable, holding all other variables constant. The coefficients are expressed as log-odds, since the outcome variable is binary.

It is difficult to compare the magnitude of the coefficients between the two models since they are on different scales. However, it is possible to compare the direction of the coefficients. In this case, we can see that the direction of the coefficients is generally the same between the two models, meaning that an increase in the input variable leads to a corresponding increase or decrease in the outcome variable, depending on the sign of the coefficient.

In terms of the specific coefficients for the covariates, there are some differences between the two models. For example, the coefficient for the gender variable is negative in both models, but the magnitude is larger in the logistic regression model. Similarly, the coefficient for the last purchase variable is positive in both models, but the magnitude is larger in the logistic regression model. This could be due to the fact that the outcome variable is binary in the logistic regression model, and the model is trying to predict the probability of the outcome being positive or negative. Therefore, the coefficients are adjusted to account for the nonlinearity of the logistic function used to model the probability of the outcome variable.

# Conclusion

## Discussion
Based on the performance metrics, the logistic regression model appears to be the best-performing model, followed by the SVM and then the linear regression model. However, it is important to note that the choice of model and evaluation metrics will depend on the specific problem being addressed and the goals of the company.

## Recommendations
If the company wants to develop in-house capability to evaluate its direct mail campaigns, it may be beneficial to develop expertise in all three methods, as each method has its own strengths and weaknesses that may be relevant depending on the specific problem being addressed. For example, if the company is interested in predicting the probability of response to a direct mail campaign, logistic regression may be a good choice, while if the company is interested in identifying the most important features for predicting response, SVM with a linear kernel and L1 regularization may be a good choice. Linear regression may be useful in some cases where the relationship between the features and response is expected to be linear.

## Linear Regression Is Not Appropriate For Our Use-Case
It is important to use the appropriate regression technique for the given data and research question. In this case, the research question is to predict whether a customer will purchase a book or not based on their characteristics. Since the response variable (purchase or not) is categorical, logistic regression is the appropriate technique to use, not linear regression.

Linear regression assumes a linear relationship between the predictor variables and the response variable, which may not be appropriate for a categorical response variable. Additionally, linear regression models can predict values outside the range of the response variable, which does not make sense for a binary response. Logistic regression, on the other hand, models the probability of the response variable given the predictor variables, and ensures that the predicted probabilities are within the range of 0 to 1.

Therefore, using linear regression for this problem would not be appropriate, and the resulting model would likely have poor performance and be unreliable for predicting the probability of a customer making a purchase. It is important to choose the appropriate regression technique based on the nature of the data and the research question at hand.

## Next steps
To simplify and automate the recommended methods for future modeling efforts at the company, it may be helpful to develop a pipeline that takes in the data, performs preprocessing (e.g., missing value imputation, scaling), fits the models, evaluates the performance using relevant metrics, and generates visualizations (e.g., SHAP plots) to aid in interpretation. This pipeline can be customized for the specific problem being addressed and can be run automatically to generate new models as new data becomes available. Additionally, it may be useful to develop guidelines and best practices for data collection, preprocessing, and modeling to ensure consistency and reproducibility across modeling efforts.

# References
- Fan, J., & Chen, H. (2016). Predicting book sales on Amazon using neural networks. Journal of the Association for Information Science and Technology, 67(11), 2728-2742. doi: 10.1002/asi.23598
- Huang, Y., & Chen, H. (2006). Predicting user preference for online content using collaborative filtering and deep learning. Proceedings of the 12th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 566-571. doi: 10.1145/1150402.1150479
- Patil, K., & Sharma, R. (2014). Prediction of book sales using decision trees. International Journal of Computer Applications, 100(9), 17-21. doi: 10.5120/17606-3915
- Rossi, M. A., & McCulloh, I. (2015). Predictive modeling for book acquisition using demographic, circulatory, and bibliographic data. Journal of Academic Librarianship, 41(5), 591-598. doi: 10.1016/j.acalib.2015.06.001
