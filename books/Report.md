# Executive Summary
In this analysis, we used data from Bookbinders Book Club (BBBC) to explore the use of predictive modeling approaches to improve the efficacy of its direct mail program. Specifically, we analyzed a subset of the database containing data for 400 customers who purchased the book and 1200 customers who did not, to identify the factors that influence book purchasing behavior.

We evaluated three commonly used machine learning algorithms – linear regression, logistic regression, and SVM – to predict book purchasing behavior based on customer characteristics. We found that the logistic regression model had the highest accuracy score, but relatively low precision, recall, and F1 scores, indicating that the model is better at predicting true negatives than true positives. The SVM model had a lower accuracy score than the logistic regression model, and even lower precision, recall, and F1 scores. The linear regression model had the lowest accuracy score and the highest MSE and MAE scores, indicating that it is not very good at predicting the target variable accurately.

Based on the performance metrics, it may be beneficial for BBBC to develop expertise in all three methods, as each method has its own strengths and weaknesses that may be relevant depending on the specific problem being addressed. Additionally, a pipeline can be developed to automate and simplify future modeling efforts at the company.

# Problem
Bookbinders Book Club (BBBC) is a distributor of specialty books through direct marketing. In anticipation of using database marketing, BBBC made a strategic decision to build and maintain a detailed database about its members containing all relevant information about them. The company is exploring whether to use predictive modeling approaches to improve the efficacy of its direct mail program. The objective of this analysis is to identify the factors that influence book purchasing behavior based on customer characteristics, to help BBBC develop a response model that can be used to improve the efficacy of its direct mail program.

# Literature Review
Describe use of machine learning in prediction of book buying behavio based on customer characteristics

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


# Data
Data processing stuff, distributions

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

## SHAP Analysis

# Conclusion

## Discussion
Based on the performance metrics, the logistic regression model appears to be the best-performing model, followed by the SVM and then the linear regression model. However, it is important to note that the choice of model and evaluation metrics will depend on the specific problem being addressed and the goals of the company.

## Recommendations
If the company wants to develop in-house capability to evaluate its direct mail campaigns, it may be beneficial to develop expertise in all three methods, as each method has its own strengths and weaknesses that may be relevant depending on the specific problem being addressed. For example, if the company is interested in predicting the probability of response to a direct mail campaign, logistic regression may be a good choice, while if the company is interested in identifying the most important features for predicting response, SVM with a linear kernel and L1 regularization may be a good choice. Linear regression may be useful in some cases where the relationship between the features and response is expected to be linear.

## Next steps
To simplify and automate the recommended methods for future modeling efforts at the company, it may be helpful to develop a pipeline that takes in the data, performs preprocessing (e.g., missing value imputation, scaling), fits the models, evaluates the performance using relevant metrics, and generates visualizations (e.g., SHAP plots) to aid in interpretation. This pipeline can be customized for the specific problem being addressed and can be run automatically to generate new models as new data becomes available. Additionally, it may be useful to develop guidelines and best practices for data collection, preprocessing, and modeling to ensure consistency and reproducibility across modeling efforts.

# References
Fan, J., & Chen, H. (2016). Predicting book sales on Amazon using neural networks. Journal of the Association for Information Science and Technology, 67(11), 2728-2742. doi: 10.1002/asi.23598
Huang, Y., & Chen, H. (2006). Predicting user preference for online content using collaborative filtering and deep learning. Proceedings of the 12th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 566-571. doi: 10.1145/1150402.1150479
Patil, K., & Sharma, R. (2014). Prediction of book sales using decision trees. International Journal of Computer Applications, 100(9), 17-21. doi: 10.5120/17606-3915
Rossi, M. A., & McCulloh, I. (2015). Predictive modeling for book acquisition using demographic, circulatory, and bibliographic data. Journal of Academic Librarianship, 41(5), 591-598. doi: 10.1016/j.acalib.2015.06.001