# I. Executive Summary
This report aims to analyze the performance of predictive models and the risk of selected stocks in the stock market. The report includes an evaluation of six different predictive models on data for each stock, and the risk of each stock is calculated based on their volatility.

# II. The Problem
## Introduction
The stock market is a complex and dynamic environment, with various factors affecting the performance of stocks. Understanding the trends and risks of stocks is crucial for investors to make informed decisions and maximize their returns. Predictive models can be used to forecast the performance of stocks, while stock risk analysis can help investors identify potential risks and manage their portfolios.

## Purpose of Study
The purpose of this study is to evaluate the performance of different predictive models and calculate the risk of selected stocks in the stock market. The study aims to provide insights into the most effective predictive models for different stocks and identify stocks with high and low risk levels. The findings of this study can be useful for investors, financial analysts, and other stakeholders in the stock market.

## Questions
This study aims to answer the following questions:
* What is the most effective predictive model for each stock?
* Which stocks have the highest and lowest risks based on their volatility?
* How can the findings of this study be used to inform investment decisions and manage stock portfolios?

## Outline
The report is organized as follows:
* Section I presents the executive summary
* Section II provides the problem, introduction and background of the study
* Section III provides a literature review of relevant studies and methodologies used in stock market prediction and risk analysis.
* Section IV describes the data sources, data collection process, and data preprocessing steps used in the analysis.
* Section V presents the results of the predictive models and the stock risk analysis.
* Section VI discusses the implications of the findings and provides recommendations for investors and other stakeholders.
* Section VII provides a conclusion and suggests areas for future research.

# III. Review of Related Literature
## Machine Learning for Prediction Models
### Linear Regression
Linear regression is a simple and widely used regression model that is used to model the relationship between a dependent variable and one or more independent variables. It is a linear approach that assumes a linear relationship between the dependent and independent variables. It aims to minimize the sum of the squared differences between the actual and predicted values of the dependent variable.

### Decision Tree Regressor
Decision tree regressor is a non-linear regression model that is used to model the relationship between the dependent variable and one or more independent variables. It is a tree-like model that splits the data into smaller and smaller subsets, based on the values of the independent variables. It aims to create a model that predicts the value of the dependent variable by traversing the tree from the root node to a leaf node.

### XGBoost Regressor
XGBoost (Extreme Gradient Boosting) regressor is a decision tree-based ensemble model that is used for regression problems. It is similar to decision tree regressor but uses an ensemble of decision trees to improve the prediction accuracy. It is a gradient boosting algorithm that trains weak models in a sequential manner and combines them to form a stronger model. It also uses regularization techniques to prevent overfitting.

### Ridge Regression
Ridge regression is a linear regression model that is used to handle multicollinearity in the data. It adds a penalty term to the sum of the squared differences between the actual and predicted values of the dependent variable. The penalty term is proportional to the square of the coefficients of the independent variables, which shrinks them towards zero. It helps to reduce the complexity of the model and prevent overfitting.

### Lasso Regression
Lasso regression is also a linear regression model that is used to handle multicollinearity in the data. It adds a penalty term to the sum of the absolute values of the coefficients of the independent variables. It tends to produce sparse models by forcing some coefficients to be exactly zero. It helps to reduce the complexity of the model and prevent overfitting.

### Support Vector Regression (SVR)
SVR is a non-linear regression model that is used to model the relationship between the dependent variable and one or more independent variables. It is based on the concept of support vector machine (SVM) and aims to find the hyperplane that maximizes the margin between the predicted and actual values of the dependent variable. It uses a kernel function to transform the data into a higher-dimensional space to make it separable.

### Elastic Net Regression
Elastic net regression is a linear regression model that combines the features of ridge and lasso regression. It adds a penalty term that is a combination of the squared sum of the coefficients of the independent variables (ridge regression) and the sum of the absolute values of the coefficients of the independent variables (lasso regression). It helps to balance the strengths and weaknesses of both ridge and lasso regression.

## Comparing Machine Learning Models for Regression Task
In summary, the main differences between these regression models lie in their underlying assumptions, techniques used to handle multicollinearity and overfitting, and the complexity of the models. Linear regression assumes a linear relationship between the dependent and independent variables, while decision tree regressor, xgboost regressor, and SVR are non-linear models. Ridge and lasso regression add penalty terms to the sum of the squared differences or absolute values of the coefficients of the independent variables to handle multicollinearity and prevent overfitting. Elastic net regression combines the features of both ridge and lasso regression. XGBoost regressor and elastic net regression are more complex than the other models and can achieve better prediction accuracy in certain cases.

## Stock Risk
### Stock Volatility
The standard deviation is a measure of the dispersion of returns around the mean return of an asset or portfolio. It provides an indication of the volatility of returns and is often used as a measure of risk. The higher the standard deviation, the greater the risk associated with the investment.
### Capital Asset Pricing Model
CAPM is a model that attempts to calculate the expected return of an asset or portfolio based on its level of systematic risk or beta. Systematic risk is the risk that cannot be diversified away, and it is represented by the asset's or portfolio's sensitivity to the market's overall movements. CAPM uses the market risk premium, the risk-free rate, and the asset's beta to estimate the expected return. The formula for CAPM is as follows:

$r = R_\text{f} + \beta \times (R_\text{m} - R_\text{f})$

Where:

* $r$ is the expected return on an asset or portfolio
* $R_f$ is the risk-free rate of return
* $\beta$ is the systematic risk or beta of the asset or portfolio
* $R_m$ is the expected return on the market
* $(R_m - R_f)$ is the market risk premium, which represents the additional return expected from investing in the market compared to a risk-free investment.

# IV. Methodology
## Variables

### Pre-Processing Variables
* `quarter`: the yearly quarter (1 = Jan-Mar; 2 = Apr-Jun). **1 = Training Set; 2 = Testing Set**.
* `stock`: the stock symbol.
* `date`: the last business day of the week (typically a Friday).
* `open`: the price of the stock at the beginning of the week.
* `high`: the highest price of the stock during the week.
* `low`: the lowest price of the stock during the week.
* `close`: the price of the stock at the end of the week.
* `volume`: the number of shares of stock that traded hands in the week.
* `percent_change_price`: the percentage change in price throughout the week.
* `percent_change_volume_over_last_week`: the percentage change in the number of shares of stock that traded hands for this week compared to the previous week.
* `previous_weeks_volume`: the number of shares of stock that traded hands in the previous week.
* `next_weeks_open`: the opening price of the stock in the following week. **TARGET**.
* `next_weeks_close`: the closing price of the stock in the following week.
* `percent_change_next_weeks_price`: the percentage change in price of the stock in the following week.
* `days_to_next_dividend`: the number of days until the next dividend.
* `percent_return_next_dividend`: the percentage of return on the next dividend.

### Post-Processing Variables
In summary, we derived `year`, `month`, and `day_of_week` as integers from `date`. All features below were used for training and testing except for `weekly_return` and `stock_volatility`.

* `stock`: the stock symbol.
* `year`: derived from `date`.
* `month`: derived from `date`.
* `day_of_week`: derived from `date`.
* `open`: the price of the stock at the beginning of the week.
* `high`: the highest price of the stock during the week.
* `low`: the lowest price of the stock during the week.
* `close`: the price of the stock at the end of the week.
* `volume`: the number of shares of stock that traded hands in the week.
* `percent_change_price`: the percentage change in price throughout the week.
* `percent_change_volume_over_last_week`: the percentage change in the number of shares of stock that traded hands for this week compared to the previous week.
* `previous_weeks_volume`: the number of shares of stock that traded hands in the previous week.
* `next_weeks_open`: the opening price of the stock in the following week. **TARGET**.
* `next_weeks_close`: the closing price of the stock in the following week.
* `percent_change_next_weeks_price`: the percentage change in price of the stock in the following week.
* `days_to_next_dividend`: the number of days until the next dividend.
* `percent_return_next_dividend`: the percentage of return on the next dividend.
* `weekly_return`: the difference between the closing price (`close`) and opening price (`open`) of the stock for each week and then dividing it by the opening price. This results in the percentage change in price from the opening to the closing of the week, which represents the weekly return for that stock. **Used for risk analysis and not for training the machine learning models**.
* `stock_volatility`: the standard deviation of the weekly_return values across all weeks in the dataset. This gives an indication of how much the returns fluctuate over time and is often used as a measure of the stock's volatility. **Used for risk analysis and not for training the machine learning models**.

## Model Development

In stock market prediction models, one common goal is to predict the next week's opening price for a stock. The opening price is influenced by various factors, including news events, market sentiment, and technical analysis, and can provide insights into the direction of the stock's price movement. 

Accurately predicting the next week's opening price can help investors make informed investment decisions and evaluate the performance of their investment strategies. To achieve this, we trained different regression models, including linear regression, decision tree regression, XGBoost regression, SVM regression, ridge regression, lasso regression, and elastic net regression, to predict the next week's opening price.

Refer to the class `StockModel` for a detailed view of the model development. See the implementation on `main.ipynb`.

## Data Sampling
A sample of historical data from the Dow Jones was used. We used data from 1/7/2011-3/25/2011 (quarter 1) and 4/1/2011-6/24/2011 (quarter 2) for the following stocks: 

<table>
  <thead>
    <tr>
      <th>Stock</th>
      <th>Name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>AA</td>
      <td>Alcoa Corporation</td>
    </tr>
    <tr>
      <td>AXP</td>
      <td>American Express Company</td>
    </tr>
    <tr>
      <td>BA</td>
      <td>The Boeing Company</td>
    </tr>
    <tr>
      <td>BAC</td>
      <td>Bank of America Corporation</td>
    </tr>
    <tr>
      <td>CAT</td>
      <td>Caterpillar Inc.</td>
    </tr>
    <tr>
      <td>CSCO</td>
      <td>Cisco Systems, Inc.</td>
    </tr>
    <tr>
      <td>CVX</td>
      <td>Chevron Corporation</td>
    </tr>
    <tr>
      <td>DD</td>
      <td>DuPont de Nemours, Inc.</td>
    </tr>
    <tr>
      <td>DIS</td>
      <td>The Walt Disney Company</td>
    </tr>
    <tr>
      <td>GE</td>
      <td>General Electric Company</td>
    </tr>
    <tr>
      <td>HD</td>
      <td>The Home Depot, Inc.</td>
    </tr>
    <tr>
      <td>HPQ</td>
      <td>HP Inc.</td>
    </tr>
    <tr>
      <td>IBM</td>
      <td>International Business Machines Corporation</td>
    </tr>
    <tr>
      <td>INTC</td>
      <td>Intel Corporation</td>
    </tr>
    <tr>
      <td>JNJ</td>
      <td>Johnson &amp; Johnson</td>
    </tr>
    <tr>
      <td>JPM</td>
      <td>JPMorgan Chase &amp; Co.</td>
    </tr>
    <tr>
      <td>KO</td>
      <td>The Coca-Cola Company</td>
    </tr>
    <tr>
      <td>KRFT</td>
      <td>The Kraft Heinz Company</td>
    </tr>
    <tr>
      <td>MCD</td>
      <td>McDonald's Corporation</td>
    </tr>
    <tr>
      <td>MMM</td>
      <td>3M Company</td>
    </tr>
    <tr>
      <td>MRK</td>
      <td>Merck &amp; Co., Inc.</td>
    </tr>
    <tr>
      <td>MSFT</td>
      <td>Microsoft Corporation</td>
    </tr>
    <tr>
      <td>PFE</td>
      <td>Pfizer Inc.</td>
    </tr>
    <tr>
      <td>PG</td>
      <td>The Procter &amp; Gamble Company</td>
    </tr>
    <tr>
      <td>T</td>
      <td>AT&amp;T Inc.</td>
    </tr>
    <tr>
      <td>TRV</td>
      <td>The Travelers Companies, Inc.</td>
    </tr>
    <tr>
      <td>UTX</td>
      <td>United Technologies Corporation</td>
    </tr>
    <tr>
      <td>VZ</td>
      <td>Verizon Communications Inc.</td>
    </tr>
    <tr>
      <td>WMT</td>
      <td>Walmart Inc.</td>
    </tr>
    <tr>
      <td>XOM</td>
      <td>Exxon Mobil Corporation</td>
    </tr>
  </tbody>
</table>


## Data Source
In our study, we utilized a secondary data source obtained from a previous analysis conducted by Brown et al. (2013). The data source included historical stock prices for companies listed in the Dow Jones Index, as well as other relevant financial data.

Using a secondary data source allowed us to leverage the work done by previous researchers and focus our efforts on developing and comparing regression models to predict the next week's opening price. By utilizing a pre-existing data source, we were able to avoid the time and resources required to collect and clean the data ourselves, allowing us to focus on the model building process. However, it is important to note that using a secondary data source also has limitations, such as potential biases in the original data collection and potential issues with the data quality. Therefore, we took steps to carefully evaluate the data and ensure that it was appropriate for use in our analysis.

## Data Modeling
We defined a class named `StockModel` that contains various methods for training and evaluating different regression models for stock market prediction. The class takes in an instance of `StockData` class, which is assumed to have the input data containing features and the target variable.

The class contains several methods that perform various data pre-processing tasks such as adding date features, cleaning up data by converting dollar columns to float, and dropping missing values, all discussed in detail under the **Data** section. Additionally, the class contains several methods for training different regression models, including linear regression, decision tree regression, XGBoost regression, SVM regression, ridge regression, lasso regression, and elastic net regression. 

The `evaluate_model` method takes a trained model as input and evaluates its performance on the test data by calculating various evaluation metrics, including root mean squared error, mean absolute error, R-squared score, and average percentage error. 

### Root Mean Squared Error (RMSE)
The RMSE measures the average distance between the predicted and actual values. It is calculated as the square root of the average of the squared differences between the predicted and actual values. The lower the RMSE value, the better the model's performance. The formula for RMSE is as follows:

$\text{RMSE} = \sqrt{\text{mean}((y_\text{true} - y_\text{pred})^2)}$

### Mean Absolute Error (MAE)
The MAE measures the average absolute difference between the predicted and actual values. It is calculated as the average of the absolute differences between the predicted and actual values. The lower the MAE value, the better the model's performance. The formula for MAE is as follows:

$\text{MAE} = \text{mean}(|y_\text{true} - y_\text{pred}|)$

### R-squared (R^2) score
The R-squared score measures the proportion of the variance in the target variable that is explained by the model. It ranges from 0 to 1, with higher values indicating better performance. A value of 0 means that the model does not explain any variance in the target variable, while a value of 1 means that the model perfectly explains the variance in the target variable. The formula for R-squared is as follows:

$R^2 = 1 - \frac{\text{SS}_\text{res}}{\text{SS}_\text{tot}}$

where $\text{SS}_\text{res}$ is the sum of squared residuals and $\text{SS}_\text{tot}$ is the total sum of squares.

### Average Percentage Error (APE)
The APE measures the average percentage difference between the predicted and actual values. It is calculated as the average of the absolute percentage differences between the predicted and actual values. The lower the APE value, the better the model's performance. The formula for APE is as follows:

$APE = \text{mean} \left( \left| \frac{y_{\text{true}} - y_{\text{pred}}}{y_{\text{true}}} \right| \right) \times 100$

Overall, the `StockModel` class provides a comprehensive toolset for training, evaluating, and predicting stock market trends using various regression models. The class is designed to help users select the best model for their specific prediction task and provides an easy-to-use interface for accessing and manipulating stock market data.

## Methodological Assumptions and Limitations
In our study, we made several assumptions and faced some limitations. First, our analysis assumes that the historical data we used is representative of the future behavior of the stocks, which may not always be the case. 

Furthermore, we relied on secondary data from an external source, which means we had limited control over the data collection process and the quality of the data. We also only considered a limited set of predictor variables and did not account for external factors, such as macroeconomic indicators or industry trends, that may affect the stock prices.

Another limitation is that our study only focused on a specific time period and did not consider long-term trends or events that may impact the stock prices. Additionally, the models we used are only as good as the data and assumptions they are built upon, and may not perform as well in different market conditions.

Finally, we acknowledge that there are several other statistical models and techniques that could be used to predict stock prices, such as time-series analysis or more advanced machine learning algorithms, which may yield different results.

# V. Data
## Data Cleaning and Processing
The `StockData` class contains methods for managing and accessing stock market data. The `__init__` method initializes an instance of `StockData` with input data passed either as a CSV filepath or as a pandas DataFrame. The `get_train_data` and `get_test_data` methods return the training and testing data subsets, respectively.

The `StockModel` class contains methods for training and evaluating stock market prediction models. The `__init__` method initializes an instance of `StockModel` with an instance of `StockData`, a list of column names representing the input features to be used for training, and the name of the column representing the target variable to be predicted. The other methods include `add_date_features`, which adds year, month, and day of week features to the data, `clean_up`, which cleans up the data by converting dollar sign columns to float and updating the features list, `drop_missing`, which drops missing values from the data.

## Data limitations
See section **Methodological Assumptions and Limitations**.

# VI. Findings

## Performance of Predictive Models
In our analysis, we tested six different predictive models on data for each stock. We used a custom metric score to rank the performance between the models, which takes into account the RMSE, MAE, APE values, and 1 minus the R2 value. The model with the lowest metric score was considered the best performing model.

Overall, we found that linear regression was the most common and best performing model for 18 stocks, followed by Ridge regression as the top model for 7 stocks. XGBoost, Lasso, and Elastic Net regression were the best models for 3, 1, and 1 stocks, respectively. However, decision tree regression was not the leading model for any of the stocks.

Specifically, we found that linear regression was the best model for predicting trends in stocks AA, AXP, BAC, CAT, DD, DIS, GE, HD, JPM, KRFT, MMM, MRK, MSFT, PFE, T, TRV, UTX, and VZ. The trends for stocks CVX, HPQ, INTC, JNJ, KO, MCD, and XOM were best predicted using Ridge regression. XGBoost was the best model for stocks BA, CSCO, and WMT, while Lasso regression and Elastic Net regression were the best models for predicting trends in IBM and PG, respectively.

The individual RMSE, MAE, APE, and R2 values for all six models per each stock, along with their calculated custom metric, can be found in the supplemental `./evaluation_metrics` directory (link [here](https://github.com/eriosta/analytics-applications/tree/main/stock_market/evaluation_metrics)).

![Best Performing Models by Stock](https://user-images.githubusercontent.com/84827743/229398922-e267f310-8b79-4c6e-8ad2-7ac41ff8cae0.png)

## Stock Risk
To assess the risk of each stock, we first calculated the weekly return for each week by subtracting the opening price from the closing price and dividing by the opening price. This gives us a percentage change in price for that week, which represents the weekly return for that stock. We then calculated the standard deviation of the weekly returns for each stock across all weeks to determine the stock volatility. This metric reflects the degree of fluctuation in returns over time and is commonly used to measure stock risk. We ranked the stocks by their risk and compared them to the median risk value for all stocks. Based on our analysis, we found that CSCO, HPQ, BAC, AA, AXP, MRK, HD, DIS, DD, PFE, BA, CAT, XOM, TRV, and CVX had risks above the median, while JPM, MCD, MMM, GE, INTC, KO, IBM, KRFT, T, WMT, JNJ, UTX, MSFT, VZ, and PG had risks below the median.

![Risks by Stock](https://user-images.githubusercontent.com/84827743/229398947-b0a8b8ec-df99-463d-bace-3e339bbe578b.png)

# VII. Conclusions and Recommendations

## Discussion

### Predicting Next Week's Open Stock Price
The results of our analysis showed that linear regression was the most commonly used and best-performing model for predicting the trends of most stocks. This suggests that linear regression is a reliable method for predicting stock trends and can be used by investors to make informed investment decisions. Additionally, we found that Ridge regression was the top model for several stocks, indicating that it can also be a useful predictive model for investors.

However, we also found that decision tree regression was not the leading predictive model for any of the stocks, which suggests that this method may not be the most effective for stock prediction. This highlights the importance of testing multiple models to find the best-performing one for each stock.

### Choosing Volatility or CAPM

In terms of stock risk, our analysis found that some stocks had risks above or below the median, which can be useful information for investors when making investment decisions. This suggests that investors should consider not only the potential returns of a stock but also the level of risk associated with it.

Calculating risk with standard deviation and calculating risk with the Capital Asset Pricing Model (CAPM) are two different methods for assessing the risk of an investment.

Standard deviation measures the volatility of returns for a particular security or portfolio over a given period of time. It is a measure of the dispersion of returns around the average return for the period. In the context of stock market investments, a higher standard deviation indicates greater risk, as the returns are more volatile and unpredictable. Standard deviation only takes into account historical price data, and does not consider other factors that may influence future returns, such as market trends or news events.

On the other hand, the CAPM is a model that uses the relationship between the expected return of an asset and the market risk premium to estimate the required return of the asset, based on its risk. The CAPM takes into account both the risk of the asset (as measured by beta) and the risk of the market as a whole (as measured by the market risk premium), and provides an estimate of the expected return that investors would require for holding that asset. The CAPM is based on several assumptions, including that investors are rational and risk-averse, and that markets are efficient.

In summary, while standard deviation and the CAPM are both methods for assessing risk, they differ in their approach and the factors they take into account. Standard deviation focuses on historical price data, while the CAPM considers the risk of the asset and the market as a whole, and provides an estimate of the expected return based on that risk. Investors may use both methods together, as they provide complementary information about the risk of an investment.

## Conclusions 
Our analysis provides insights into the performance of different predictive models and the risks associated with individual stocks. We found that linear regression and Ridge regression were the most effective predictive models for most stocks, while decision tree regression was not as effective. Additionally, our analysis showed that some stocks had risks above or below the median, highlighting the importance of considering risk when making investment decisions.

## Recommendations

Based on our findings, we recommend that investors consider using linear regression or Ridge regression when predicting stock trends. Additionally, investors should consider the level of risk associated with a stock before making investment decisions. We also recommend that future studies explore other predictive models and metrics to determine the most effective methods for predicting stock trends and risk.

## Alternative Methodologies

There are several alternative methodologies that could be considered for future analyses.

One alternative is to use machine learning algorithms, such as neural networks or random forests, instead of traditional regression models. These algorithms can capture more complex patterns and relationships in the data, potentially leading to improved predictive accuracy. Additionally, ensemble methods, which combine multiple models to make predictions, could also be explored.

Another alternative is to incorporate additional features into the analysis, such as news sentiment data or macroeconomic indicators. These factors can have a significant impact on stock prices and could improve the accuracy of the predictive models.

Finally, an alternative approach would be to analyze the data at a higher frequency, such as daily or hourly, instead of weekly. This could potentially provide more granular insights into short-term trends and fluctuations in stock prices.

Overall, there are several alternative methodologies that could be explored in future analyses to potentially improve the accuracy and insights gained from stock market prediction models.

# VIII. References
James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). An introduction to statistical learning. New York: Springer.

Montgomery, D. C., Peck, E. A., & Vining, G. G. (2012). Introduction to linear regression analysis. John Wiley & Sons.

Breiman, L., Friedman, J., Stone, C. J., & Olshen, R. A. (1984). Classification and regression trees. CRC press.

Hastie, T., Tibshirani, R., & Friedman, J. (2009). The elements of statistical learning: data mining, inference, and prediction. Springer.

Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (pp. 785–794).

Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression: Biased estimation for nonorthogonal problems. Technometrics, 12(1), 55-67.

James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). An introduction to statistical learning. New York: Springer.

Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. Journal of the Royal Statistical Society: Series B (Methodological), 58(1), 267-288.

Hastie, T., Tibshirani, R., & Friedman, J. (2009). The elements of statistical learning: data mining, inference, and prediction. Springer.

Drucker, H., Burges, C. J., Kaufman, L., Smola, A. J., & Vapnik, V. (1997). Support vector regression machines. In Advances in neural information processing systems (pp. 155-161).

Schölkopf, B., & Smola, A. J. (2002). Learning with kernels: support vector machines, regularization, optimization, and beyond. MIT press.

Zou, H., & Hastie, T. (2005). Regularization and variable selection via the elastic net. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 67(2), 301-320.

Friedman, J., Hastie, T., & Tibshirani, R. (2010). Regularization paths for generalized linear models via coordinate descent. Journal of statistical software, 33(1), 1-22.

Brown, M. S., Pelosi, M., & Dirska, H. (2013). Dynamic-radius species-conserving genetic algorithm for the financial forecasting of Dow Jones Index stocks. In F. Masseglia, P. Poncelet, & M. Teisseire (Eds.), Machine Learning and Data Mining in Pattern Recognition (pp. 27-41). Springer.
