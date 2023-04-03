import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class StockData:
    """A class for managing and accessing stock market data.

    Attributes:
    -----------
    data : pandas.DataFrame
        The original data, loaded from a CSV file or passed as a DataFrame.
    train_data : pandas.DataFrame
        The subset of data corresponding to the first quarter.
    test_data : pandas.DataFrame
        The subset of data corresponding to the second quarter.

    Methods:
    --------
    get_train_data()
        Returns the training data subset.
    get_test_data()
        Returns the testing data subset.
    """
    def __init__(self, data):
        """Initialize the StockData object.

        Parameters:
        -----------
        data : str or pandas.DataFrame
            The input data. If a string, it should be a filepath to a CSV file. If a DataFrame, it will be used directly.
        """
        if isinstance(data, str):
            self.data = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            self.data = data
        else:
            raise ValueError("Invalid input data. Expected a filepath (string) or a pandas DataFrame.")
        
        self.data['date'] = pd.to_datetime(self.data['date'])

        self.data = self.data.drop('stock', axis=1)

        self.train_data = self.data[self.data['quarter'] == 1]
        self.test_data = self.data[self.data['quarter'] == 2]
        self.train_data.set_index('date', inplace=True)
        self.test_data.set_index('date', inplace=True)

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data

class StockModel:
    """A class for training and evaluating stock market prediction models.

    Attributes:
    -----------
    data : StockData
        An instance of StockData that contains the data to be used for training and testing.
    features : list
        A list of column names representing the input features to be used for training.
    target : str
        The name of the column representing the target variable to be predicted.
    train_data : pandas.DataFrame
        The subset of data used for training the model.
    test_data : pandas.DataFrame
        The subset of data used for evaluating the model.

    Methods:
    --------
    add_date_features()
        Adds year, month, and day of week features to the data.
    clean_up()
        Cleans up the data by converting dollar sign columns to float and updating the features list.
    drop_missing()
        Drops missing values from the data.
    train_linear_regression()
        Trains a linear regression model on the training data.
    train_decision_tree(max_depth=3)
        Trains a decision tree regression model on the training data.
    train_xgboost_regressor(max_depth=3, n_estimators=100, learning_rate=0.1)
        Trains an XGBoost regression model on the training data.
    train_svm(kernel='rbf', C=1.0, epsilon=0.1)
        Trains an SVM regression model on the training data.
    train_ridge_regression(alpha=1.0)
        Trains a Ridge regression model on the training data.
    train_lasso_regression(alpha=1.0)
        Trains a Lasso regression model on the training data.
    train_elastic_net_regression(alpha=1.0, l1_ratio=0.5)
        Trains an ElasticNet regression model on the training data.
    evaluate_model(model)
        Evaluates the trained model on the test data and returns the root mean squared error, mean absolute error, R^2 score, and average percentage error.
    get_risks()
        Computes and returns the stock volatility based on the weekly return.

    """
    def __init__(self, data, features, target):
        """Initialize the StockModel object.

        Parameters:
        -----------
        data : StockData
            An instance of StockData containing the input data.
        features : list
            A list of column names representing the input features to be used for training.
        target : str
            The name of the column representing the target variable to be predicted.
        """
        self.data = data
        self.features = features
        self.target = target
        self.train_data = data.get_train_data()
        self.test_data = data.get_test_data()
    
    def add_date_features(self):
        """Adds year, month, and day of week features to the data."""
        self.train_data.loc[:, 'year'] = self.train_data.index.year
        self.train_data.loc[:, 'month'] = self.train_data.index.month
        self.train_data.loc[:, 'dayofweek'] = self.train_data.index.dayofweek
        self.test_data.loc[:, 'year'] = self.test_data.index.year
        self.test_data.loc[:, 'month'] = self.test_data.index.month
        self.test_data.loc[:, 'dayofweek'] = self.test_data.index.dayofweek
        self.features += ['year', 'month', 'dayofweek']

    def clean_up(self):
        """Cleans up the data by converting dollar sign columns to float and updating the features list."""
        # convert dollar sign columns to float
        dollar_cols = ['open', 'high', 'low', 'close', 'next_weeks_open', 'next_weeks_close']
        for col in dollar_cols:
            self.train_data[col] = self.train_data[col].apply(lambda x: float(x.replace('$', '')))
            self.test_data[col] = self.test_data[col].apply(lambda x: float(x.replace('$', '')))
        self.features = [f for f in self.train_data.columns if f != self.target]

    def drop_missing(self):
        """Drops missing values from the data."""
        # drop missing values
        self.train_data.dropna(inplace=True)
        self.test_data.dropna(inplace=True)
        
    def train_linear_regression(self):
        """Trains a linear regression model on the training data."""
        X_train = self.train_data[self.features]
        y_train = self.train_data[self.target]
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model

    def train_decision_tree(self, max_depth=3):
        """Trains a decision tree regression model on the training data."""
        X_train = self.train_data[self.features]
        y_train = self.train_data[self.target]
        model = DecisionTreeRegressor(max_depth=max_depth)
        model.fit(X_train, y_train)
        return model
    
    def train_xgboost_regressor(self, max_depth=3, n_estimators=100, learning_rate=0.1):
        """Trains an XGBoost regression model on the training data."""
        X_train = self.train_data[self.features]
        y_train = self.train_data[self.target]
        model = XGBRegressor(max_depth=max_depth, n_estimators=n_estimators, learning_rate=learning_rate)
        model.fit(X_train, y_train)
        return model

    def train_svm(self, kernel='rbf', C=1.0, epsilon=0.1):
        """Trains an SVM regression model on the training data."""
        X_train = self.train_data[self.features]
        y_train = self.train_data[self.target]
        model = SVR(kernel=kernel, C=C, epsilon=epsilon)
        model.fit(X_train, y_train)
        return model
    
    def train_ridge_regression(self, alpha=1.0):
        """Trains a Ridge regression model on the training data."""
        X_train = self.train_data[self.features]
        y_train = self.train_data[self.target]
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        return model

    def train_lasso_regression(self, alpha=1.0):
        """Trains a Lasso regression model on the training data."""
        X_train = self.train_data[self.features]
        y_train = self.train_data[self.target]
        model = Lasso(alpha=alpha)
        model.fit(X_train, y_train)
        return model

    def train_elastic_net_regression(self, alpha=1.0, l1_ratio=0.5):
        """Trains an ElasticNet regression model on the training data."""
        X_train = self.train_data[self.features]
        y_train = self.train_data[self.target]
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
        model.fit(X_train, y_train)
        return model

    def evaluate_model(self, model):
        """Evaluates the given model on the test data."""
        X_test = self.test_data[self.features]
        y_test = self.test_data[self.target]
            
        y_pred = model.predict(X_test)
            
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        ape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        return rmse, mae, r2, ape

    def get_risks(self):
        """Returns the risks associated with the stock."""
        stock_data = self.train_data
        stock_data['weekly_return'] = (stock_data['close'] - stock_data['open']) / stock_data['open']
        stock_volatility = stock_data['weekly_return'].std()
        return stock_volatility