import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error

class BBBC_Model:
    """
    A class for building and evaluating models to predict customer purchase behavior based on several independent variables.

    Attributes:
    ----------
    data_path : str
        The path to the data file.
    response_var : str
        The dependent variable for the analysis - purchase or no purchase of the book.
    test_size : float
        The proportion of the data to use for testing the models. Default is 0.2.

    Methods:
    -------
    load_data():
        Loads the data from the specified file.
    preprocess_data(data):
        Preprocesses the data by removing missing values, creating dummy variables, and splitting into training and testing sets.
    build_linear_model(X_train, y_train):
        Builds and trains a linear regression model.
    build_logit_model(X_train, y_train):
        Builds and trains a logistic regression model.
    build_svm_model(X_train, y_train):
        Builds and trains a support vector machine model.
    evaluate_model(model, X_test, y_test):
        Evaluates the performance of the model using various metrics.
    select_best_model():
        Builds, trains, and evaluates several models and selects the best-performing model based on the F1 score.
    """
    def __init__(self, data_path=None, response_var=None, test_size=0.2, train_path=None, test_path=None):
        self.data_path = data_path
        self.response_var = response_var
        self.test_size = test_size
        self.train_path = train_path
        self.test_path = test_path

    def load_data(self):
        data = pd.read_csv(self.data_path)
        return data

    def preprocess_data(self, data=None):
        """
        Preprocesses the data by removing missing values, creating dummy variables, and splitting into training and testing sets.

        Parameters:
        ----------
        data : pandas DataFrame
            The data to preprocess.
        train_path : str or None, optional
            The path to the CSV file containing the custom training set. If provided, the data argument is ignored.
        test_path : str or None, optional
            The path to the CSV file containing the custom testing set. If provided, the data argument is ignored.

        Returns:
        -------
        X_train, X_test, y_train, y_test : pandas DataFrames
            The preprocessed training and testing sets.
        """
        if (self.train_path is not None and self.test_path is not None):
            X_train = pd.read_csv(self.train_path)
            y_train = X_train[self.response_var]
            X_train.drop(columns=[self.response_var], inplace=True)
            X_test = pd.read_csv(self.test_path)
            y_test = X_test[self.response_var]
            X_test.drop(columns=[self.response_var], inplace=True)
        else:
            # Remove any missing values
            data.dropna(inplace=True)
            
            # Create dummy variables for categorical variables
            data = pd.get_dummies(data, columns=["Gender"])
            
            # Split the data into training and testing sets
            X = data.drop(columns=[self.response_var])
            y = data[self.response_var]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=42)
        
        return X_train, X_test, y_train, y_test


    def build_linear_model(self, X_train, y_train):
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model

    def build_logit_model(self, X_train, y_train):
        model = LogisticRegression()
        model.fit(X_train, y_train)
        return model

    def build_svm_model(self, X_train, y_train):
        model = SVC()
        model.fit(X_train, y_train)
        return model

    def evaluate_model(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):  # check if model is binary classification
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
        else:
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            return {"mse": mse, "mae": mae}
    
    def select_best_model(self):
        if self.train_path is not None and self.test_path is not None:
            X_train, X_test, y_train, y_test = self.preprocess_data()
        else:
            data = self.load_data()    
            X_train, X_test, y_train, y_test = self.preprocess_data(data)
            
        linear_model = self.build_linear_model(X_train, y_train)
        linear_metrics = self.evaluate_model(linear_model, X_test, y_test)
        
        logit_model = self.build_logit_model(X_train, y_train)
        logit_metrics = self.evaluate_model(logit_model, X_test, y_test)
        
        svm_model = self.build_svm_model(X_train, y_train)
        svm_metrics = self.evaluate_model(svm_model, X_test, y_test)
        
        weights = {"accuracy": 1, "precision": 1, "recall": 1, "f1": 1, "mse": 0.5, "mae": 0.5}
        
        linear_score = sum(weights[key] * value for key, value in linear_metrics.items())
        logit_score = sum(weights[key] * value for key, value in logit_metrics.items())
        svm_score = sum(weights[key] * value for key, value in svm_metrics.items())
        
        models = {"linear": {"model": linear_model, "score": linear_score},
                "logit": {"model": logit_model, "score": logit_score},
                "svm": {"model": svm_model, "score": svm_score}}
        
        best_model = max(models, key=lambda x: models[x]["score"])
        return best_model
    

models = BBBC_Model(response_var='Choice',train_path='books/BBBC-Train.csv',test_path='books/BBBC-Test.csv')

models.select_best_model()
