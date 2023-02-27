import statsmodels.api as sm
from sklearn.svm import SVC
from sklearn.utils import resample
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error, roc_auc_score, confusion_matrix
import shap
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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
        if self.data_path:
            data = pd.read_csv(self.data_path)
        else:
            data = pd.concat([pd.read_csv(self.train_path), pd.read_csv(self.test_path)], ignore_index=True)
        return data
    
    def analyze_high_cardinality(self, data, columns, threshold=0.9):
        """
        This method analyzes high cardinality between variables in a pandas DataFrame.
        :param data: pandas DataFrame
        :param columns: list of column names to analyze
        :param threshold: threshold for the proportion of unique values to the number of rows
        :return: a list of columns with high cardinality
        """
        high_cardinality_columns = []
        for col in columns:
            if len(data[col].unique()) / data.shape[0] > threshold:
                high_cardinality_columns.append(col)
        return high_cardinality_columns
    
    def spearman_correlation(self, data, response_var=None):
        """
        This method performs Spearman correlation and visualizes the correlation matrix.
        :param data: pandas DataFrame
        :param response_var: name of the response variable (optional)
        """
        if response_var:
            corr = data.drop(response_var, axis=1).corr(method='spearman')
        else:
            corr = data.corr(method='spearman')
        sns.set(style="white")
        mask = np.triu(np.ones_like(corr, dtype=bool))
        f, ax = plt.subplots(figsize=(11, 9))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        sns.heatmap(corr, mask=mask, cmap=cmap, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
        plt.show()

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
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        return model

    def build_svm_model(self, X_train, y_train):
        model = SVC(kernel='rbf', C=100, probability=True)
        model.fit(X_train, y_train)
        return model
    
    def predict(self, model, X_new):
        y_pred = model.predict(X_new)
        return y_pred

    def evaluate_model(self, model, X_test, y_test):
        """
        Evaluates the performance of the given model on the test data.

        Parameters:
        ----------
        model : object
            The trained model to evaluate.
        X_test : pandas DataFrame
            The test data features.
        y_test : pandas DataFrame
            The test data response variable.

        Returns:
        -------
        dict
            A dictionary of evaluation metrics.
        """
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]  # probability of positive class
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            specificity = tn / (tn + fp)
            sensitivity = tp / (tp + fn)
            return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "auc": auc, "specificity": specificity, "sensitivity": sensitivity}
        else:
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            return {"mse": mse, "mae": mae}

    
    def select_best_model(self):
        """
        Builds and evaluates three different models (linear regression, logistic regression, and SVM) on the training and testing sets of the given data. The method prints the evaluation metrics for each model and returns the best-performing model based on weighted scores.

        Parameters:
        ----------
        None

        Returns:
        -------
        tuple
            A tuple containing the name of the best-performing model and the trained model object.
        """
        if self.train_path is not None and self.test_path is not None:
            X_train, X_test, y_train, y_test = self.preprocess_data()
        else:
            data = self.load_data()    
            X_train, X_test, y_train, y_test = self.preprocess_data(data)
        
        print("Metrics")
        print("=======")
        print("")

        linear_model = self.build_linear_model(X_train, y_train)
        linear_metrics = self.evaluate_model(linear_model, X_test, y_test)
        print("Linear regression:")
        print(pd.DataFrame.from_dict(linear_metrics, orient='index').rename(columns={0:''}).to_string())

        logit_model = self.build_logit_model(X_train, y_train)
        logit_metrics = self.evaluate_model(logit_model, X_test, y_test)
        print("")

        print("Logistic regression:")
        print(pd.DataFrame.from_dict(logit_metrics, orient='index').rename(columns={0:''}).to_string())

        svm_model = self.build_svm_model(X_train, y_train)
        svm_metrics = self.evaluate_model(svm_model, X_test, y_test)
        print("")
        
        print("SVM:")
        print(pd.DataFrame.from_dict(svm_metrics, orient='index').rename(columns={0:''}).to_string())
        
        weights = {"accuracy": 1, "precision": 1, "recall": 1, "f1": 1, "auc": 1, "sensitivity": 1, "specificity": 1, "mse": 0.5, "mae": 0.5}
        
        linear_score = sum(weights[key] * value for key, value in linear_metrics.items())
        logit_score = sum(weights[key] * value for key, value in logit_metrics.items())
        svm_score = sum(weights[key] * value for key, value in svm_metrics.items())
        print("")
        print("Weighted scores")
        print("===============")
        print("Linear regression, weighted score: ", linear_score)
        print("Logistic regression, weighted score: ", logit_score)
        print("SVM, weighted score: ", svm_score)
        
        models = {"linear": {"model": linear_model, "score": linear_score, "metrics":linear_metrics},
                "logit": {"model": logit_model, "score": logit_score, "metrics":logit_metrics},
                "svm": {"model": svm_model, "score": svm_score, "metrics":svm_metrics}}
        
        best_model = max(models, key=lambda x: models[x]["score"])
        model = models[best_model]["model"]

        return best_model, model
    
    def shap_analysis(self, model, X_test, plot_dependence=False):
            """
            Performs SHAP analysis on the given model and test data and visualizes the SHAP values for the first instance in the test data. Optionally, the method can also visualize two-way dependence plots for each feature.

            Parameters:
            ----------
            model : object
                The trained model to analyze.
            X_test : pandas DataFrame
                The test data features.
            plot_dependence : bool, optional
                Whether to plot two-way dependence plots for each feature.

            Returns:
            -------
            None
            """
            # Initialize the SHAP explainer for the given model and test data
            explainer = shap.Explainer(model, X_test)

            # Compute SHAP values for the test data
            shap_values = explainer(X_test)

            # Visualize the SHAP values for the first instance in the test data
            shap.plots.waterfall(shap_values[0], max_display=10)
            
            # Compute the SHAP summary plot
            shap.summary_plot(shap_values, X_test)

            # Show the plot
            plt.show()
            
            # Visualize two-way dependence plots for each feature
            if plot_dependence:
                for feature in X_test.columns:
                    shap.plots.scatter(shap_values[:, feature], color=shap_values)


class ModelAnalyzer:
    """
    A class for analyzing and visualizing the coefficients of a linear, logistic, or SVM model.

    Parameters:
    ----------
    model_type : str
        The type of model to analyze. Must be one of 'linear', 'logistic', or 'svm'.
    X_train : pandas DataFrame
        The training data features.
    y_train : pandas DataFrame
        The training data response variable.

    Returns:
    -------
    None
    """
    def __init__(self, model_type, X_train, y_train):
        self.model_type = model_type
        self.X_train = X_train
        self.y_train = y_train
        self.summary_table = None
        self.coefs = None
        self.lower_cis = None
        self.upper_cis = None
        
    def analyze(self):
        """
        Trains the model and extracts the coefficient values and 95% CIs.

        Parameters:
        ----------
        None

        Returns:
        -------
        None
        """
        if self.model_type == 'linear':
            # Train a linear regression model
            model = sm.OLS(self.y_train, self.X_train)
            results = model.fit()

            # Create a summary table
            self.summary_table = results.summary2().tables[1]

        elif self.model_type == 'logistic':
            # Train a logistic regression model
            model = sm.Logit(self.y_train, self.X_train)
            results = model.fit()

            # Create a summary table
            self.summary_table = results.summary2().tables[1]

        elif self.model_type == 'svm':
            # Train an SVM model
            model = SVC(kernel='linear', C=1, probability=True)
            model.fit(self.X_train, self.y_train)

            # Bootstrap the data to obtain 95% CIs
            n_bootstraps = 1000
            bootstrapped_coefs = []
            for i in range(n_bootstraps):
                # Resample the data with replacement
                X_resampled, y_resampled = resample(self.X_train, self.y_train)

                # Fit the SVM model to the resampled data
                model.fit(X_resampled, y_resampled)

                # Store the coefficient values
                bootstrapped_coefs.append(model.coef_)

            # Compute the 95% CIs
            self.lower_cis = np.percentile(bootstrapped_coefs, 2.5, axis=0)
            self.upper_cis = np.percentile(bootstrapped_coefs, 97.5, axis=0)

        else:
            raise ValueError('Invalid model type')

        # Extract the coefficient values and 95% CIs
        if self.model_type in ['linear', 'logistic']:
            self.coefs = self.summary_table['Coef.']
            self.lower_cis = self.summary_table['[0.025']
            self.upper_cis = self.summary_table['0.975]']

    def get_covariate_table(self):
        """
        Creates a table of covariates, coefficients, and 95% CIs.

        Parameters:
        ----------
        None

        Returns:
        -------
        covariate_table : pandas DataFrame
            A table of covariates, coefficients, and 95% CIs.
        """
        # Create a table of covariates, coefficients, and 95% CIs
        covariate_table = pd.DataFrame({'Covariate': self.X_train.columns,
                                        'Coefficient': self.coefs,
                                        '95% CI (Lower)': self.lower_cis,
                                        '95% CI (Upper)': self.upper_cis})
        return covariate_table
