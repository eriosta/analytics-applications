import pandas as pd
import numpy as np
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, recall_score, precision_score, accuracy_score, roc_auc_score
import shap
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="No data for colormapping provided")
warnings.filterwarnings("ignore", message="Unable to determine Axes to steal space for Colorbar")


class CustomerRetention:
    def __init__(self, data_path, features):
        self.df = pd.read_csv(data_path)[features]
        self.data = self.df.drop(['ret_exp','crossbuy','sow','freq','profit'], axis=1)
        self.data_acquired = self.df[self.data['acquisition'] == 1]
        self.rf_dur = RandomForestRegressor(random_state=42)
        self.rf_acq = RandomForestClassifier(random_state=42)
        self.logit_acq = LogisticRegression(random_state=42)


    def perform_k_fold_cross_validation(self, k=5):
        # Prepare the data
        X = self.data.drop(['acquisition', 'duration'], axis=1)
        X_acquired = self.data_acquired.drop(['acquisition', 'duration'], axis=1)
        y_acq = self.data['acquisition']
        y_dur = self.data_acquired['duration']

        # Perform k-fold cross-validation for the classifier
        classifier_scores = cross_val_score(self.rf_acq, X, y_acq, cv=k, scoring='roc_auc')
        classifier_mean_score = classifier_scores.mean()

        # Perform k-fold cross-validation for the regressor
        regressor_scores = cross_val_score(self.rf_dur, X_acquired, y_dur, cv=k, scoring='r2')
        regressor_mean_score = regressor_scores.mean()

        # Print the results
        print(f"Classifier {k}-fold cross-validation mean AUC: {classifier_mean_score:.4f}")
        print(f"Regressor {k}-fold cross-validation mean R2 score: {regressor_mean_score:.4f}")

    def print_corr_matrix(self):
        corr_matrix = self.data.corr()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap="YlGnBu", ax=ax)
        plt.title("Correlation Matrix")
        plt.show()

    def predict_acquisition_and_duration(self):
        # Task 1: Predict which customers will be acquired and for how long they will be retained
        X = self.data.drop(['acquisition', 'duration'], axis=1)
        X_acquired = self.data_acquired.drop(['acquisition', 'duration'], axis=1)
        y_acq = self.data['acquisition']
        y_dur = self.data_acquired['duration']

        # Train-test split for acquisition prediction
        X_train_acq, X_test_acq, y_train_acq, y_test_acq = train_test_split(X, y_acq, test_size=0.2, random_state=42)

        # Train-test split for duration prediction
        X_train_dur, X_test_dur, y_train_dur, y_test_dur = train_test_split(X_acquired, y_dur, test_size=0.2, random_state=42)

        # Fit the models
        self.rf_acq.fit(X_train_acq, y_train_acq)
        self.rf_dur.fit(X_train_dur, y_train_dur)

        self.logit_acq.fit(X_train_acq, y_train_acq)


        # Make predictions
        y_pred_acq = self.rf_acq.predict(X_test_acq)
        y_pred_dur = self.rf_dur.predict(X_test_dur)

        # Save the train-test splits and predictions as instance variables
        self.X_train, self.X_test = X_train_acq, X_test_acq
        self.y_train_acq, self.y_test_acq = y_train_acq, y_test_acq
        self.X_train_dur, self.X_test_dur = X_train_dur, X_test_dur
        self.y_train_dur, self.y_test_dur = y_train_dur, y_test_dur
        self.y_pred_acq, self.y_pred_dur = y_pred_acq, y_pred_dur

    def hyperparameter_optimization(self):
        # Task 2: Compute variable importance to detect interactions and optimize hyperparameters for acquired customers.
        X = self.data.drop(['acquisition', 'duration'], axis=1)
        y_acq = self.data['acquisition']

        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        grid_search = GridSearchCV(estimator=self.rf_acq, param_grid=param_grid, cv=5, n_jobs=-1, verbose=0)
        grid_search.fit(X, y_acq)

        best_params = grid_search.best_params_
        print(best_params)

        # Update the rf_acq with the best hyperparameters found
        self.rf_acq = RandomForestClassifier(n_estimators=best_params['n_estimators'],
                                             max_depth=best_params['max_depth'],
                                             min_samples_split=best_params['min_samples_split'],
                                             min_samples_leaf=best_params['min_samples_leaf'],
                                             random_state=42)

        self.rf_acq.fit(X, y_acq)

        importances = self.rf_acq.feature_importances_
        indices = np.argsort(importances)[::-1]
        for f in range(X.shape[1]):
            print("%d. %s (%f)" % (f + 1, X.columns[indices[f]], importances[indices[f]]))

    def compare_models(self, model_type):

        def bootstrap_ci(y_true, y_pred, func, n_bootstrap=1000, alpha=0.05):
            indices = np.arange(len(y_true))
            bootstrap_samples = [func(y_true[resampled_indices], y_pred[resampled_indices]) for resampled_indices in (resample(indices) for _ in range(n_bootstrap))]
            return np.percentile(bootstrap_samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])

        # Task 4: Compare performance of different models
        n_bootstrap = 1000
        alpha = 0.05

        if model_type == 'regressor':
            model = RandomForestRegressor(random_state=42)
            model.fit(self.X_train_dur, self.y_train_dur)
            y_pred = model.predict(self.X_test_dur)
            metrics = ['MSE', 'MAE', 'R2']
            metric_funcs = [mean_squared_error, mean_absolute_error, r2_score]

        elif model_type == 'classifier':
            model = RandomForestClassifier(random_state=42)
            model.fit(self.X_train, self.y_train_acq)
            y_pred = model.predict(self.X_test)
            metrics = ['Accuracy', 'Recall', 'Precision', 'AUROC']
            metric_funcs = [accuracy_score, recall_score, precision_score, roc_auc_score]
        
        elif model_type == 'logit':
            model = LogisticRegression(random_state=42)
            model.fit(self.X_train, self.y_train_acq)
            y_pred = model.predict(self.X_test)
            metrics = ['Accuracy', 'Recall', 'Precision', 'AUROC']
            metric_funcs = [accuracy_score, recall_score, precision_score, roc_auc_score]

        else:
            print("Invalid model type. Choose either 'regressor' or 'classifier' or 'logit'")
            return

        results_with_ci = []
        for metric, metric_func in zip(metrics, metric_funcs):
            result = metric_func(self.y_test_acq if model_type in ['classifier', 'logit'] else self.y_test_dur, y_pred)
            ci = bootstrap_ci(self.y_test_acq.values if model_type in ['classifier', 'logit'] else self.y_test_dur.values, y_pred, metric_func, n_bootstrap, alpha)
            results_with_ci.append((result, ci))

        results_df = pd.DataFrame(results_with_ci, columns=['Mean', 'CI'], index=metrics)
        print(results_df)

    def shap_analysis(self, model_type):
        """
        Perform SHAP analysis and visualize the results with summary and dependence plots.

        Args:
        model_type (str): Either 'regressor' or 'classifier', depending on the model to analyze.
        """

        if model_type == 'regressor':
            explainer = shap.Explainer(self.rf_dur)
            shap_values = explainer(self.X_test_dur)
            self.shap_regressor = shap_values.values

            # Plot the summary plot
            shap.summary_plot(shap_values, self.X_test_dur)

            for feature in self.X_test_dur.columns:
                shap.dependence_plot(feature, shap_values.values, self.X_test_dur)

        elif model_type == 'classifier':
            explainer = shap.TreeExplainer(self.rf_acq)
            shap_values = explainer.shap_values(self.X_test)

            # Combine SHAP values for both classes
            combined_shap_values = shap_values[0] + shap_values[1]

            self.shap_classifier = combined_shap_values

            # Plot the summary plot
            shap.summary_plot(combined_shap_values, self.X_test)

            for feature in self.X_test.columns:
                shap.dependence_plot(feature, combined_shap_values, self.X_test)

        else:
            print("Invalid model type. Choose either 'regressor' or 'classifier'.")

