from books.model import BBBC_Model, ModelAnalyzer
    
models = BBBC_Model(response_var='Choice',train_path='books/BBBC-Train.csv',test_path='books/BBBC-Test.csv')

models.select_best_model()

X_train, X_test, y_train, y_test = models.preprocess_data()

# models.shap_analysis(models.build_logit_model(X_train,y_train), X_test=X_test)

# models.shap_analysis(models.build_linear_model(X_train,y_train), X_test=X_test)


# Train an SVM model
# svm_model = models.build_svm_model(X_train,y_train)

# # Initialize the SHAP explainer
# explainer = shap.KernelExplainer(lambda x: svm_model.decision_function(x), X_train)
# explainer.feature_names = list(X_train.columns)

# # Compute SHAP values for the test data
# shap_values = explainer.shap_values(X_test)

# # Visualize the SHAP values
# shap.initjs()
# shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0])