from books.model import BBBC_Model, ModelAnalyzer
    
models = BBBC_Model(response_var='Choice',train_path='books/BBBC-Train.csv',test_path='books/BBBC-Test.csv')

data = models.load_data()

cols = list(data.columns)
models.analyze_high_cardinality(data=data,columns=cols,threshold=0.9)

models.spearman_correlation(data=data,response_var='Choice')

# BBBC is considering a similar mail campaign in the Midwest where it has data for 50,000
# customers. Such mailings typically promote several books. The allocated cost of the mailing is
# $0.65/addressee (including postage) for the art book, and the book costs $15 to purchase and
# mail. The company allocates overhead to each book at 45% of cost. The selling price of the
# book is $31.95. Based on the model, which customers should Bookbinders target? How much
# more profit would you expect the company to generate using these models as compare to
# sending the mail offer to the entire list. 


models.select_best_model()

X_train, X_test, y_train, y_test = models.preprocess_data()

cov = ModelAnalyzer(model_type='linear',X_train=X_train,y_train=y_train)
cov.analyze()
cov.get_covariate_table()

cov = ModelAnalyzer(model_type='logistic',X_train=X_train,y_train=y_train)
cov.analyze()
cov.get_covariate_table()

# cov = ModelAnalyzer(model_type='svm',X_train=X_train,y_train=y_train)
# cov.analyze()
# cov.get_covariate_table()


models.shap_analysis(models.build_logit_model(X_train,y_train), X_test=X_test,plot_dependence=True)

models.shap_analysis(models.build_linear_model(X_train,y_train), X_test=X_test, plot_dependence=True)


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