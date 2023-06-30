# import shap
# import lime
# import lime.lime_tabular
# import numpy as np
# import xgboost
#
# class Explainability:
#     def explainability(self, model, X_train, y_train, X_test, y_test, method):
#         # SHAP
#         explainer = shap.TreeExplainer(model)
#         shap_values = explainer.shap_values(X_train, y_train)
#         expected_value = explainer.expected_value
#
#         # Generate summary dot plot
#         shap.summary_plot(shap_values, X_train, title="SHAP summary plot")
#
#         # Generate summary bar plot
#         shap.summary_plot(shap_values, X_train, plot_type="bar")
#
#         # Generate waterfall plot
#         shap.plots._waterfall.waterfall_legacy(expected_value, shap_values[79], features=X_train.loc[79, :],
#                                                feature_names=X_train.columns, max_display=15, show=True)
#
#         # Generate dependence plot
#         shap.dependence_plot("worst concave points", shap_values, X_train, interaction_index="mean concave points")
#
#         # Generate multiple dependence plots
#         for name in X_train.columns:
#             shap.dependence_plot(name, shap_values, X_train)
#         shap.dependence_plot("worst concave points", shap_values, X_train, interaction_index="mean concave points")
#
#         # Generate force plot - Multiple rows
#         shap.force_plot(explainer.expected_value, shap_values[:100, :], X_train.iloc[:100, :])
#
#         # Generate force plot - Single
#         shap.force_plot(explainer.expected_value, shap_values[0, :], X_train.iloc[0, :])
#
#         # Generate Decision plot
#         shap.decision_plot(expected_value, shap_values[79], link='logit', features=X_train.loc[79, :],
#                            feature_names=(X_train.columns.tolist()), show=True, title="Decision Plot")
#
#
#         # LIME
#         explainer = lime.lime_tabular.LimeTabularExplainer(X_test.to_numpy(), feature_names=X_test.columns,
#                                                            class_names=['0', '1'], verbose=True)
#
#         i = 10
#         exp = explainer.explain_instance(X_test[i], model.predict_proba, num_features=20)
#         exp.show_in_notebook(show_table=True)
