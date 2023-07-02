import pandas as pd
import seaborn
import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.inspection import permutation_importance


from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import PartialDependenceDisplay

# ULTIMI DUE METODI DA TESTARE

class Explainability:
    def explainability(self, model, X_train, y_train, X_test, y_test, prediction, classifier, method):
        features = list(X_test.columns.values)

        # Confusion Matrix
        if method == "confusionmatrix":
            if classifier == "svm":
                prediction = np.argmax(prediction, axis=1)
            truth = np.argmax(y_test, axis=1)
            confusion_matrix = metrics.confusion_matrix(truth, prediction)

            print("\nConfusion Matrix generated")
            print(confusion_matrix)

            # Seaborn confusion matrix
            confusion_matrix = pd.DataFrame(confusion_matrix,
                                            index=["Vulnerable", "Not vulnerable"],
                                            columns=["Vulnerable", "Not vulnerable"])
            plt.figure(figsize=(20, 10))
            seaborn.heatmap(confusion_matrix, annot=True)
            plt.title('Confusion Matrix ' + classifier)
            plt.ylabel('Truth Values')
            plt.xlabel('Predicted Values')
            plt.show()

        # Permutation feature importance
        elif method == "permutation":
            r = permutation_importance(classifier, X_test, y_test,
                                       n_repeats=30,
                                       random_state=0)
            for i in r.importances_mean.argsort()[::-1]:
                if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
                    print(f"{features[i]:<8}"
                        f"{r.importances_mean[i]:.3f}"
                        f" +/- {r.importances_std[i]:.3f}")


        # Partial Dependence Plots
        else:
            X, y = make_hastie_10_2(random_state=0)
            clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(X, y)
            features = [0, 1, (0, 1)]
            PartialDependenceDisplay.from_estimator(clf, X, features)
            plt.gcf()
            plt.gca()

        # # SHAP
        # explainer = shap.TreeExplainer(classifier)
        # shap_values = explainer.shap_values(X_train, y_train)
        # expected_value = explainer.expected_value
        #
        # # Generate summary dot plot
        # shap.summary_plot(shap_values, X_train, title="SHAP summary plot")
        #
        # # Generate summary bar plot
        # shap.summary_plot(shap_values, X_train, plot_type="bar")
        #
        # # Generate waterfall plot
        # shap.plots._waterfall.waterfall_legacy(expected_value, shap_values[79], features=X_train.loc[79, :],
        #                                        feature_names=X_train.columns, max_display=15, show=True)
        #
        # # Generate dependence plot
        # shap.dependence_plot("worst concave points", shap_values, X_train, interaction_index="mean concave points")
        #
        # # Generate multiple dependence plots
        # for name in X_train.columns:
        #     shap.dependence_plot(name, shap_values, X_train)
        # shap.dependence_plot("worst concave points", shap_values, X_train, interaction_index="mean concave points")
        #
        # # Generate force plot - Multiple rows
        # shap.force_plot(explainer.expected_value, shap_values[:100, :], X_train.iloc[:100, :])
        #
        # # Generate force plot - Single
        # shap.force_plot(explainer.expected_value, shap_values[0, :], X_train.iloc[0, :])
        #
        # # Generate Decision plot
        # shap.decision_plot(expected_value, shap_values[79], link='logit', features=X_train.loc[79, :],
        #                    feature_names=(X_train.columns.tolist()), show=True, title="Decision Plot")
        #
        # # LIME
        # explainer = lime.lime_tabular.LimeTabularExplainer(X_test.to_numpy(), feature_names=X_test.columns,
        #                                                    class_names=['0', '1'], verbose=True)
        #
        # i = 10
        # exp = explainer.explain_instance(X_test[i], model.predict_proba, num_features=20)
        # exp.show_in_notebook(show_table=True)
