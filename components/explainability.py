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
    def explainability(self, X_test, y_test, prediction, classifier, method):
        X_test = pd.DataFrame(X_test)
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
            plt.title('Confusion Matrix ' + str(classifier))
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

