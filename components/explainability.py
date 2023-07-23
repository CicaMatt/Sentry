import warnings

import pandas as pd
import seaborn
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from sklearn import metrics
from sklearn.inspection import permutation_importance
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import PartialDependenceDisplay


class Explainability:
    def explainability(self, x_training, x_test, truth, prediction, classifier, selected_column_names, method):
        x_test = pd.DataFrame(x_test)
        features = list(x_test.columns.values)

        # Confusion Matrix
        if "confusionmatrix" in method:
            if classifier == "svm":
                prediction = np.argmax(prediction, axis=1)

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
        if "permutation" in method:
            r = permutation_importance(classifier, x_test, truth,
                                       n_repeats=30,
                                       random_state=0)
            # Iteration on importance mean scores of feature in descending order
            for i in r.importances_mean.argsort()[::-1]:
                # Check if mean importance of the feature, minus two times the std deviation is higher than 0,
                # to determine if feature importance is meaningful when compared to permutation variability
                if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
                    print(f"{features[i]:<8}"
                          f"{r.importances_mean[i]:.3f}"
                          f" +/- {r.importances_std[i]:.3f}")


        # Partial Dependence Plots
        if "partialdependence" in method:
            warnings.filterwarnings('ignore')
            x_training = pd.DataFrame(x_training, columns=selected_column_names)
            n_cols = 2
            n_rows = int(len(x_training.columns)/n_cols)
            fig, ax = plt.subplots(n_rows, n_cols, figsize=(10, 12))
            PartialDependenceDisplay.from_estimator(classifier, x_training, features=np.arange(x_training.shape[1]), feature_names=selected_column_names, ax=ax, n_cols=n_cols)
            fig.suptitle('Partial Dependence Plots')
            fig.tight_layout()
            plt.show()

