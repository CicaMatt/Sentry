import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2


class Selection:

    def selection(self, x_training, x_testing, method):
        # Variance Threshold
        if method == "variancethreshold":
            x_training = VarianceThreshold().fit_transform(x_training)
            x_testing = VarianceThreshold().transform(x_testing)

        # KBest
        elif method == "kbest":
            k_best = SelectKBest(chi2, k=6)

            features_x_training = np.delete(x_training, 14, 1)
            features_x_testing = np.delete(x_testing, 14, 1)
            labels = x_training[:, 14]

            x_training = k_best.fit_transform(features_x_training, labels)
            x_testing = k_best.transform(features_x_testing)

        # Pearson's Correlation
        else:
            corr = x_training.corr()["vulnerable"].sort_values(ascending=False)[1:]
            abs_corr = abs(corr)
            relevant_features = abs_corr[abs_corr > 0.4]
            x_training = x_training.loc[:, relevant_features]

            x_testing = x_testing.loc[:, relevant_features]

        return x_training, x_testing, labels


