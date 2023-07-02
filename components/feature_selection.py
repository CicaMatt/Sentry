import numpy as np
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2


class Selection:

    def selection(self, x_training, x_testing, method):
        labels_full = x_training["vulnerable"]
        # Variance Threshold
        if method == "variancethreshold":
            x_training = VarianceThreshold().fit_transform(x_training)
            x_testing = VarianceThreshold().transform(x_testing)

        # KBest
        elif method == "kbest":
            features = x_training.iloc[:, :-1]
            labels = x_training.iloc[:, -1]
            x_training = SelectKBest(chi2, k=6).fit_transform(features, labels)
            x_testing = SelectKBest(chi2, k=6).transform(features, labels)

        # Pearson's Correlation
        else:
            corr = x_training.corr()["vulnerable"].sort_values(ascending=False)[1:]
            abs_corr = abs(corr)
            relevant_features = abs_corr[abs_corr > 0.4]
            x_training = x_training.loc[:, relevant_features]

            x_testing = x_testing.loc[:, relevant_features]

        return x_training, x_testing


