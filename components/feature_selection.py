import numpy as np
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2


class Selection:

    def selection(data, method):
        # Variance Threshold
        if method == "Variance Threshold":
            data = VarianceThreshold().fit_transform(data)
            return data

        # KBest
        elif method == "KBest":
            data = SelectKBest(chi2, k=80).fit_transform(data, labels_full)
            return data

        # Pearson's Correlation
        else:
            cor_support, cor_feature = Selection.cor_selector(X, labels_full, num_feats)
            return cor_feature

    def cor_selector(X, y, num_feats):
        cor_list = []
        feature_name = X.columns.tolist()
        # calculate the correlation with y for each feature
        for i in X.columns.tolist():
            cor = np.corrcoef(X[i], y)[0, 1]
            cor_list.append(cor)
        # replace NaN with 0
        cor_list = [0 if np.isnan(i) else i for i in cor_list]
        # feature name
        cor_feature = X.iloc[:, np.argsort(np.abs(cor_list))[-num_feats:]].columns.tolist()
        # feature selection? 0 for not select, 1 for select
        cor_support = [True if i in cor_feature else False for i in feature_name]
        return cor_support, cor_feature
