import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2
from sklearn.preprocessing import StandardScaler


class Preprocessing:
    def data_preprocessing(data, labels_full):
        # Feature Scaling - Z-Score Normalization
        print("Scaling training set features")
        data = StandardScaler().fit_transform(data)
        # data = MinMaxScaler().fit_transform(data)
        data = pd.DataFrame(data)

        # Outliers removal - Z Score approach
        # print("Removing outliers")
        # filtered = (np.abs(data) < 3).all(axis=1)
        # data['type'] = lab
        # data = data[filtered]
        # labels_full = pd.get_dummies(data['type'], prefix='type')
        # data = data.drop(columns='type')
        # print(data.shape)

        # Outliers removal - Interquartile Range approach
        # print("Removing outliers")
        # Q1, Q3 = data.quantile(0.25), data.quantile(0.75)
        # IQR = Q3 - Q1
        # IQR_outliers = data[((data < (Q1 - 1.5 * IQR)) |(data > (Q3 + 1.5 * IQR))).any(axis=1)]
        # data = data[~((data < (Q1 - 1.5 * IQR)) |(data > (Q3 + 1.5 * IQR))).any(axis=1)]
        # data['type'] = labels_column
        # labels_full = pd.get_dummies(data['type'], prefix='type')
        # data = data.drop(columns='type')
        # print(data.shape)

        # Feature Selection
        print("Feature Selection")
        data = VarianceThreshold().fit_transform(data)
        data = SelectKBest(chi2, k=80).fit_transform(data, labels_full)
        data = pd.DataFrame(data)

        # Feature Extraction
        # print("Feature Extraction")
        # data = PCA(80).fit_transform(data)
        # data = pd.DataFrame(data)

        return data
