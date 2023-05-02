import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class Scaling:
    def scaling(data, labels_full, method):
        # Feature Scaling - Z-Score Normalization
        print("Scaling training set features")
        data = StandardScaler().fit_transform(data)
        # data = MinMaxScaler().fit_transform(data)
        data = pd.DataFrame(data)

        return data
