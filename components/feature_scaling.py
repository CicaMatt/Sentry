from sklearn.preprocessing import StandardScaler, MinMaxScaler


class Scaling:
    def scaling(self, data, method):
        print("Scaling training set features")
        if method == "zscore" or method == "default":
            # Feature Scaling - Z-Score Normalization
            data = StandardScaler().fit_transform(data)
        else:
            data = MinMaxScaler().fit_transform(data)

        return data
