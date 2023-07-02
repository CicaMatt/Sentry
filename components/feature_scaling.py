from sklearn.preprocessing import StandardScaler, MinMaxScaler


class Scaling:
    def scaling(self, x_training, x_testing, method):
        print("Scaling training and test set features")

        if method == "zscore" or method == "default":
            # Feature Scaling - Z-Score Normalization
            x_training = StandardScaler().fit_transform(x_training)
            x_testing = StandardScaler().transform(x_testing)

        else:
            x_training = MinMaxScaler().fit_transform(x_training)
            x_testing = MinMaxScaler().transform(x_testing)

        return x_training, x_testing
