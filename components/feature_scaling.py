from sklearn.preprocessing import StandardScaler, MinMaxScaler


class Scaling:
    def scaling(self, x_training, x_testing, method):
        print("Scaling training and test set features")

        if method == "zscore" or method == "default":
            # Feature Scaling - Z-Score Normalization
            scaler = StandardScaler()

            x_training = scaler.fit_transform(x_training)
            x_testing = scaler.transform(x_testing)

        else:
            # Feature Scaling - Min-Max Scaling
            scaler = MinMaxScaler()

            x_training = scaler.fit_transform(x_training)
            x_testing = scaler.transform(x_testing)

        return x_training, x_testing
