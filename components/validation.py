from sklearn.model_selection import train_test_split, KFold


class Validation:
    def data_validation(data, labels_full, method):
        # training data for the neural net
        training_data = data.values

        # labels for training
        labels = labels_full.values

        # K-Fold Validation - 10 Fold
        if method == "kfold":
            print("Validation - K-Fold Validation - 10 Fold")
            kf = KFold(n_splits=10, shuffle=True)
            return kf.split(training_data), training_data, labels

        # Train/Test split - 80/20
        if method == "ttsplit":
            print("Validation - 80/20 split")
            x_training, x_testing, y_training, y_testing = train_test_split(training_data, labels, test_size=0.20,
                                                                            random_state=42)
            return x_training, x_testing, y_training, y_testing
