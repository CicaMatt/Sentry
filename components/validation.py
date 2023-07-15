import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold


class Validation:
    def data_validation(self, data, method):
        # training data for the neural net
        training_data = data.values

        # labels for training
        labels_full = pd.get_dummies(data['vulnerable'], prefix='vulnerable')
        labels = labels_full.values

        # Train/Test split - 80/20
        if method == "ttsplit" or method == "default":
            print("Validation - Train/Test Split - 80/20")
            x_training, x_testing, y_training, y_testing = train_test_split(training_data, labels, test_size=0.20,
                                                                            random_state=42)
            y_training = np.argmax(y_training, axis=1)
            y_testing = np.argmax(y_testing, axis=1)

            return x_training, x_testing, y_training, y_testing

        # K-Fold Validation - 10 Fold
        if method == "kfold":
            print("Validation - K-Fold Validation - 10 Fold")
            kf = KFold(n_splits=10, shuffle=True)
            return kf.split(training_data), training_data, labels

        else:
            print("Validation - Stratified K-Fold Validation - 5 Fold")
            skf = StratifiedKFold(n_splits=5, shuffle=True)
            return skf.split(training_data, labels.argmax(1)), training_data, labels
