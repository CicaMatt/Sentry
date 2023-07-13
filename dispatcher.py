import pickle
import numpy as np
import pandas as pd

from components.data_balancing import Balancing
from components.data_cleaning import Cleaning
from components.explainability import Explainability
from components.feature_scaling import Scaling
from components.feature_selection import Selection
from components.metrics import Metrics
from components.classification import Classification
from components.setup import Setup
from components.validation import Validation


class Dispatcher:

    def __init__(self, data, repo_link, path, to_predict):
        self.data = data
        self.repo_link = repo_link
        self.dir_path = path
        self.dataset_to_predict = to_predict

    def start(self):
        # Data setup
        data = Setup().data_setup("dataset.csv")

        # Data Cleaning
        data = Cleaning().cleaning(data, self.data['Data Cleaning'])
        columns = data.columns

        # Validation - Train/Test Split
        if self.data['Validation'] == "ttsplit":
            x_training, x_testing, y_training, y_testing = Validation().data_validation(data, self.data['Validation'])
            x_training = x_training[:, :-1]
            x_testing = x_testing[:, :-1]

            # Feature Scaling
            self.scaler, x_training, x_testing = Scaling().scaling(x_training, x_testing, self.data['Feature Scaling'])

            # Feature Selection
            self.selector, x_training, x_testing, self.selected_features = Selection().selection(x_training, x_testing,
                                                                 columns, y_training, self.data['Feature Selection'])

            x_training = np.hstack((x_training, y_training.reshape(-1, 1)))
            # Data Balancing
            x_training, y_training = Balancing().dataBalancing(x_training, y_training, self.data['Data Balancing'])

            x_training = x_training[:, :-1]
            # Model classification
            prediction, self.classifier = Classification().data_classification(x_training, x_testing, y_training,
                                                                        self.data['Classifier'])

            # Metrics calculation
            Metrics().metrics(y_testing, prediction)

            # Model explanation
            Explainability().explainability(x_testing, y_testing, prediction, self.classifier,
                                            self.data['Explaination Method'])

        # Validation - Stratified or Standard K Fold Validation
        else:
            labels_full = data["vulnerable"]
            indexes, data, labels = Validation().data_validation(data, self.data['Validation'])
            best_accuracy = 0
            best_precision = 0
            best_recall = 0
            best_f1 = 0
            best_mean = 0

            fold = 1
            best_fold = 0

            for training_index, testing_index in indexes:
                print("\nFold #" + str(fold))
                x_training, x_testing = data[training_index], data[testing_index]
                y_training, y_testing = labels[training_index], labels[testing_index]

                y_training = np.argmax(y_training, axis=1)
                y_testing = np.argmax(y_testing, axis=1)

                x_training = x_training[:, :-1]
                x_testing = x_testing[:, :-1]

                # Feature Scaling
                scaler, x_training, x_testing = Scaling().scaling(x_training, x_testing, self.data['Feature Scaling'])

                # Feature Selection
                selector, x_training, x_testing, selected_features = Selection().selection(x_training, x_testing, columns, y_training,
                                                            self.data['Feature Selection'])

                # print(x_training.shape)
                # print(y_training.shape)

                x_training = np.hstack((x_training, y_training.reshape(-1, 1)))
                # Data Balancing
                x_training, y_training = Balancing().dataBalancing(x_training, y_training, self.data['Data Balancing'])
                x_training = x_training[:, :-1]

                # Model classification
                prediction, classifier = Classification().data_classification(x_training, x_testing, y_training, self.data['Classifier'])

                # Metrics calculation
                accuracy, precision, recall, f1 = Metrics().metrics(y_testing, prediction)
                if (accuracy + precision + recall + f1)/4 > best_mean:
                    best_mean = (accuracy + precision + recall + f1)/4
                    best_accuracy = accuracy
                    best_precision = precision
                    best_recall = recall
                    best_f1 = f1
                    best_fold = fold
                    self.classifier = classifier
                    self.scaler = scaler
                    self.selector = selector
                    self.prediction = prediction
                    self.features_testing = x_testing
                    self.testing_labels = y_testing
                fold += 1


            print("\nBest fold: #" + str(best_fold))
            print("Accuracy: " + "{:.2%}".format(float(best_accuracy)))
            print("Precision: " + "{:.2%}".format(float(best_precision)))
            print("Recall: " + "{:.2%}".format(float(best_recall)))
            print("F1: " + "{:.2%}".format(float(best_f1)))


            # Model explanation
            Explainability().explainability(self.features_testing, self.testing_labels, self.prediction, self.classifier,
                                            self.data['Explaination Method'])

        # Saving model and preprocessing components
        pickle.dump(self.scaler, open(self.dir_path + "/scaler.sav", 'wb'))
        if self.selector is not None:
            pickle.dump(self.selector, open(self.dir_path + "/selector.sav", 'wb'))
        pickle.dump(self.classifier, open(self.dir_path + "/classifier.sav", 'wb'))

        # Final prediction on another test set
        print("\n\nPrediction on input data...")
        self.dataset_to_predict = self.scaler.fit_transform(self.dataset_to_predict)
        columns_predict = columns.drop(labels=['vulnerable'])
        if self.selector is None and self.data['Feature Selection'] != "default":
            self.dataset_to_predict = pd.DataFrame(self.dataset_to_predict, columns=columns_predict)
            self.dataset_to_predict = self.dataset_to_predict.loc[:, self.selected_features]
        elif self.data['Feature Selection'] != "default":
            self.dataset_to_predict = self.selector.transform(self.dataset_to_predict)
        predictions = self.classifier.predict(self.dataset_to_predict)

        get_first_char = np.vectorize(lambda x: int(np.floor(x)))
        predictions = get_first_char(predictions)
        complete_dataset = self.dataset_to_predict
        complete_dataset = pd.DataFrame(complete_dataset)
        complete_dataset.insert(len(complete_dataset.columns), "vulnerable", predictions)
        complete_dataset.to_csv("generated_dataset.csv")

        # # Hyperparameters optimization
        # model = HP_Optimization().hp_optimization(self.data['Hyper-parameters Optimization'])
        #

