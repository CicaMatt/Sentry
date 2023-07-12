import pickle

import numpy as np
import pandas as pd
from numpy import savetxt
from sklearn.preprocessing import MinMaxScaler

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
        if (self.data['Validation'] == "ttsplit"):
            x_training, x_testing, y_training, y_testing = Validation().data_validation(data, self.data['Validation'])
            x_training = x_training[:, :-1]
            x_testing = x_testing[:, :-1]

            # Feature Scaling
            self.scaler, x_training, x_testing = Scaling().scaling(x_training, x_testing, self.data['Feature Scaling'])

            # Feature Selection
            self.selector, x_training, x_testing, self.selected_features = Selection().selection(x_training, x_testing, self.data['Feature Selection'], columns, y_training)

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
            indexes, training_data, labels = Validation().data_validation(data, labels_full, self.data['Validation'])
            best_accuracy = 0

            for training_index, testing_index in indexes:
                x_training, x_testing = training_data[training_index], training_data[testing_index]
                y_training, y_testing = labels[training_index], labels[testing_index]

                # Feature Scaling
                scaler, x_training, x_testing = Scaling().scaling(x_training, x_testing, self.data['Feature Scaling'])

                # Feature Selection
                selector, x_training, x_testing = Selection().selection(x_training, x_testing, y_training, y_testing,
                                                            self.data['Feature Selection'])

                # Data Balancing
                x_training, y_training = Balancing().dataBalancing(x_training, self.data['Data Balancing'])

                # Model classification
                prediction, classifier = Classification().data_classification(x_training, x_testing, y_training,
                                                                            y_testing, self.data['Classifier'])

                # Metrics calculation
                accuracy = Metrics().metrics(y_testing, prediction)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    self.classifier = classifier
                    self.scaler = scaler
                    self.selector = selector
                    self.features_testing = x_testing
                    self.testing_labels = y_testing


            # Model explanation
            # Explainability().explainability(self.features_testing, self.testing_labels, prediction, self.classifier,
            #                                 self.data['Explaination Method'])

        #save models
        pickle.dump(self.scaler, open(self.dir_path + "/scaler.sav", 'wb'))
        if self.selector is not None:
            pickle.dump(self.selector, open(self.dir_path + "/selector.sav", 'wb'))
        pickle.dump(self.classifier, open(self.dir_path + "/classifier.sav", 'wb'))

        # Prediction on another test set
        print("\n\nPrediction on input data...")
        self.dataset_to_predict = self.scaler.fit_transform(self.dataset_to_predict)
        columns_predict = columns.drop(labels=['vulnerable'])
        if self.selector is None and self.data['Feature Selection'] != "default":
            self.dataset_to_predict = pd.DataFrame(self.dataset_to_predict, columns=columns_predict)
            self.dataset_to_predict = self.dataset_to_predict.loc[:, self.selected_features]
        elif self.data['Feature Selection'] != "default":
            self.dataset_to_predict = self.selector.transform(self.dataset_to_predict)
        print(self.dataset_to_predict)
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

