import pickle

import numpy as np
from numpy import savetxt
from sklearn.preprocessing import MinMaxScaler

from components.data_balancing import Balancing
from components.data_cleaning import Cleaning
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

        selector, scaler, balancer, classifier = None, None, None, None

        # Validation - Train/Test Split
        if (self.data['Validation'] == "ttsplit"):
            x_training, x_testing, y_training, y_testing = Validation().data_validation(data, self.data['Validation'])

            # Feature Scaling
            scaler, x_training, x_testing = Scaling().scaling(x_training, x_testing, self.data['Feature Scaling'])

            # Feature Selection
            selector, x_training, x_testing, labels = Selection().selection(x_training, x_testing, self.data['Feature Selection'])

            # Data Balancing
            balancer, x_training, y_training = Balancing().dataBalancing(x_training, labels, self.data['Data Balancing'])

            # Model classification
            prediction, classifier = Classification().data_classification(x_training, x_testing, y_training, y_testing,
                                                                        self.data['Classifier'])

            # Metrics calculation
            Metrics().metrics(y_testing, prediction, classifier)

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
                balancer, x_training = Balancing().dataBalancing(x_training, self.data['Data Balancing'])

                # Model classification
                prediction, classifier = Classification().data_classification(x_training, x_testing, y_training,
                                                                            y_testing, self.data['Classifier'])

                # Metrics calculation
                accuracy = Metrics().metrics(y_testing, prediction, classifier)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    self.classifier = classifier

        #save models
        pickle.dump(scaler, open(self.dir_path + "/scaler.sav", 'wb'))
        pickle.dump(selector, open(self.dir_path + "/selector.sav", 'wb'))
        pickle.dump(balancer, open(self.dir_path + "/balancer.sav", 'wb'))
        pickle.dump(classifier, open(self.dir_path + "/classifier.sav", 'wb'))


        # Prediction on another test set
        print("\n\nPrediction on second test set")
        self.dataset_to_predict = scaler.fit_transform(self.dataset_to_predict)
        self.dataset_to_predict = selector.transform(self.dataset_to_predict)

        predictions = classifier.predict(self.dataset_to_predict)
        complete_dataset = self.dataset_to_predict
        np.insert(complete_dataset, 6, predictions, axis=1)
        savetxt('generated_dataset.csv', complete_dataset, delimiter=',')

        # # Hyperparameters optimization
        # model = HP_Optimization().hp_optimization(self.data['Hyper-parameters Optimization'])
        #
        # # Model explanation
        # Explainability().explainability(model, x_training, y_training, x_testing, y_testing, prediction, classifier,
        #                               self.data['Explaination Method'])
