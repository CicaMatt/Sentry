import pickle
import warnings

import numpy as np
import pandas as pd

from components.statistical_tests import StatisticalTests
from components.data_balancing import Balancing
from components.data_cleaning import Cleaning
from components.feature_scaling import Scaling
from components.feature_selection import Selection
from components.setup import Setup
from components.validation import Validation


class Comparer:
    def __init__(self, configuration, dataset, path1, path2):
        self.configuration = configuration
        self.dataset = dataset
        self.path1 = path1
        self.path2 = path2

    def start(self):
        warnings.filterwarnings('ignore')

        # x_training = pd.read_csv(self.path2 + '/x_training.csv')
        # y_training = pd.read_csv(self.path2 + '/y_training.csv').values.flatten()
        # x_testing = pd.read_csv(self.path2 + '/x_testing.csv')
        # y_testing = pd.read_csv(self.path2 + '/y_testing.csv').values.flatten()

        # Data setup
        data = Setup().data_setup(self.dataset)

        # Data Cleaning
        data, filename_column = Cleaning().cleaning(data, self.configuration['Data Cleaning'])
        columns = data.columns

        labels = (pd.get_dummies(data['vulnerable'], prefix='vulnerable')).values

        x_training, x_testing, y_training, y_testing = Validation().data_validation(data,
                                                                                    self.configuration['Validation'])

        x_training = x_training[:, :-1]
        x_testing = x_testing[:, :-1]

        scaler, x_training, x_testing = Scaling().scaling(x_training, x_testing, self.configuration['Feature Scaling'])

        if self.configuration['Feature Selection'] == "kbest":
            selector, x_training, x_testing, selected_features = Selection().selection(x_training, x_testing,
                                                                                            columns, y_training,
                                                                                            y_testing,
                                                                                            self.configuration[
                                                                                                'Feature Selection'],
                                                                                            self.configuration["K"])
        else:
            selector, x_training, x_testing, selected_features = Selection().selection(x_training,
                                                                                            x_testing,
                                                                                            columns,
                                                                                            y_training,
                                                                                            y_testing,
                                                                                            self.configuration[
                                                                                                'Feature Selection'])

        x_training = np.hstack((x_training, y_training.reshape(-1, 1)))
        x_training, y_training, balancer = Balancing().dataBalancing(x_training, y_training,
                                                                          self.configuration['Data Balancing'])
        x_training = x_training[:, :-1]

        first_model = pickle.load(open(self.path1 + '/classifier.sav', 'rb'))
        first_prediction = pd.read_csv(self.path1 + '/prediction.csv')

        second_model = pickle.load(open(self.path2 + '/classifier.sav', 'rb'))
        second_prediction = pd.read_csv(self.path2 + '/prediction.csv')

        # su questo si deve applicare il preprocessing prima
        # StatisticalTests.two_proportions_test(self=self, x_training=x_training, y_training=y_training, x_test=x_testing, y_test=y_testing, first_model=first_model, second_model=second_model)

        print(labels.shape)
        first_prediction = first_prediction.values.flatten()
        second_prediction = second_prediction.values.flatten()
        print(first_prediction.shape)
        print(second_prediction.shape)

        # bisogna salvare anche le predizioni nel configuration
        y_testing_imported = pd.read_csv(self.path2 + '/y_testing.csv').values.flatten()
        StatisticalTests.mcnemar_test(self, y_testing_imported, first_prediction, second_prediction)

        StatisticalTests.cv_paired_test(self, x_training, y_training, first_model, second_model)
