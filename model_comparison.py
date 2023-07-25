import os
import pickle
import warnings

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import shapiro

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

        # Data setup
        data = Setup().data_setup(self.dataset, training=True)

        # Data Cleaning
        data, filename_column = Cleaning().cleaning(data, self.configuration['Data Cleaning'])
        columns = data.columns

        print("Shapiro-Wilk test:", shapiro(data))

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
        # first_prediction = pd.read_csv(self.path1 + '/prediction.csv')

        second_model = pickle.load(open(self.path2 + '/classifier.sav', 'rb'))
        # second_prediction = pd.read_csv(self.path2 + '/prediction.csv')

        first_prediction = first_model.predict(x_testing)
        second_prediction = second_model.predict(x_testing)

        StatisticalTests.two_proportions_test(self=self, x_training=x_training, y_training=y_training, x_test=x_testing,
                                              y_test=y_testing, first_model=first_model, second_model=second_model)

        pd.DataFrame(y_testing).to_csv(self.path1 + "/y_testing_new.csv")
        y_testing = pd.read_csv(self.path1 + '/y_testing_new.csv').to_numpy().flatten()
        os.remove(self.path1 + "/y_testing_new.csv")
        pd.DataFrame(first_prediction).to_csv(self.path1 + "/first_prediction_new.csv")
        first_prediction = pd.read_csv(self.path1 + '/first_prediction_new.csv').to_numpy().flatten()
        os.remove(self.path1 + "/first_prediction_new.csv")
        pd.DataFrame(second_prediction).to_csv(self.path1 + "/second_prediction_new.csv")
        second_prediction = pd.read_csv(self.path1 + '/second_prediction_new.csv').to_numpy().flatten()
        os.remove(self.path1 + "/second_prediction_new.csv")

        StatisticalTests.mcnemar_test(self, y_testing, first_prediction, second_prediction)

        print("First model lift score:")
        StatisticalTests.lift_score_test(self, y_testing, first_prediction)
        print("Second model lift score:")
        StatisticalTests.lift_score_test(self, y_testing, second_prediction)


