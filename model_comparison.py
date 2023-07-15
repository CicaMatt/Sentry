import pickle

import pandas as pd

from components import statistical_tests
from components.validation import Validation


def main():
    print("STEFFEN!")
    data = pd.read_csv('dataset_pango.csv')
    filename_column = data[data.columns[0]]
    data = data.drop(columns=data.columns[0])
    labels = (pd.get_dummies(data['vulnerable'], prefix='vulnerable'))

    x_training, x_testing, y_training, y_testing = Validation().data_validation(data, "ttsplit")

    first_model = pickle.load(open('pango/configuration0/classifier.sav', 'rb'))
    first_scaler = pickle.load(open('pango/configuration0/scaler.sav', 'rb'))
    first_selector = pickle.load(open('pango/configuration0/selector.sav', 'rb'))
    first_prediction = pd.read_csv('pango/configuration0/prediction.csv')

    second_model = pickle.load(open('pango/configuration1/classifier.sav', 'rb'))
    second_scaler = pickle.load(open('pango/configuration1/scaler.sav', 'rb'))
    second_selector = pickle.load(open('pango/configuration1/selector.sav', 'rb'))
    second_prediction = pd.read_csv('pango/configuration1/prediction.csv')

    x_training = first_scaler.fit_transform(x_training)
    x_testing = first_scaler.transform(x_testing)

    x_training = first_selector.fit_transform(x_training, labels)
    x_testing = first_selector.transform(x_testing)

    # su questo si deve applicare il preprocessing prima
    statistical_tests.StatisticalTests.two_proportions_test(x_training, x_testing, y_training, y_testing, first_model,
                                                            second_model)

    # bisogna salvare anche le predizioni nel configuration
    statistical_tests.StatisticalTests.mcnemar_test(y_testing, first_prediction, second_prediction)

    statistical_tests.StatisticalTests.cv_paired_test(data.values, labels.values, first_model, second_model)


if __name__ == "__main__":
    main()
