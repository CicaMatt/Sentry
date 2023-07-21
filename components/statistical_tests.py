import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from mlxtend.evaluate import proportion_difference, mcnemar_table, mcnemar, paired_ttest_5x2cv, lift_score


class StatisticalTests:

    # Compare the difference between two proportions or frequencies in two different groups or conditions,
    # to determine whether there is a significant difference between the proportions of success or failure in
    # two independent samples, based on the null hypothesis that there is no significant difference between
    # the two proportions and the alternative hypothesis that the two proportions are different
    def two_proportions_test(self, x_training, y_training, x_test, y_test, first_model, second_model):
        # First we fit the classification algorithms
        first_model.fit(x_training, y_training)
        second_model.fit(x_training, y_training)
        # Generate the predictions
        rf_y = first_model.predict(x_test)
        knn_y = second_model.predict(x_test)
        # Calculate the accuracy
        acc1 = accuracy_score(y_test, rf_y)
        acc2 = accuracy_score(y_test, knn_y)
        # Run the test
        print("Proportions Z-Test")
        z, p = proportion_difference(acc1, acc2, n_1=len(y_test))
        print(f"z statistic: {z}, p-value: {p}\n")

    # Compare the difference between two proportions or frequencies between two conditions or related treatments,
    # to determine if there is a significant difference between two classifiers
    def mcnemar_test(self, y_test, first_prediction, second_prediction):
        print("McNemar's test")
        table = mcnemar_table(y_target=y_test, y_model1=first_prediction, y_model2=second_prediction)
        chi2_, p = mcnemar(ary=table, corrected=True)
        print(f"chi² statistic: {chi2_}, p-value: {p}\n")

    # Scoring function to compute the LIFT metric, the ratio of correctly predicted positive examples
    # and the actual positive examples in the test dataset.
    def lift_score_test(self, y_test, prediction):
        print("Lift Score test")
        score = lift_score(y_test, prediction, binary=True, positive_label=1)
        print("Score: " + str(score))
