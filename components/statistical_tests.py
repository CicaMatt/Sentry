from sklearn.metrics import accuracy_score
from mlxtend.evaluate import proportion_difference, mcnemar_table, mcnemar, paired_ttest_5x2cv


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


    # Compare the performance of two models using a configuration of 5-fold cross-validation (CV), where at each
    # iteration of cross-validation it calculates a 't' statistic to assess whether the difference is statistically
    # significant, taking into account the dependence between coupled measures. The result is a value of p (p-value)
    # that indicates the probability of obtaining a difference in the performance of the models at least that
    # observed, assuming that there is no real difference between the models.
    # If the p-value is below a predefined significance threshold (usually 0.05), it is concluded that there
    # is a significant difference between the performance of the models.
    def cv_paired_test(self, training_data, labels, first_model, second_model):
        print("5x2 CV Paired t-test")
        t, p = paired_ttest_5x2cv(estimator1=first_model, estimator2=second_model, X=training_data, y=labels,
                                  random_seed=42)
        print(f"t statistic: {t}, p-value: {p}\n")
