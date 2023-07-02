from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import NearMiss, RandomUnderSampler


class Balancing:
    def dataBalancing(self, x_training, method):
        features = x_training.iloc[:, :-1]
        labels = x_training.iloc[:, -1]

        if method == "default" or method == "smote":
            sm = SMOTE()
            x_training_bal = sm.fit_resample(features, labels)

        elif method == "nearmiss":
            nm = NearMiss()
            x_training_bal = nm.fit_resample(features, labels)

        elif method == "oversampling":
            ros = RandomOverSampler(random_state=42)
            x_training_bal = ros.fit_resample(features, labels)

        else:
            rus = RandomUnderSampler(random_state=42)
            x_training_bal = rus.fit_resample(features, labels)

        return x_training_bal

