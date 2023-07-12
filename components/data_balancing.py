import numpy as np
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import NearMiss, RandomUnderSampler


class Balancing:
    def dataBalancing(self, x_training, labels, method):
        # features = np.delete(x_training, 5, 1)
        # features = x_training.iloc[:, :-1]
        # labels = x_training.iloc[:, -1]

        if method == "default" or method == "smote":
            balancer = SMOTE()
            x_training_bal, y_training_bal = balancer.fit_resample(x_training, labels)

        elif method == "nearmiss":
            balancer = NearMiss()
            x_training_bal, y_training_bal = balancer.fit_resample(x_training, labels)

        elif method == "oversampling":
            balancer = RandomOverSampler(random_state=42)
            x_training_bal, y_training_bal = balancer.fit_resample(x_training, labels)

        else:
            balancer = RandomUnderSampler(random_state=42)
            x_training_bal, y_training_bal = balancer.fit_resample(x_training, labels)

        return x_training_bal, y_training_bal

