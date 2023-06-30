from imblearn.over_sampling import SMOTE

class Balancing:
    def dataBalancing(self, data, method):

        if method == "default" or method == "smote":
            sm = SMOTE()
            features = data.iloc[:, :-1]
            labels = data.iloc[:, -1]
            train_features_bal, train_labels_bal = sm.fit_resample(features, labels)

        return data