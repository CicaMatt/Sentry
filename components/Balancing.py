from imblearn.over_sampling import SMOTE

class Balancing:
    def dataBalancing(data, labels_full, method):
        # Data Balancing
        sm = SMOTE()
        print("TRAIN FEATURES: \n", data)
        train_features_bal, train_labels_bal = sm.fit_resample(data, labels_full)

        return data