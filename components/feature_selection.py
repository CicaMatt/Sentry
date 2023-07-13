import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2


class Selection:
    def selection(self, x_training, x_testing, columns, labels, method):
        # print("Selecting best features for dataset")


        selector = None
        selected_features = None

        if method == "default":
            return selector, x_training, x_testing, selected_features
        # Variance Threshold
        if method == "variancethreshold":
            selector = VarianceThreshold()
            x_training = selector.fit_transform(x_training)
            x_testing = selector.transform(x_testing)

        # KBest
        elif method == "kbest":
            selector = SelectKBest(chi2, k=6)

            features_x_training = np.delete(x_training, 14, 1)
            features_x_testing = np.delete(x_testing, 14, 1)

            x_training = selector.fit_transform(features_x_training, labels)
            x_testing = selector.transform(features_x_testing)

        # Pearson's Correlation
        else:
            print(columns.argmax())
            x_testing = pd.DataFrame(x_testing, columns=columns)
            x_training = pd.DataFrame(x_training, columns=columns)

            # Calcola la matrice di correlazione di Pearson
            corr_matrix = np.corrcoef(x_training, rowvar=False)
            # Trova l'indice della colonna target
            target_index = x_training.columns.get_loc("vulnerable")
            # Seleziona le colonne con una correlazione superiore alla soglia specificata
            selected_features = []
            for i, corr_value in enumerate(corr_matrix[target_index]):
                if i != target_index and abs(corr_value) > 0.2:
                    selected_features.append(x_training.columns[i])
            x_training = x_training.loc[:, selected_features]
            x_testing = x_testing.loc[:, selected_features]

        return selector, x_training, x_testing, selected_features


