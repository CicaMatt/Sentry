import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


class Cleaning:
    def cleaning(self, data, method):
        print("Data preparation...")
        #
        # if method == "dataimputation" or method =="default":
        #     df_mode = data.copy()
        #     mode_imputer = SimpleImputer(strategy='most_frequent')
        #     df_mode['MaxSpeed'] = mode_imputer.fit_transform(df_mode['MaxSpeed'].values.reshape(-1, 1))
        if method == "shuffling":
            # Shuffling rows of dataframe, done due to consecutive dataset entry being similar to each other
            sampler = np.random.permutation(len(data))
            # Indexing data according to sampler indexes
            data = data.take(sampler)
        else:
            # Removing all duplicates
            data = data.drop_duplicates()

        return data
