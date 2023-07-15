import numpy as np
from sklearn.impute import SimpleImputer


class Cleaning:
    def cleaning(self, data, method):
        print("Data preparation...")
        if method == "dataimputation" or method =="default":
            for column in data:
                data[column] = data[column].fillna(data[column].mean())
        elif method == "shuffling":
            # Shuffling rows of dataframe, done due to consecutive dataset entry being similar to each other
            sampler = np.random.permutation(len(data))
            # Indexing data according to sampler indexes
            data = data.take(sampler)
        else:
            # Removing all duplicates
            data = data.drop_duplicates()

        return data
