import numpy as np
from sklearn.impute import SimpleImputer


class Cleaning:
    def cleaning(self, data, method):
        print("Data preparation...")
        if "dataimputation" in method or method =="default":
            for column in data:
                data[column] = data[column].fillna(data[column].mean())
        if "shuffling" in method:
            # Shuffling rows of dataframe, done due to consecutive dataset entry being similar to each other
            sampler = np.random.permutation(len(data))
            # Indexing data according to sampler indexes
            data = data.take(sampler)
        if "duplicatesremoval" in method:
            # Removing all duplicates
            data = data.drop_duplicates()

        filename_column = data[data.columns[0]]
        data = data.drop(columns=data.columns[0])

        return data, filename_column
