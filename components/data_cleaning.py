import numpy as np
import pandas as pd


class Cleaning:
    def cleaning(data, method):
        print("Data preparation...")
        # Removing all duplicates
        data = data.drop_duplicates()

        # Shuffling rows of dataframe, done due to consecutive dataset entry being similar to each other
        sampler = np.random.permutation(len(data))
        # Indexing data according to sampler indexes
        data = data.take(sampler)

        # Parallel dataset is created to be used to check later the belonging to a certain class
        # si crea un ulteriore dataset dove si associa 0 o 1 in relazione all'appartenza ad un csv, in base al prefisso 'type'
        # ad ogni label viene aggiunto all'inizio 'type'
        labels_full = pd.get_dummies(data['type'], prefix='type')

        # Dropping total_labels from training dataset
        data = data.drop(columns='type')
        return data, labels_full
