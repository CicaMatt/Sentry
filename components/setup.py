import pandas as pd


class Setup:
    def data_setup(self):
        print("Reading file...")
        data = pd.read_csv("../data/dataset.csv")
        return data
