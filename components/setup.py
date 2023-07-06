import pandas as pd


class Setup:

    def data_setup(self):
        print("Reading file...")
        data = pd.read_csv("dataset.csv")
        data = data.drop(columns=["filename"])
        return data
