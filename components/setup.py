import pandas as pd


class Setup:

    def data_setup(self, filename):
        print("Reading file...")
        data = pd.read_csv(filename)
        data = data.drop(columns=["filename"])
        return data
