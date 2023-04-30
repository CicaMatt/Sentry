import pandas as pd


class Setup:
    def data_setup(path):
        print("Reading file...")
        data = pd.read_csv(str(path))
        print(data.head(20))
        return data
