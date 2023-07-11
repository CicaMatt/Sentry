import pandas as pd


class Setup:

    def data_setup(self, filename):
        print("Reading file...")
        data = pd.read_csv(filename)
        data = data.drop(columns=["filename"])
        count_zeros = data.eq(0).sum(axis=1)

        # Filtra le righe che hanno meno o uguale a 3 valori zero
        df_filtered = data[count_zeros <= 3]

        return df_filtered
