import sys

import pandas as pd

from DatasetFormatException import DatasetFormatException


class Setup:

    def data_setup(self, filename):
        try:
            print("Reading file...")
            data = pd.read_csv(filename)
            allowed_extensions = ('.c', '.py', '.java')
            data = data[data.iloc[:, 0].str.endswith(allowed_extensions)]
            if data.iloc[:, 0].count() == 0:
                raise DatasetFormatException("Provide files with extension: '.c', '.py', '.java' in the first column")
            data = data.drop(columns=data.columns[0])
            #testa se nell'ultima colonna del dataset ci sono solo 0 o 1
            if not data[data.columns[len(data.columns)-1]].isin([0, 1]).all():
                raise DatasetFormatException("Provide dataset with the last column having values either 0 or 1")
            count_zeros = data.eq(0).sum(axis=1)

            # Filtra le righe che hanno meno o uguale a 3 valori zero
            df_filtered = data[count_zeros <= 3]
        except Exception as e:
            sys.exit(e.args[0])

        return df_filtered
