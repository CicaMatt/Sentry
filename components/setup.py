import sys

import pandas as pd

from DatasetFormatException import DatasetFormatException


class Setup:

    def data_setup(self, filename, training):
        try:
            # print("Reading file...")
            data = pd.read_csv(filename, na_values=["n/n", "na", "--", "nan", "NaN"])

            # Filtra le righe che hanno meno o uguale a 3 valori zero
            count_zeros = data.eq(0).sum(axis=1)
            data = data[count_zeros <= 3]

            allowed_extensions = ('.c', '.py', '.java')
            data = data[data.iloc[:, 0].str.endswith(allowed_extensions)]

            if data.iloc[:, 0].count() == 0:
                raise DatasetFormatException("Provide files with extension: '.c', '.py', '.java' in the first column")

            # filename_column = data[data.columns[0]]
            # data = data.drop(columns=data.columns[0])
            # testa se nell'ultima colonna del dataset ci sono solo 0 o 1
            if training:
                if not data[data.columns[len(data.columns)-1]].isin([0, 1]).all():
                    raise DatasetFormatException("Provide dataset with the last column having values either 0 or 1")

            # controlla se in tutte le feature ci sono valori interi o float
            # numeric_columns = data.select_dtypes(include=['int', 'float']).columns
            # if len(numeric_columns) != len(data.columns):
            #     raise DatasetFormatException("Provide a dataset with integer or float features as "
            #                                  "specified by the accepted format")

        except DatasetFormatException as e:
            print(e, file=sys.stderr)
            while True:
                pass

        return data
