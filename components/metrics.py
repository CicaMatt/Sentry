import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics
import seaborn

class Metrics:
    def metrics(self, truth, prediction):
        # argmax restituisce gli indici dei valori massimi lungo un asse
        # la flag serve perchè la prediction di alcuni classificatori viene restituita in modo diverso

        # print(classifier.__class__.__name__)

        # vedere bene se ci vuole questo pezzo
        # if classifier.__class__.__name__ != "SVC":
        #     prediction = np.argmax(prediction, axis=1)


        accuracy_score = metrics.accuracy_score(truth, prediction)
        precision_score = metrics.precision_score(truth, prediction, average='weighted', zero_division=0)
        recall_score = metrics.recall_score(truth, prediction, average='weighted')
        f1_score = metrics.f1_score(truth, prediction, average="weighted")

        print("Accuracy: " + "{:.2%}".format(float(accuracy_score)))
        print("Precision: " + "{:.2%}".format(float(precision_score)))
        print("Recall: " + "{:.2%}".format(float(recall_score)))
        print("F1: " + "{:.2%}".format(float(f1_score)))

        return accuracy_score
