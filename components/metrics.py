import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics
import seaborn

total_labels = []


class Metrics:
    def metrics(y_testing, prediction, name, flag=0, partial=0):
        # argmax restituisce gli indici dei valori massimi lungo un asse
        # la flag serve perchè la prediction di alcuni classificatori viene restituita in modo diverso
        if flag == 0:
            prediction = np.argmax(prediction, axis=1)
        truth = np.argmax(y_testing, axis=1)
        accuracy_score = metrics.accuracy_score(truth, prediction)
        precision_score = metrics.precision_score(truth, prediction, average='weighted', zero_division=0)
        recall_score = metrics.recall_score(truth, prediction, average='weighted')
        f1_score = metrics.f1_score(truth, prediction, average="weighted")
        confusion_matrix = metrics.confusion_matrix(truth, prediction)
        multilabel_confusion_matrix = metrics.multilabel_confusion_matrix(truth, prediction)

        print("Accuracy: " + "{:.2%}".format(float(accuracy_score)))
        print("Precision: " + "{:.2%}".format(float(precision_score)))
        print("Recall: " + "{:.2%}".format(float(recall_score)))
        print("F1: " + "{:.2%}".format(float(f1_score)))

        print("\nConfusion Matrix generated")
        # print(confusion_matrix)
        print("Multilabel Confusion Matrix:")
        print(multilabel_confusion_matrix)

        # Seaborn confusion matrix
        confusion_matrix = pd.DataFrame(confusion_matrix, index=total_labels, columns=total_labels)
        plt.figure(figsize=(20, 10))
        seaborn.heatmap(confusion_matrix, annot=True)
        plt.title('Confusion Matrix ' + name)
        plt.ylabel('Truth Values')
        plt.xlabel('Predicted Values')
        plt.show()
