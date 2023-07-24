import os
import re
import shutil
import sys
import pandas as pd
import requests
import yaml

from components import dataset_generation
from components.metrics import Metrics
from components.setup import Setup
from dispatcher import Dispatcher
from YAMLFileFormatException import YAMLFileFormatException
from model_comparison import Comparer


def verifica_link_github(link):
    try:
        response = requests.get(link)
        if response.status_code == 200:
            return 1
        else:
            print("Link does not exists")
            return 0
    except requests.exceptions.RequestException as e:
        print("Error during request: ", e)
        return 0


def main():
    args = sys.argv[1:]
    if args.__len__() != 1:
        print("Only path to YAML file needed", file=sys.stderr)
        while True:
            pass

    with open(args[0]) as f:
        data = yaml.full_load(f)
    path_training = None

    #così si accede ai singoli elementi della configurazione
    #print(data['configurations'][0][0]['Classifier'])
    try:
        if "dataset" not in data or data["dataset"].lower() == "default":
            data["dataset"] = "dataset.csv"
        elif not os.path.exists(data["dataset"]):
            raise YAMLFileFormatException("Path to the dataset was not found")
        else:
            path_training = data["dataset"]

        if not verifica_link_github(data["repo"]):
            raise YAMLFileFormatException("Wrong Repository path")

        repo_link = data["repo"]

        # controlli relativi al yaml file per ogni parametro letto per vedere se ci sono input errati quindi dare
        # un messaggio d'errore. Se il parametro non viene inserito dall'utente viene aggiunto con valore default
        # la copia serve per iterare mentre si aggiungono gli elementi default all'originale
        copy = dict(data)
        n = 0
        for pipeline in copy['configurations']:
            for i in pipeline:
                if i != n:
                    raise YAMLFileFormatException("Order of configurations is wrong, enter numbers from 0 to N")
                n += 1
                if not str(i).isdigit():
                    raise YAMLFileFormatException("Insert int number for configurations")
                # Data Cleaning
                if not "Data Cleaning" in pipeline[i]:
                    data['configurations'][i][i]["Data Cleaning"] = "default"
                else:
                    cleaning = pipeline[i]["Data Cleaning"].lower()
                    if not ("dataimputation" in cleaning or "shuffling" in cleaning or "duplicatesremoval" in cleaning):
                        raise YAMLFileFormatException("Wrong Data Cleaning input inserted")
                # Feature Scaling
                if not "Feature Scaling" in pipeline[i]:
                    data['configurations'][i][i]["Feature Scaling"] = "default"
                else:
                    scaling = pipeline[i]["Feature Scaling"].lower()
                    if not (scaling == "zscore" or scaling == "minmax"):
                        raise YAMLFileFormatException("Wrong Feature Scaling input inserted")
                # Feature Selection
                if not "Feature Selection" in pipeline[i]:
                    data['configurations'][i][i]["Feature Selection"] = "default"
                else:
                    selection = pipeline[i]["Feature Selection"].lower()
                    if not (selection == "kbest" or selection == "variancethreshold" or selection == "pearsoncorrelation"):
                        raise YAMLFileFormatException("Wrong Feature Selection input inserted")

                    if selection == "kbest":
                        if "K" not in pipeline[i]:
                            raise YAMLFileFormatException("Specify the desired number of features via the parameter K")
                        else:
                            stringa = str(pipeline[i]["K"]).strip()
                            if not stringa.isdigit():
                                raise YAMLFileFormatException("K must be an integer")
                # Data Balancing
                if not "Data Balancing" in pipeline[i]:
                    data['configurations'][i][i]["Data Balancing"] = "default"
                else:
                    balancing = pipeline[i]["Data Balancing"].lower()
                    if not (balancing == "smote" or balancing == "nearmiss" or balancing == "undersampling" or balancing == "oversampling"):
                        raise YAMLFileFormatException("Wrong Data Balancing input inserted")
                # Classification
                if not "Classifier" in pipeline[i]:
                    data['configurations'][i][i]["Classifier"] = "default"
                else:
                    classifier = pipeline[i]["Classifier"].lower()
                    if not (classifier == "svm" or classifier == "randomforest" or classifier == "kneighbors"):
                        raise YAMLFileFormatException("Wrong Classifier input inserted")
                # Validation
                if not "Validation" in pipeline[i]:
                    data['configurations'][i][i]["Validation"] = "default"
                else:
                    validation = pipeline[i]["Validation"].lower()
                    if not (validation == "ttsplit" or validation == "kfold" or validation == "stratifiedfold"):
                        raise YAMLFileFormatException("Wrong Validation input inserted")
                # Explainability
                if not "Explaination Method" in pipeline[i]:
                    data['configurations'][i][i]["Explaination Method"] = "default"
                else:
                    explain = pipeline[i]["Explaination Method"].lower()
                    if not ("confusionmatrix" in explain or "permutation" in explain or "partialdependence" in explain):
                        raise YAMLFileFormatException("Wrong Explaination Method input inserted")

        # Generating dataset from repository link
        dataset_generation.start(repo_link=repo_link)
        dataset = "generated_dataset.csv"
        to_predict = Setup().data_setup(dataset, training=False)
        # vulnerable = to_predict["vulnerable"]
        # to_predict = to_predict.drop(columns=["vulnerable"])

        root = repo_link.rsplit('/', 1)[-1]
        if os.path.exists(root):
            shutil.rmtree(root)
        os.mkdir(root)
        for pipeline in data['configurations']:
            for i in pipeline:
                print("Configuration n." + str(i))
                path = root + './configuration'
                path += str(i)
                os.mkdir(path)
                dispatcher = Dispatcher(data['configurations'][i][i], path, to_predict, path_training)
                dispatcher.start()

                # predicted = pd.read_csv(path + "/generated_dataset.csv")
                # y_predicted = predicted["vulnerable"]
                # print("Prediction metrics:")
                # Metrics().metrics(vulnerable, y_predicted)
                print("\n\n------------------------------------------------------------\n\n")

        if len(copy['configurations']) < 2 and 'statistical test' in copy:
            raise YAMLFileFormatException("There must be at least two configurations to compare them")
        elif 'statistical test' in copy:
            n = 0
            for comparison in copy['statistical test']:
                for i in comparison:
                    if i != n:
                        raise YAMLFileFormatException("Order of tests is wrong, enter numbers from 0 to N")
                    n += 1
                    if not str(i).isdigit():
                        raise YAMLFileFormatException("Index must be an integer")
                    pattern = r'^\d+,\s*\d+$'
                    match = re.match(pattern, data['statistical test'][i][i])
                    if match:
                        numbers = match.group().split(',')
                        num1, num2 = int(numbers[0]), int(numbers[1])
                        if num1 == num2:
                            raise YAMLFileFormatException("You have to compare two different configurations")
                        if num1 > len(copy['configurations'])-1 or num2 > len(copy['configurations'])-1:
                            raise YAMLFileFormatException("Input entered not matching the configuration number")
                    else:
                        raise YAMLFileFormatException("The string entered does not match the format: n,n")

                    # Statistical tests
                    print("Statistical tests")
                    path1 = root + "/configuration" + str(num1)
                    path2 = root + "/configuration" + str(num2)
                    comparer = Comparer(data['configurations'][i][i], path_training, path1, path2)
                    # comparer = Comparer(data['configurations'][i][i], "/generated_dataset.csv", path1, path2)
                    comparer.start()

    except Exception as e:
        print(e, file=sys.stderr)
        while True:
            pass

    while True:
        pass


if __name__ == "__main__":
    main()



