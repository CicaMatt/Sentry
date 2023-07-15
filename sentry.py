import os
import shutil
import sys
import pandas as pd
import requests
import yaml
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
        sys.exit("Only path to YAML file needed")

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

        #controlli relativi al yaml file per ogni parametro letto per vedere se ci sono input errati quindi dare
        # un messaggio d'errore. Se il parametro non viene inserito dall'utente viene aggiunto con valore default
        #la copia serve per iterare sulla copia e aggiungere gli elementi default all'originale
        copy = dict(data)
        for pipeline in copy['configurations']:
            for i in pipeline:
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
    except Exception as e:
        sys.exit(e.args[0])

    # Generating dataset from repository link
    # Dataset_generation.start(repo_link=repo_link)
    dataset = "dataset_django.csv"
    to_predict = Setup().data_setup(dataset)
    vulnerable = to_predict["vulnerable"]
    to_predict = to_predict.drop(columns=["vulnerable"])

    root = repo_link.rsplit('/', 1)[-1]
    if os.path.exists(root):
        shutil.rmtree(root)
    os.mkdir(root)
    for pipeline in data['configurations']:
        for i in pipeline:
            path = root + './configuration'
            path += str(i)
            os.mkdir(path)
            # print(data['configurations'][i][i])
            dispatcher = Dispatcher(data['configurations'][i][i], path, to_predict, path_training)
            dispatcher.start()

    predict = pd.read_csv("generated_dataset.csv")
    y_predict = predict["vulnerable"]
    Metrics().metrics(vulnerable, y_predict)

    # Statistical tests
    print("\n\nStatistical tests")
    path1 = "pango/configuration0"
    path2 = "pango/configuration1"
    comparer = Comparer(data['configurations'][i][i], dataset, path1, path2)
    comparer.start()

if __name__ == "__main__":
    main()



