import os
import sys
import yaml

from DatasetNotFoundException import DatasetNotFoundException
from dispatcher import Dispatcher
from YAMLFileFormatException import YAMLFileFormatException


def main():
    args = sys.argv[1:]
    if args.__len__() != 1:
        sys.exit("Only path to YAML file needed")

    with open(args[0]) as f:
        data = yaml.full_load(f)

    #così si accede ai singoli elementi della configurazione
    #print(data['configurations'][0][0]['Classifier'])
    try:
        if data["dataset"].lower() == "default":
            data["dataset"] = "path/to/dataset.csv"
        elif not os.path.exists(data["dataset"]):
            raise DatasetNotFoundException("Path to the dataset was not found")

        #controlli relativi al yaml file per ogni parametro letto per vedere se ci sono input errati quindi dare
        # un messaggio d'errore. Se il parametro non viene inserito dall'utente viene aggiunto con valore default
        #la copia serve per iterare sulla copia e aggiungere gli elementi default all'originale
        copy = dict(data)
        for pipeline in copy['configurations']:
            for i in pipeline:
                if not "Data Cleaning" in pipeline[i]:
                    data['configurations'][i][i]["Data Cleaning"] = "default"
                else:
                    cleaning = pipeline[i]["Data Cleaning"].lower()
                    if not (cleaning == "dataimputation" or cleaning == "shuffling" or cleaning == "duplicatesremoval"):
                        raise YAMLFileFormatException("Wrong Data Cleaning input inserted")
                if not "Feature Scaling" in pipeline[i]:
                    data['configurations'][i][i]["Feature Scaling"] = "default"
                else:
                    scaling = pipeline[i]["Feature Scaling"].lower()
                    if not (scaling == "zscore" or scaling == "minmax"):
                        raise YAMLFileFormatException("Wrong Feature Scaling input inserted")
                if not "Feature Selection" in pipeline[i]:
                    data['configurations'][i][i]["Feature Selection"] = "default"
                else:
                    selection = pipeline[i]["Feature Selection"].lower()
                    if not (selection == "kbest" or selection == "variancethreshold" or selection == "pearsoncorrelation"):
                        raise YAMLFileFormatException("Wrong Feature Selection input inserted")
                if not "Data Balancing" in pipeline[i]:
                    data['configurations'][i][i]["Data Balancing"] = "default"
                else:
                    balancing = pipeline[i]["Data Balancing"].lower()
                    if not (balancing == "smote" or balancing == "nearmiss" or balancing == "undersampling" or balancing == "oversampling"):
                        raise YAMLFileFormatException("Wrong Data Balancing input inserted")
                if not "Hyper-parameters Optimization" in pipeline[i]:
                    data['configurations'][i][i]["Hyper-parameters Optimization"] = "default"
                else:
                    optim = pipeline[i]["Hyper-parameters Optimization"].lower()
                    if not (optim == "randomsearch" or optim == "gridsearch" or optim == "bayessearch"):
                        raise YAMLFileFormatException("Wrong Hyper-parameters Optimization input inserted")
                if not "Classifier" in pipeline[i]:
                    data['configurations'][i][i]["Classifier"] = "default"
                else:
                    classifier = pipeline[i]["Classifier"].lower()
                    if not (classifier == "svm" or classifier == "RandomForest" or classifier == "decisiontree"):
                        raise YAMLFileFormatException("Wrong Classifier input inserted")
                if not "Validation" in pipeline[i]:
                    data['configurations'][i][i]["Validation"] = "default"
                else:
                    validation = pipeline[i]["Validation"].lower()
                    if not (validation == "ttsplit" or validation == "kfold" or validation == "nestedfold"):
                        raise YAMLFileFormatException("Wrong Validation input inserted")
                if not "Metric" in pipeline[i]:
                    data['configurations'][i][i]["Metric"] = "default"
                else:
                    metric = pipeline[i]["Metric"].lower()
                    if not (metric == "accuracy" or metric == "precision" or metric == "recall"):
                        raise YAMLFileFormatException("Wrong Metric input inserted")
                if not "Explaination Method" in pipeline[i]:
                    data['configurations'][i][i]["Data Cleaning"] = "default"
                else:
                    explain = pipeline[i]["Explaination Method"].lower()
                    if not (explain == "confusionmatrix" or explain == "permutation" or explain == "partialdependence"):
                        raise YAMLFileFormatException("Wrong Explaination Method input inserted")
    except Exception as e:
        sys.exit(e.args[0])

    for pipeline in data['configurations']:
        for i in pipeline:
            print(pipeline[i])
            #Dispatcher(data['configurations'][i][i], data['dataset'])

    #quando finiscono tutte le chiamate facciamo test statistici e statistica descrittiva

if __name__ == "__main__":
    main()



