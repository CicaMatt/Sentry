import sys
import yaml

from dispatcher import Dispatcher


def main():
    args = sys.argv[1:]
    if args.__len__() != 1:
        sys.exit("Only path to YAML file needed")

    with open(args[0]) as f:
        data = yaml.full_load(f)

    #così si accede ai singoli elementi della configurazione
    #print(data['configurations'][0][0]['Classificator'])

    for pipeline in data['configurations']:
        for i in pipeline:
            Dispatcher(data['configurations'][i][i], data['dataset'])

    #dobbiamo aggiungere i controlli al yaml file per ogni parametro letto per vedere se ci sono input errati e dare un messaggio d'errore

    #quando finiscono tutte le chiamate facciamo test statistici e statistica descrittiva

if __name__ == "__main__":
    main()



