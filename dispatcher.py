from components.Balancing import Balancing
from components.Cleaning import Cleaning
from components.Scaling import Scaling
from components.Selection import Selection
from components.metrics import Metrics
from components.classification import Classification
from components.setup import Setup
from components.validation import Validation

class Dispatcher:

    def __init__(self, data, dataset):
        self.data = data
        self.datasetPath = dataset

    def start(self):

        # Data setup
        data = Setup.data_setup(self.datasetPath)

        # Data Cleaning
        data, labels_full = Cleaning.cleaning(data, self.data['Data Cleaning'])

        # Feature Scaling
        data = Scaling.scaling(data, labels_full, self.data['Feature Scaling'])

        #Feature Selection
        data = Selection.selection(data, self.data['Feature Selection'])

        #Data Balancing
        data = Balancing.dataBalancing(data, self.data['Data Balancing'])

        # Validation setup
        x_training, x_testing, y_training, y_testing = Validation.data_validation(data, labels_full)

        #Hyper-parameters Optimization


        # Model classification
        prediction, classifier = Classification.data_classification(x_training, x_testing, y_training, y_testing, self.data['Classifier'])

        #dobbiamo salvare il modello
        self.classifier = classifier

        # Metrics calculation
        Metrics.metrics(y_testing, prediction, "Metrics", 0)

        #Explaination Method