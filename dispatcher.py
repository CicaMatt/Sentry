from components.data_balancing import Balancing
from components.data_cleaning import Cleaning
from components.explainability import Explainability
from components.hp_optimization import HP_Optimization
from components.feature_scaling import Scaling
from components.feature_selection import Selection
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

        # Feature Selection
        data = Selection.selection(data, self.data['Feature Selection'])

        # Data Balancing
        data = Balancing.dataBalancing(data, self.data['Data Balancing'])

        # Validation setup
        x_training, x_testing, y_training, y_testing = Validation.data_validation(data, labels_full, self.data['Validation'])

        # Hyperparameters optimization
        model = HP_Optimization.hp_optimization(self.data['Hyper-parameters Optimization'])

        # Model classification
        prediction, classifier = Classification.data_classification(x_training, x_testing, y_training, y_testing, self.data['Classifier'])

        #dobbiamo salvare il modello
        self.classifier = classifier

        # Metrics calculation
        Metrics.metrics(y_testing, prediction, self.data['Metric'], 0)

        # Model explanation
        Explainability.explainability(model, x_training, y_training, x_testing, y_testing, self.data['Explaination Method'])
