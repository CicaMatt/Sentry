from components.data_balancing import Balancing
from components.data_cleaning import Cleaning
# from components.explainability import Explainability
from components.hp_optimization import HP_Optimization
from components.feature_scaling import Scaling
from components.feature_selection import Selection
from components.metrics import Metrics
from components.classification import Classification
from components.setup import Setup
from components.validation import Validation


class Dispatcher:

    def __init__(self, data, repo_link):
        self.data = data
        self.repo_link = repo_link

    def start(self):
        # Data setup
        data = Setup.data_setup()

        # Data Cleaning
        data = Cleaning.cleaning(data, self.data['Data Cleaning'])



        # Feature Scaling
        data = Scaling.scaling(data, self.data['Feature Scaling'])

        # Feature Selection
        data = Selection.selection(data, self.data['Feature Selection'])

        # Data Balancing
        data = Balancing.dataBalancing(data, self.data['Data Balancing'])

        # Hyperparameters optimization
        model = HP_Optimization.hp_optimization(self.data['Hyper-parameters Optimization'])

        # Validation setup
        if (self.data['Validation'] == "ttsplit"):
            x_training, x_testing, y_training, y_testing = Validation.data_validation(data, self.data['Validation'])

            # Model classification
            prediction, classifier = Classification.data_classification(x_training, x_testing, y_training, y_testing,
                                                                        self.data['Classifier'])

            # dobbiamo salvare il modello
            self.classifier = classifier

            # Metrics calculation
            Metrics.metrics(y_testing, prediction, self.data['Metric'], 0)

        else:
            indexes, training_data, labels = Validation.data_validation(data, labels_full, self.data['Validation'])
            best_accuracy = 0

            for training_index, testing_index in indexes:
                x_training, x_testing = training_data[training_index], training_data[testing_index]
                y_training, y_testing = labels[training_index], labels[testing_index]

                prediction, classifier = Classification.data_classification(x_training, x_testing, y_training,
                                                                            y_testing, self.data['Classifier'])

                # Metrics calculation
                accuracy = Metrics.metrics(y_testing, prediction, self.data['Metric'], 0)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    self.classifier = classifier

        # Model explanation
        Explainability.explainability(model, x_training, y_training, x_testing, y_testing,
                                      self.data['Explaination Method'])
