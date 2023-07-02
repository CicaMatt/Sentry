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

    def __init__(self, data, repo_link):
        self.data = data
        self.repo_link = repo_link

    def start(self):
        # Data setup
        data = Setup.data_setup()

        # Data Cleaning
        data = Cleaning.cleaning(data, self.data['Data Cleaning'])

        # Validation - Train/Test Split
        if (self.data['Validation'] == "ttsplit"):
            x_training, x_testing, y_training, y_testing = Validation.data_validation(data, self.data['Validation'])

            # Feature Scaling
            x_training, x_testing = Scaling.scaling(x_training, x_testing, self.data['Feature Scaling'])

            # Feature Selection
            x_training, x_testing = Selection.selection(x_training, x_testing, y_training, y_testing,
                                                        self.data['Feature Selection'])

            # Data Balancing
            x_training = Balancing.dataBalancing(x_training, self.data['Data Balancing'])

            # Model classification
            prediction, classifier = Classification.data_classification(x_training, x_testing, y_training, y_testing,
                                                                        self.data['Classifier'])

            # dobbiamo salvare il modello
            self.classifier = classifier

            # Metrics calculation
            Metrics.metrics(y_testing, prediction, classifier)

        # Validation - Stratified or Standard K Fold Validation
        else:
            labels_full = data["vulnerable"]
            indexes, training_data, labels = Validation.data_validation(data, labels_full, self.data['Validation'])
            best_accuracy = 0

            for training_index, testing_index in indexes:
                x_training, x_testing = training_data[training_index], training_data[testing_index]
                y_training, y_testing = labels[training_index], labels[testing_index]

                # Feature Scaling
                x_training, x_testing = Scaling.scaling(x_training, x_testing, self.data['Feature Scaling'])

                # Feature Selection
                x_training, x_testing = Selection.selection(x_training, x_testing, y_training, y_testing,
                                                            self.data['Feature Selection'])

                # Data Balancing
                x_training = Balancing.dataBalancing(x_training, self.data['Data Balancing'])

                # Model classification
                prediction, classifier = Classification.data_classification(x_training, x_testing, y_training,
                                                                            y_testing, self.data['Classifier'])

                # Metrics calculation
                accuracy = Metrics.metrics(y_testing, prediction, classifier)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    self.classifier = classifier

        # Hyperparameters optimization
        model = HP_Optimization.hp_optimization(self.data['Hyper-parameters Optimization'])

        # Model explanation
        Explainability.explainability(model, x_training, y_training, x_testing, y_testing, prediction, classifier,
                                      self.data['Explaination Method'])
