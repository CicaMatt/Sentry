from components.metrics import Metrics
from components.classification import Classification
from components.preprocessing import Preprocessing
from components.setup import Setup
from components.preparation import Preparation
from components.validation import Validation

# Data setup
data = Setup.data_setup("./data/NASA_MDP/csv/CM1.csv")

# Data preparation
data, labels_full = Preparation.data_preparation(data)

# Data preprocessing
data = Preprocessing.data_preprocessing(data, labels_full)

# Validation setup
x_training, x_testing, y_training, y_testing = Validation.data_validation(data, labels_full)

# Model classification
prediction = Classification.data_classification(x_training, x_testing, y_training, y_testing)

# Metrics calculation
Metrics.metrics(y_testing, prediction, "Metrics", 0)
