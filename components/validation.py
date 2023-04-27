from sklearn.model_selection import train_test_split


class Validation:
    def data_validation(data, labels_full):

        # .values transform data in a tabular DataFrame format into a multidimensional NumPy array

        # training data for the neural net
        training_data = data.values

        # total_labels for training
        labels = labels_full.values


        # Train/Test split - 80/20
        print("Validation - 80/20 split")
        x_training, x_testing, y_training, y_testing = train_test_split(training_data, labels, test_size=0.20, random_state=42)
        return x_training, x_testing, y_training, y_testing
