import time
from sklearn import neighbors


class Classification:
    def data_classification(x_training, x_testing, y_training, y_testing):
        # Training phase
        print("Training...")
        model = neighbors.KNeighborsClassifier(n_neighbors=5)

        start = time.time()
        model.fit(x_training, y_training)
        print("\nTraining time: " + str(time.time() - start)[0:7] + "s")

        # Prediction phase
        print("Prediction...")
        prediction = model.predict(x_testing)
        print("Total time: " + str(time.time() - start)[0:7] + "s\n")
        return prediction, model
