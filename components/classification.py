import time
from sklearn import neighbors, svm, ensemble


class Classification:
    def data_classification(self, x_training, x_testing, y_training, y_testing, classifier):

        if classifier == "kneighbors":
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

        elif classifier == "svm":
            print("Training...")
            model = svm.SVC(kernel='linear')

            start = time.time()
            print(x_training)
            print(y_training)
            model.fit(x_training, y_training)
            print("\nTraining time: " + str(time.time() - start)[0:7] + "s")

            # Prediction phase
            print("Prediction...")
            prediction = model.predict(x_testing)
            print("Total time: " + str(time.time() - start)[0:7] + "s\n")

        else:
            print("Training...")
            model = ensemble.RandomForestClassifier(criterion="entropy", random_state=100, min_samples_leaf=5,
                                                    warm_start=True)
            start = time.time()
            model.fit(x_training, y_training)
            print("\nTraining time: " + str(time.time() - start)[0:7] + "s")

            # Prediction phase
            print("Prediction...")
            prediction = model.predict(x_testing)
            print("Total time: " + str(time.time() - start)[0:7] + "s\n")

        return prediction, model






