import numpy as np
from numpy import mean
from scipy.constants import hp
from scipy.optimize import fmin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from hyperopt import tpe, STATUS_OK, Trials, hp, fmin, STATUS_OK, space_eval


class HP_Optimization:
    def svm_opt(self, model, X_train, y_train):
        C_range = np.logspace(-10, 10, 21)
        print(f'The list of values for C are {C_range}')
        # List of gamma values
        gamma_range = np.logspace(-10, 10, 21)
        print(f'The list of values for gamma are {gamma_range}')

        # Space
        space = {
            'C': hp.choice('C', C_range),
            'gamma': hp.choice('gamma', gamma_range.tolist() + ['scale', 'auto']),
            'kernel': hp.choice('kernel', ['rbf', 'poly'])
        }
        # Set up the k-fold cross-validation
        kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)

        # Objective function
        def objective(params):
            svc = SVC(**params)
            scores = cross_val_score(svc, X_train, y_train, cv=kfold, scoring='accuracy', n_jobs=-1)
            # Extract the best score
            best_score = mean(scores)
            # Loss must be minimized
            loss = - best_score
            # Dictionary with information for evaluation
            return {'loss': loss, 'params': params, 'status': STATUS_OK}

        # Trials to track progress
        bayes_trials = Trials()
        # Optimize
        best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=100, trials=bayes_trials)

    def random_forest_opt(self, model, X_train, y_train):
        # Random Forest
        param_grid = {
            'n_estimators': [25, 50, 100, 150],
            'max_features': ['sqrt', 'log2', None],
            'max_depth': [3, 6, 9],
            'max_leaf_nodes': [3, 6, 9],
        }
        grid_search = GridSearchCV(RandomForestClassifier(),
                                   param_grid=param_grid)
        grid_search.fit(X_train, y_train)
        print(grid_search.best_estimator_)

    def knn_opt(self, model, X_train, y_train):
        # K-Nearest Neighbors
        param_grid = {'n_neighbors': [5, 7, 9, 11, 13, 15],
                       'weights': ['uniform', 'distance'],
                       'metric': ['minkowski', 'euclidean', 'manhattan']}

        gs = GridSearchCV(KNeighborsClassifier(), param_grid, verbose=1, cv=3, n_jobs=-1)
        g_res = gs.fit(X_train, y_train)
        print(g_res.best_score_, g_res.best_params_)

