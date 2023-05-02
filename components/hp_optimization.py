from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


class HP_Optimization:
    def hp_optimization(self, model):
        # da cambiare con il modello dato in input
        model = LogisticRegression()

        # grid_vals e param_vals sono i parametri che scegliamo di ottimizzare
        grid_vals = {'penalty': ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1]}
        opt_model = GridSearchCV(estimator=model, param_grid=grid_vals, scoring='accuracy',
                                 cv=6, refit=True, return_train_score=True)

        #param_vals = {'max_depth': [200, 500, 800, 1100], 'n_estimators': [100, 200, 300, 400],
        #              'learning_rate': [0.001, 0.01, 0.1, 1, 10]]}
        #random_rf = RandomizedSearchCV(estimator=model, param_distributions=param_vals,
        #n_iter = 10, scoring = 'accuracy', cv = 5,
        #refit = True, n_jobs = -1)

        return opt_model
