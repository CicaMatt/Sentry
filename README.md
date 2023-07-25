# Sentry: a customizable pipeline for vulnerability prediction

# Download the execution file
Click on this link to get Sentry: https://shorturl.at/flovB

<br/>

Considering the current limitation of the existing vulnerability prediction systems, our project will have the goal to develop an engineered ML pipeline for training, validating, and exporting a vulnerability prediction model which might potentially be employed within DevOps, with a particular focus on the usability, accessibility and quality of the product.


Following a possible version of configuration file:
```+yaml
---

repo: https://path/to/Github/repository
dataset: "path/to/dataset.csv"

configurations:
    - 0:
          Feature Scaling: zscore
          Data Balancing: smote
          Classifier: svm
          Validation: ttsplit
          Explaination Method: permutation
    - 1:
          Feature Scaling: minmax
          Feature Selection: kbest
          K: 9
          Data Balancing: smote
          Classifier: randomforest
          Validation: ttsplit
          Explaination Method: permutation
    - 2:
          Feature Scaling: minmax
          Feature Selection: pearsoncorrelation
          Data Balancing: oversampling
          Classifier: randomforest
          Validation: kfold
          Explaination Method: confusionmatrix

statistical test:
      - 0: 1, 0
      - 1: 1, 2
```

The possible values for each node are:
* Data Cleaning    (not exclusive)
    * dataimputation
    * shuffling
    * duplicatesremoval
* Feature Scaling
    * zscore
    * minmax
* Feature Selection
    * kbest (require K parameter)
        * K
    * variancethreshold
    * pearsoncorrelation
* Data Balancing
  * smote
  * nearmiss
  * undersampling
  * oversampling
* Classifier
  * svm
  * randomforest
  * kneighbors
* Validation
  * ttsplit
  * kfold
  * stratifiedfold
* Explaination Method (not exclusive)
  * confusionmatrix
  * permutation
  * partialdependence
