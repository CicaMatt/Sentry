# Sentry: a customizable pipeline for vulnerability prediction

# Download the execution file
Click on this link to get Sentry: https://shorturl.at/ipx56

<br/>

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
