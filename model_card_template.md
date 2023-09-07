# Model Card - Salary Classification

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
- Developped for an Udacity Course.
- Random Forest Classifier.
- Binary classification trained to predict 2 classes of incomes for an individual based on multiple demographic features.


## Intended Use
- Intended to be used to compare inequalities of incomes accross demographic caracteristics such as sex, race, relationship status, education...

- The dataset is not an honest representation of the demographic picture of a typical society. This dataset used to train the model is imbalanced, only 25% have an income above 50, it is also the case for multiple features (sex, race, occupation).


## Training Data

- Publicly available Census Bureau data containing 32561 rows, 14 features and the target variable (salary). The target variable can take 2 classes : >50K or <=50K.

## Evaluation Data

- Random split of the training data (size = 20%).

## Metrics
4 metrics were used to evaluate the performance of the model :

- fbeta_score
- precision_score
- recall_precision
- accuracy_score

The current model performs relatively well for a basic ML model without hyperparameter tuning, here are the metrics 

- precision : 0.7272
- recall: 0.6200
- fbeta : 0.6693
- accuracy metric: 0.8499

## Ethical Considerations

## Caveats and Recommendations
- Include more features.
- Handling the imbalanced proportions of the target variabke with a oversampling or undersampling approch used to cure those kind of problems.
- Experiment with other ML models : xgboost for instance that tend to have more accurate predictions that traditional algorithms.

