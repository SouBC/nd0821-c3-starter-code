# Script to train machine learning model.
import logging
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from ml.data import process_data, get_salary_class
from ml.model import train_model, compute_model_metrics, inference


def run_training(data):

    # Add the necessary imports for the starter code.

    # Add code to load in the data.

    # Remove spaces from columns names
    data.columns = [colname.replace(' ', '') for colname in data.columns]

    print(f"data.shape : {format(data.shape)}")
    # Optional enhancement, use K-fold cross validation instead of a
    # train-test split.
    train, test = train_test_split(data, test_size=0.20, random_state=0)

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    print("train.shape : {}".format(X_train.shape))

    # Proces the test data with the process_data function.

    X_test, y_test, encoder, lb = process_data(
        test, categorical_features=cat_features,
        label="salary", training=False, encoder=encoder, lb=lb)
    print("test.shape : {}".format(X_test.shape))

    # Train and save a model.

    model = train_model(X_train, y_train)

    logging.info('Save model & encoders:')

    preds = inference(model, X_test)

    precision, recall, fbeta = compute_model_metrics(y_test, preds)

    print("precision : {}".format(precision))
    print("recall: {}".format(recall))
    print("fbeta : {}".format(fbeta))
    print("accuracy metric: {}".format(accuracy_score(y_test, preds)))

    for feature in cat_features:
        
        classes = test[feature].unique()

        for cla in classes:
            row_slice = test[feature] == cla
            precision, recall, fbeta = compute_model_metrics(y_test[row_slice], model.predict(X_test[row_slice]))
            print("{} - {} precision : {}".format(feature, cla, precision))
            print("{} - {} recall : {}".format(feature, cla, recall))
            print("{} - {} fbeta : {}".format(feature, cla, fbeta))

    joblib.dump((model,encoder, lb, cat_features), "starter/model/model_lr.pkl")

    return model, encoder, lb, cat_features


def run_predict(input_data, path_model):

    (model,encoder, lb, cat_features) = joblib.load(path_model)
    input_data, y, encoder, lb = process_data(
        input_data, categorical_features=cat_features,
        training=False, encoder=encoder, lb=lb)

    preds = inference(model, input_data)
    mapping = get_salary_class(lb)
    preds = [key for key, value in mapping.items() if value == preds][0]
    return preds

    # row_slice = X_val["ConvexArea"] < 90407.3
    # print(f1_score(y_val[row_slice], lr.predict(X_val[row_slice])))


if __name__ == "__main__":
    data = pd.read_csv('starter/data/census.csv')
    model, encoder, lb, cat_features = run_training(data)

