# Script to train machine learning model.
import logging
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from ml.data import process_data, get_salary_class
from ml.model import train_model, compute_model_metrics, inference


def run_training(data):

    # Add code to load in the data.

    # Remove spaces from columns names
    data.columns = [colname.replace(' ', '') for colname in data.columns]

    print(f"data.shape : {format(data.shape)}")
    # Optional enhancement, use K-fold cross validation instead of a
    # train-test split.
    train, test = train_test_split(data, test_size=0.20, random_state=0)
    print(test.head())

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
    
    
    # Train and save a model.

    model = train_model(X_train, y_train)

    preds = inference(model, X_test)

    precision, recall, fbeta = compute_model_metrics(y_test, preds)

    print("precision : {}".format(precision))
    print("recall: {}".format(recall))
    print("fbeta : {}".format(fbeta))
    print("accuracy metric: {}".format(accuracy_score(y_test, preds)))

    # add return here to experiment other configurations

    with open('slice_output.txt', 'a') as f:
        for feature in cat_features:

            classes = test[feature].unique()

            for cla in classes:
                row_slice = test[feature] == cla
                precision, recall, fbeta = compute_model_metrics(y_test[row_slice], model.predict(X_test[row_slice]))
                f.write("\n{} - {} precision : {}\n".format(feature, cla, precision))
                f.write("{} - {} recall : {}\n".format(feature, cla, recall))
                f.write("{} - {} fbeta : {}\n".format(feature, cla, fbeta))
    
    test['preds'] = preds
    print(test.head())
    return model,encoder, lb, cat_features


def save_model(model,encoder, lb, cat_features):
    logging.info('Save model & encoders:')
    joblib.dump((model,encoder, lb, cat_features), "model/model_rf.pkl")

def run_predict(input_data, path_model):

    (model,encoder, lb, cat_features) = joblib.load(path_model)

    input_data_np, y, encoder, lb = process_data(
        input_data, categorical_features=cat_features,
        training=False, encoder=encoder, lb=lb)
    
    preds = inference(model, input_data_np)
    mapping = get_salary_class(lb)
    print(mapping)
    preds_class = [mapping[value] for value in preds ]
    return preds, preds_class

if __name__ == "__main__":
    data = pd.read_csv('data/census.csv')
    model,encoder, lb, cat_features = run_training(data)
    save_model(model,encoder, lb, cat_features)
    y = data["salary"]
    data = data.drop("salary", axis=1)
    preds, preds_classes = run_predict(data, "model/model_rf.pkl")

