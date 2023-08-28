# Script to train machine learning model.

from sklearn.model_selection import train_test_split
import pandas as pd
from ml.data import process_data
from ml.model import train_model, compute_model_metrics
from sklearn.metrics import accuracy_score

def run_training():

    # Add the necessary imports for the starter code.

    # Add code to load in the data.
    data = pd.read_csv('data/census.csv')

    # Remove spaces from columns names
    data.columns = [colname.replace(' ', '') for colname in data.columns]

    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    train, test = train_test_split(data, test_size=0.20)

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

    # Proces the test data with the process_data function.

    # Train and save a model.

    model = train_model(X_train, y_train)

    preds = model.predict(X_train)

    precision, recall, fbeta = compute_model_metrics(y_train, preds)

    print("precision : {}".format(precision))
    print("recall: {}".format(recall))
    print("fbeta : {}".format(fbeta))
    print("accuracy metric: {}".format(accuracy_score(y_train, preds)))


if __name__ == "__main__":
    run_training()