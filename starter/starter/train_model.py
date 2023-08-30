# Script to train machine learning model.

from sklearn.model_selection import train_test_split
import pandas as pd
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference
from sklearn.metrics import accuracy_score

def run_training():

    # Add the necessary imports for the starter code.

    # Add code to load in the data.
    data = pd.read_csv('data/census.csv')

    # Remove spaces from columns names
    data.columns = [colname.replace(' ', '') for colname in data.columns]

    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    train, test = train_test_split(data, test_size=0.20, random_state=0)

    print("test.shape : {}".format(test.shape))
    print(test.head(1))

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

    X_test, y_test, encoder, lb = process_data(
        test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )
    print("test.shape : {}".format(X_test.shape))

    # Train and save a model.

    model = train_model(X_train, y_train)

    preds = inference(model, X_test)
    print(preds[0])
    print(y_test[0])
    precision, recall, fbeta = compute_model_metrics(y_test, preds)

    print("precision : {}".format(precision))
    print("recall: {}".format(recall))
    print("fbeta : {}".format(fbeta))
    print("accuracy metric: {}".format(accuracy_score(y_test, preds)))

def run_predict(df, encoder, lb, model):

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
        df, categorical_features=cat_features, label="salary", 
        training=False, encoder=encoder, lb=lb)
    

if __name__ == "__main__":
    run_training()
