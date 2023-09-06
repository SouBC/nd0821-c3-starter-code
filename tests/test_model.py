import pytest
from sklearn.model_selection import train_test_split
from starter.train_model import run_predict
import pandas as pd
import joblib
from starter.ml.data import process_data
from starter.ml.model import inference

def test_run_predict(sample_input_data):
    train, test = train_test_split(sample_input_data, test_size=0.20, random_state=0)
    test = test.drop('salary', axis = 1)
    preds, preds_class = run_predict(test, "model/model_lr.pkl")
    assert preds.shape[0] == test.shape[0]
    assert preds[0] == 0

def test_inference(sample_input_data):
    (model,encoder, lb, cat_features) = joblib.load('model/model_rf.pkl')
    
    input_data_np, y, encoder, lb = process_data(
        sample_input_data, categorical_features=cat_features,
        training=False, encoder=encoder, lb=lb, label = 'salary')
    
    preds = inference(model, input_data_np) 
    assert preds.shape[0] == sample_input_data.shape[0]
    
    

