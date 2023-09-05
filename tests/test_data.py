import pytest
from starter.ml.data import process_data, category_features
import numpy as np

def test_process_data(sample_input_data):
    
    sample_input_data.columns = [colname.replace(' ', '') for colname in sample_input_data.columns]

    data_train, label_train, encoder, lb = process_data(
        sample_input_data, 
        categorical_features=category_features, 
        label="salary", 
        training=True
    )

    assert sample_input_data.shape[1] == 15
    assert data_train.shape[1] == 108
    assert isinstance(data_train, np.ndarray)



