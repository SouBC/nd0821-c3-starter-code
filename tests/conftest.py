import pytest
import pandas as pd


@pytest.fixture()
def sample_input_data():
    return pd.read_csv('starter/data/census.csv')
