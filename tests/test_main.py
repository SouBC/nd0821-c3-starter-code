"""Local test of the api"""
from fastapi.testclient import TestClient
import pandas as pd
from sklearn.model_selection import train_test_split
from main import app

client = TestClient(app)


def test_api_get_root():
    """Test of the api method GET (root)"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.text == "<html><body style='padding: 10px;'><h1>Welcome to the API</h1><div>Check the docs: <a href='/docs'>here</a></div></body></html>"


def test_make_prediction() -> None:
    """Test of the api method post (model inference)"""
    test_data = pd.read_csv('data/census.csv')
    train, test = train_test_split(test_data, test_size=0.20, random_state=0)

    test = test.iloc[4]

    payload = test.to_dict()

    # When
    response = client.post(
        "http://localhost:8000/model/",
        json=payload,
    )
    print(response.json())
    # Then
    assert response.status_code == 200
    prediction_data = response.json()

    assert prediction_data["expected_salary"]
    # assert math.isclose(prediction_data["predictions"][0], 113422, rel_tol=100)


def test_run_predict_truepositive():
    """Test reponse api - case : True positive"""

    sample = {
        "age": 45,
        "workclass": "Private",
        "fnlgt": 172822,
        "education": "11th",
        "education-num": 7,
        "marital-status": "Divorced",
        "occupation": "Transport-moving",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 2824,
        "hours-per-week": 76,
        "native-country": "United-States"
    }
    response = client.post(
        "http://localhost:8000/model/",
        json=sample,
    )
    # Then
    assert response.status_code == 200
    prediction_data = response.json()

    assert prediction_data["expected_salary"] == ">50K"


def test_run_predict_truenegative():
    """Test reponse api - case : True Negative"""
    sample = {
        "age": 27,
        "workclass": "Private",
        "fnlgt": 177119,
        "education": "Some-college",
        "education-num": 10,
        "marital-status": "Divorced",
        "occupation": "Adm-clerical",
        "relationship": "Unmarried",
        "race": "White",
        "sex": "Female",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 44,
        "native-country": "United-States"
    }
    response = client.post(
        "http://localhost:8000/model/",
        json=sample,
    )
    print(response.json())
    # Then
    assert response.status_code == 200
    prediction_data = response.json()

    assert prediction_data["expected_salary"] == "<=50K"
