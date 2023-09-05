from fastapi.testclient import TestClient
from fastapi.responses import HTMLResponse

from main import app
import pandas as pd

client = TestClient(app)

# def test_post_data_fail():
#     data = {"feature_1": -5, "feature_2": "test string"}
#     r = client.post("/data/", data=json.dumps(data))
#     assert r.status_code == 400

def test_api_get_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.text ==  "<html><body style='padding: 10px;'><h1>Welcome to the API</h1><div>Check the docs: <a href='/docs'>here</a></div></body></html>"

def test_make_prediction() -> None:
    # Given
    test_data = pd.read_csv('starter/data/census.csv')
    test_data = test_data.iloc[0,:]

    payload = test_data.to_dict()

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
