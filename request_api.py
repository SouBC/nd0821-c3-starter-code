"""Test Live app from heroku"""
import json
import requests

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

r = requests.post("https://fastapi-chapter3-51ba8448d756.herokuapp.com/model/",
                  data=json.dumps(sample),
                  timeout=5)

print(f"Result of model inference : {r.json()}")
print(f"Status code returned : {r.status_code}")
