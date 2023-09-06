from typing import Any
from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.responses import HTMLResponse
import pandas as pd
from fastapi.encoders import jsonable_encoder
import starter.train_model as starter

import os

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

class PredictionResults(BaseModel):
    expected: dict


class CensusDataInputSchema(BaseModel):
    age: int = Field(examples=[27])
    workclass: str = Field(examples=["Private"])
    fnlgt: int = Field(examples=[177119])
    education: str = Field(examples=["Some-college"])
    education_num: int = Field(alias="education-num", examples=[10])
    marital_status: str = Field(alias="marital-status", examples=["Divorced"])
    occupation: str = Field(examples=["Adm-clerical"])
    relationship: str = Field(examples=["Unmarried"])
    race: str = Field(examples=["White"])
    sex: str = Field(examples=["Female"])
    capital_gain: int = Field(alias="capital-gain", examples=[0])
    capital_loss: int = Field(alias="capital-loss", examples=[0])
    hours_per_week: int = Field(alias="hours-per-week", examples=[44])
    native_country: str = Field(
        alias="native-country",
        examples=["United-Stated"])
    


app = FastAPI(
    title="FastAPI - Project",
    version="0.1.0")


@app.get("/")
async def index():
    # return {'Msg' : 'Hello There!'}
    """Basic HTML response."""
    body = (
        "<html>"
        "<body style='padding: 10px;'>"
        "<h1>Welcome to the API</h1>"
        "<div>"
        "Check the docs: <a href='/docs'>here</a>"
        "</div>"
        "</body>"
        "</html>"
    )

    return HTMLResponse(content=body)


@app.post("/model/", status_code=200)
async def predict(input_data: CensusDataInputSchema) -> Any:
    
    input_df = pd.DataFrame(jsonable_encoder(input_data), index=[0])

    preds, preds_classes = starter.run_predict(input_df, 'model/model_rf.pkl')

    results = {
        "expected_salary": preds_classes[0]
    }

    return results
