#  Project Submission

- Github repo : https://github.com/SouBC/nd0821-c3-starter-code
- CI/CD : Githhib actions
- Data (census.csv, encoders and model) are stored in a S3 bucket and versionned with dvc.
- API deployed in heroku.

## Model Card
[Link to the income model card ->](model_card_census.md)

## Model deployment

Use Live API : [Live API](https://fastapi-chapter3-51ba8448d756.herokuapp.com/)

## CICD
A github action workflow is set up, checks unit tests and linting.

## Data 
Data used to train and all the model artifacts (model & encoders) are stored in a S3 bucket and versioned with DVC.

## Model
Model used : RandomForest with the default parameters.

## API Creation

A Restful API is created in the script main.py, containg 1 get method for the root and 1 post method for model inference. 

## API Deployment
The API is deployed in heroku. See link above.