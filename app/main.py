import os
import json
import mlflow
import uvicorn
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel


class Data(BaseModel):
    accelerations: float
    fetal_movement: float
    uterine_contractions: float
    severe_decelerations: float


app = FastAPI()


def load_model_from_registry():
    MLFLOW_TRACKING_URI = 'https://dagshub.com/renansantosmendes/teste.mlflow'
    MLFLOW_TRACKING_USERNAME = 'renansantosmendes'
    MLFLOW_TRACKING_PASSWORD = '6d730ef4a90b1caf28fbb01e5748f0874fda6077'
    os.environ['MLFLOW_TRACKING_USERNAME'] = MLFLOW_TRACKING_USERNAME
    os.environ['MLFLOW_TRACKING_PASSWORD'] = MLFLOW_TRACKING_PASSWORD

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    registered_model = client.get_registered_model('fetal_health_model')
    run_id = registered_model.latest_versions[-1].run_id
    logged_model = f'runs:/{run_id}/model'
    loaded_model = mlflow.pyfunc.load_model(logged_model)
    return loaded_model


@app.on_event("startup")
def startup_event():
    global model
    model = load_model_from_registry()

@app.get("/")
def home():
    return {"Message": "Healthy"}


@app.post('/predict')
def predict(request: Data):
    global model
    print(request)
    received_data = np.array([
        request.accelerations,
        request.fetal_movement,
        request.uterine_contractions,
        request.severe_decelerations,
    ]).reshape(1, -1)
    print(received_data)
    prediction = model.predict(data=received_data)[0]
    print(prediction)
    return json.dumps({'prediction': np.argmax(prediction[0])}, default=str)
