import os
import mlflow
import numpy as np


def config_mlflow():
    MLFLOW_TRACKING_URI = 'https://dagshub.com/renansantosmendes/teste.mlflow'
    MLFLOW_TRACKING_USERNAME = 'renansantosmendes'
    MLFLOW_TRACKING_PASSWORD = '6d730ef4a90b1caf28fbb01e5748f0874fda6077'
    os.environ['MLFLOW_TRACKING_USERNAME'] = MLFLOW_TRACKING_USERNAME
    os.environ['MLFLOW_TRACKING_PASSWORD'] = MLFLOW_TRACKING_PASSWORD

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    return client


def load_model(client):
    registered_model = client.get_registered_model('fetal_health_model_v1')
    run_id = registered_model.latest_versions[-1].run_id
    logged_model = f'runs:/{run_id}/model'
    loaded_model = mlflow.pyfunc.load_model_from_registry(logged_model)
    return loaded_model


if __name__ == "__main__":
    model = load_model(config_mlflow())
    prediction = model.predict(data=np.array([[-0.5, -0.5, -0.5, -0.5]]))
    print(prediction)