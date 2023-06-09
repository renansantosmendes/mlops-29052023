import os
import pytest
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer

from train import reset_seeds, get_data, process_data, create_model, config_mlflow, train_model


@pytest.fixture
def sample_data():
    data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [6, 7, 8, 9, 10],
        'fetal_health': [1, 1, 2, 3, 2]
    })
    return data


@pytest.fixture
def parameters():
    return {"data_file_url": "https://raw.githubusercontent.com/renansantosmendes/lectures-cdas-2023/master/fetal_health_reduced.csv",
            "mlflow_registry_model_name": "test_model",
            "mlflow_run_name": "test_train_pipeline",
            "n_epochs": 1,
            "loss": "sparse_categorical_crossentropy",
            "optimizer": "adam",
            "validation_split": 0.2,
            "verbose": 3,
            "seed": 42}


def test_get_data(parameters):
    X, y = get_data(parameters)

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert not X.empty
    assert not y.empty


def test_process_data(sample_data):
    X = sample_data.drop(["fetal_health"], axis=1)
    y = sample_data["fetal_health"]
    X_train, y_train, X_test, y_test = process_data(X, y)

    assert not X_train.empty
    assert not y_train.empty
    assert not X_test.empty
    assert not y_test.empty


def test_create_model(sample_data, parameters):
    X = sample_data.drop(["fetal_health"], axis=1)
    input_shape = X.shape[1]
    model = create_model(input_shape, parameters)

    assert isinstance(model, Sequential)
    assert len(model.layers) > 0
    assert model.trainable


def test_train_model(sample_data, parameters):
    X = sample_data.drop(["fetal_health"], axis=1)
    y = sample_data["fetal_health"]
    X_train, y_train, X_test, y_test = process_data(X, y)
    model = create_model(X_train.shape[1], parameters)
    train_model(model, X_train, y_train, parameters)

    assert model.history.history['loss'][-1] > 0.0
    assert model.history.history['val_loss'][-1] > 0.0

