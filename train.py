import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, InputLayer

import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import os
import git
import random
import numpy as np
import logging
import argparse
import mlflow
import json


parser = argparse.ArgumentParser(description='Pass train train_parameters')
parser.add_argument('--mlflow_tracking_uri', required=True)
parser.add_argument('--mlflow_tracking_username', required=True)
parser.add_argument('--mlflow_tracking_password', required=True)


def get_args(arguments):
    train_parameters = dict()
    try:
        train_parameters['mlflow_tracking_uri'] = arguments.mlflow_tracking_uri
        train_parameters['mlflow_tracking_username'] = arguments.mlflow_tracking_username
        train_parameters['mlflow_tracking_password'] = arguments.mlflow_tracking_password
    except KeyError as e:
        print(e)
    return train_parameters


def load_config_file():
    with open('model_config.json') as config_file:
        file_contents = config_file.read()
    print(file_contents)

    return json.loads(file_contents)


def reset_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_data(parameters):
    data = pd.read_csv(parameters.get('data_file_url'))
    X = data.drop(["fetal_health"], axis=1)
    y = data["fetal_health"]
    return X, y


def process_data(X, y):
    columns_names = list(X.columns)
    scaler = preprocessing.StandardScaler()
    X_df = scaler.fit_transform(X)
    X_df = pd.DataFrame(X_df, columns=columns_names)

    X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.3, random_state=42)

    y_train = y_train -1
    y_test = y_test - 1
    return X_train, y_train, X_test, y_test


def create_model(input_shape, train_parameters):
    reset_seeds(train_parameters.get('seed'))
    sequential_model = Sequential()
    sequential_model.add(InputLayer(input_shape=(input_shape,)))
    sequential_model.add(Dense(10, activation='relu'))
    sequential_model.add(Dense(10, activation='relu'))
    sequential_model.add(Dense(3, activation='softmax'))

    sequential_model.compile(loss=train_parameters.get('loss'),
                             optimizer=train_parameters.get('optimizer'),
                             metrics=['accuracy'])
    return sequential_model


def config_mlflow(train_parameters):
    os.environ['MLFLOW_TRACKING_USERNAME'] = train_parameters.get('mlflow_tracking_username')
    os.environ['MLFLOW_TRACKING_PASSWORD'] = train_parameters.get('mlflow_tracking_password')

    mlflow.set_tracking_uri(train_parameters.get('mlflow_tracking_uri'))

    mlflow.tensorflow.autolog(log_models=True,
                              log_input_examples=True,
                              log_model_signatures=True)


def train_model(keras_model, X_train, y_train, train_parameters):
    with mlflow.start_run(run_name=train_parameters.get('mlflow_run_name')) as run:
        keras_model.fit(X_train,
                        y_train,
                        epochs=train_parameters.get('n_epochs'),
                        validation_split=train_parameters.get('validation_split'),
                        verbose=train_parameters.get('verbose'))
    return run.info.run_id


def register_model(train_parameters, mlflow_run_id):
    run_uri = f'runs:/{mlflow_run_id}'
    mlflow.register_model(run_uri, train_parameters.get('mlflow_registry_model_name'))


if __name__ == '__main__':
    args = parser.parse_args()
    parameters = get_args(args)
    config = load_config_file()
    parameters.update(config)
    config_mlflow(parameters)
    X, y = get_data(parameters)
    X_train, y_train, X_test, y_test = process_data(X, y)
    model = create_model(X_train.shape[1], parameters)
    run_id = train_model(model, X_train, y_train, parameters)
    register_model(parameters, run_id)

