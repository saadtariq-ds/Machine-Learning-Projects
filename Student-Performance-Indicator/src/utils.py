""" This file contains the utility functions """
import os
import sys
import pandas as pd
import numpy as np
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from src.exception import CustomException


def save_object(file_path, file_object):
    try:
        directory_path = os.path.dirname(file_path)
        os.makedirs(directory_path, exist_ok=True)

        with open(file=file_path, mode='wb') as file:
            dill.dump(file_object, file)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]

            # Training the model
            model.fit(X_train, y_train)

            # Predicting the model
            train_prediction = model.predict(X_train)
            test_prediction = model.predict(X_test)

            # Evaluating Model Score
            train_model_score = r2_score(y_train, train_prediction)
            test_model_score = r2_score(y_test, test_prediction)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
