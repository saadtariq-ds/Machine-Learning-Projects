""" This file contains the utility functions """
import os
import sys
import pandas as pd
import numpy as np
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException
from src.logger import logging


def save_object(file_path, file_object):
    try:
        directory_path = os.path.dirname(file_path)
        os.makedirs(directory_path, exist_ok=True)

        with open(file=file_path, mode='wb') as file:
            dill.dump(file_object, file)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_model(X_train, y_train, X_test, y_test, models, params):
    try:
        report = []

        for i in range(len(list(models))):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            param = params[list(models.keys())[i]]

            logging.info("Training for `%s` started", model)

            # Grid Search
            grid_search = GridSearchCV(
                estimator=model, param_grid=param, cv=3
            )
            grid_search.fit(X_train, y_train)

            model.set_params(**grid_search.best_params_)
            model.fit(X_train, y_train)

            # Predicting the model
            train_prediction = model.predict(X_train)
            test_prediction = model.predict(X_test)

            # Evaluating Model Score
            train_model_score = r2_score(y_train, train_prediction)
            test_model_score = r2_score(y_test, test_prediction)

            report.append({
                "Model Name": model_name,
                "Train R-2 Score": train_model_score,
                "Test R-2 Score": test_model_score
            })

            logging.info("Training for `%s` ended", model)

        return report

    except Exception as e:
        raise CustomException(e, sys)

def model_hyperparameters():
    """
    This function contains the hyperparameters of different models
    """
    return {
        "Linear Regression": {},
        "K-Nearest Neighbor Regressor": {},
        "Decision Tree Regressor": {
            'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
            # 'splitter':['best','random'],
            # 'max_features':['sqrt','log2'],
        },
        "Random Forest Regressor": {
            # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
            # 'max_features':['sqrt','log2',None],
            'n_estimators': [8, 16, 32, 64, 128, 256]
        },
        "AdaBoost Regressor": {
            'learning_rate': [.1, .01, 0.5, .001],
            # 'loss':['linear','square','exponential'],
            'n_estimators': [8, 16, 32, 64, 128, 256]
        },
        "Gradient Boost Regressor": {
            # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
            'learning_rate': [.1, .01, .05, .001],
            'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
            # 'criterion':['squared_error', 'friedman_mse'],
            # 'max_features':['auto','sqrt','log2'],
            'n_estimators': [8, 16, 32, 64, 128, 256]
        },
        "XGBoost Regressor": {
            'learning_rate': [.1, .01, .05, .001],
            'n_estimators': [8, 16, 32, 64, 128, 256]
        },
        "Catboost Regressor": {
            'depth': [6, 8, 10],
            'learning_rate': [0.01, 0.05, 0.1],
            'iterations': [30, 50, 100]
        },
    }