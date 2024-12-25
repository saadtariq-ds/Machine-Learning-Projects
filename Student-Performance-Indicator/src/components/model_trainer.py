""" This module contains the training code for different models """
import os
import sys
from operator import index

import pandas as pd
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, AdaBoostRegressor,
    GradientBoostingRegressor
)
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model, model_hyperparameters
from src.configs.model_trainer_config import ModelTrainerConfig



class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, training_data, test_data):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                training_data[:, :-1],
                training_data[:, -1],
                test_data[:, :-1],
                test_data[:, -1]
            )

            models = {
                "Linear Regression": LinearRegression(),
                "K-Nearest Neighbor Regressor": KNeighborsRegressor(),
                "Decision Tree Regressor": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "Gradient Boost Regressor": GradientBoostingRegressor(),
                "XGBoost Regressor": XGBRegressor(),
                "Catboost Regressor": CatBoostRegressor(verbose=False)
            }

            params = model_hyperparameters()

            model_report: list = evaluate_model(
                X_train=X_train, y_train=y_train, X_test=X_test,
                y_test=y_test, models=models, params=params
            )

            # Saving Model Report
            pd.DataFrame(model_report).to_csv(
                self.model_trainer_config.trained_model_report_path,
                index=False, header=True
            )

            best_model_entry = max(model_report, key=lambda x: x["Test R-2 Score"])
            best_model_name = best_model_entry["Model Name"]
            best_model_score = best_model_entry["Test R-2 Score"]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No Best Model Found", sys)

            logging.info("Found best model on both training and test dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                file_object=best_model
            )

            predicted = best_model.predict(X_test)

            score = r2_score(y_test, predicted)
            return best_model_name, score

        except Exception as e:
            raise CustomException(e, sys)
