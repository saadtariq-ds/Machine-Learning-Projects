""" This module contains the complete pipeline of ingestion, transformation and training """
import os
import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging


def main():
    """
    This function contains the complete pipeline of ingestion, transformation and training
    """
    try:
        # Step: Data Ingestion
        logging.info("Data Ingestion Started")
        data_ingestion = DataIngestion()
        training_path, test_path = data_ingestion.initiate_data_ingestion()
        logging.info("Data Ingestion Completed")

        # Step: Data Transformation
        logging.info("Data Transformation Started")
        data_transformation = DataTransformation()
        training_data, test_data, preprocessor = data_transformation.initiate_data_transformation(
            train_path=training_path, test_path=test_path
        )
        logging.info("Data Transformation Completed")

        # Step: Model Training
        logging.info("Model Training Started")
        model_training = ModelTrainer()
        model_name, score = model_training.initiate_model_trainer(
            training_data=training_data, test_data=test_data)
        logging.info("Model Training Completed")
        logging.info("Best model `%s` has the r2 score: %s", model_name, score)


    except Exception as e:
        raise CustomException(e, sys)

if __name__ == "__main__":
    main()