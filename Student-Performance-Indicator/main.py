""" This module contains the complete pipeline of ingestion, transformation and training """
import os
import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
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
        training_data, test_data = data_ingestion.initiate_data_ingestion()
        logging.info("Data Ingestion Completed")

        # Step: Data Transformation
        logging.info("Data Transformation Started")
        data_transformation = DataTransformation()
        data_transformation.initiate_data_transformation(
            train_path=training_data, test_path=test_data
        )
        logging.info("Data Transformation Completed")


    except Exception as e:
        raise CustomException(e, sys)

if __name__ == "__main__":
    main()