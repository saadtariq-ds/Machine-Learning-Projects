import os
import sys
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.configs.data_transformation_config import DataTransformationConfig



class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        This function is responsible for the Data Transformation
        """
        try:
            numerical_features = [
                "writing_score", "reading_score"
            ]
            categorical_features = [
                "gender", "race_ethnicity", "parental_level_of_education",
                "lunch", "test_preparation_course"
            ]

            numerical_features_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )
            logging.info("Numerical columns `%s` standard scaling is completed", numerical_features)

            categorical_features_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
                ]
            )
            logging.info("Categorical columns `%s` encoding is completed", categorical_features)

            preprocessor = ColumnTransformer(
                [
                    ("numerical_features_pipeline", numerical_features_pipeline, numerical_features),
                    ("categorical_features_pipeline", categorical_features_pipeline, categorical_features),
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Reading Train and Test Data Completed")

            logging.info("Obtaining preprocessing object")
            preprocessing_object = self.get_data_transformer_object()

            target_column_name = "math_score"

            input_features_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_features_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training and test dataframe")

            input_features_train = preprocessing_object.fit_transform(input_features_train_df)
            input_features_test = preprocessing_object.transform(input_features_test_df)

            train_data = np.c_[input_features_train, np.array(target_feature_train_df)]
            test_data = np.c_[input_features_test, np.array(target_feature_test_df)]

            logging.info("Saved preprocessing object")

            save_object(
                file_path=self.data_transformation_config.preprocessing_obj_file_path,
                file_object=preprocessing_object
            )

            return (
                train_data,
                test_data,
                self.data_transformation_config.preprocessing_obj_file_path,
            )


        except Exception as e:
            raise CustomException(e, sys)