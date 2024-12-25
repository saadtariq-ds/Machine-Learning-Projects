""" This module contains the configuration of model training """

import os
from dataclasses import dataclass


@dataclass
class ModelTrainerConfig:
    trained_model_file_path:str = os.path.join("artifacts", "model.pkl")
