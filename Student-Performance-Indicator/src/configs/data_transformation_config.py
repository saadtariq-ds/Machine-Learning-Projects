""" This module contains the configuration of data transformation """
import os
from dataclasses import dataclass

@dataclass()
class DataTransformationConfig:
    preprocessing_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')
