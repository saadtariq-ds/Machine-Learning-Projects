""" This file contains the common functions """
import os
import sys
import pandas as pd
import numpy as np
import dill
from src.exception import CustomException



def save_object(file_path, file_object):
    try:
        directory_path = os.path.dirname(file_path)
        os.makedirs(directory_path, exist_ok=True)

        with open(file=file_path, mode='wb') as file:
            dill.dump(file_object, file)

    except Exception as e:
        raise CustomException(e, sys)