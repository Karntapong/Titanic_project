import os
from Titanic_classifier.constant import *
from Titanic_classifier.utils.common import read_yaml,create_directories
from Titanic_classifier import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from typing import Tuple
from pathlib import Path
import urllib.request as request
import zipfile
from Titanic_classifier.entity import DataIngestionConfig

class DataIngestion:
    def __init__(self,config: DataIngestionConfig):
        self.config = config
    def download_file(self):
        try:
            if not os.path.exists(self.config.local_data_file):
                filename, headers = request.urlretrieve(
                    url=self.config.source_URL,
                    filename=self.config.local_data_file
                )
                logging.info(f"{filename} downloaded! with following info: \n{headers}")
            else:
                logging.info(f"File already exists of size: {os.path.getsize(self.config.local_data_file)} bytes")
        except Exception as e:
            logging.error(f"An error occurred during file download: {e}")

        
    
    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
