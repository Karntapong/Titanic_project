import os
from Titanic_classifier.constant import *
from Titanic_classifier.utils.common import read_yaml,create_directories
from Titanic_classifier import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from typing import Tuple


@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','data.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self) -> Tuple[str, str, str]:
        logging.info('Entered the data ingestion method or component')
        try:
            df = pd.read_csv(r'research\Data\Titanic.csv')
            logging.info('Read the dataset as dataframe')
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info('Train test split initiated')
            train_set,test_set = train_test_split(df,test_size=0.2,random_state = 42)
            train_set.to_csv(self.ingestion_config.train_data_path,index= False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index =False , header = True)
            logging.info('Ingestion completed')
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                self.ingestion_config.raw_data_path
            )
        except Exception as e:
            logging.error(f'Error during data ingestion: {e}')
            return "", "", ""

if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data,df=obj.initiate_data_ingestion()
