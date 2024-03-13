from Titanic_classifier.config.configuration import ConfigurationManager
from Titanic_classifier.components.data_ingestion import DataIngestion
from Titanic_classifier.components.data_transformation import DataTransformation
from Titanic_classifier import logging

STAGE_NAME = 'Data Training stage'
class DataingestionTraningPipeline:
    def __init__(self):
        pass
    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()

class DatatransformationPipeline:
    def __init__(self):
        pass
    def main(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        data_transformation.convert()
