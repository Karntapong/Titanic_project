from Titanic_classifier.components.data_transformation import DataTransformation
from Titanic_classifier import logging
from Titanic_classifier.config.configuration import ConfigurationManager

class DatatransformationPipeline:
    def __init__(self):
        pass
    def main(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        data_transformation.convert()