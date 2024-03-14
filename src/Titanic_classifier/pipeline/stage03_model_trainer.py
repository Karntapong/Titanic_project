from Titanic_classifier.components.model_trainer import ModelTrainer
from Titanic_classifier import logging
from Titanic_classifier.config.configuration import ConfigurationManager

class ModelTrainerPipeline:
    def __init__(self):
        pass
    def main(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        model_trainer_config = ModelTrainer(config=model_trainer_config)
        train,test,ac_model,accurate = model_trainer_config.train()