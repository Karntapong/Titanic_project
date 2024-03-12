from Titanic_classifier.components.data_ingestion import DataIngestion,DataIngestionConfig
from Titanic_classifier import logger

STAGE_NAME = 'Data Training stage'
class DataingestionTraningPipeline:
    def __init__(self):
        pass
    def main(self):
        data_ingestion=DataIngestion()
        train_data,test_data,df=data_ingestion.initiate_data_ingestion()

if __name__ == '__main_':
    try:
        logger.info(f'>>>>stage {STAGE_NAME} started<<<<<<')
        obj = DataingestionTraningPipeline()
        obj.main()
        logger.info(f'{STAGE_NAME}completed')
    except Exception as e:
        logger.exception(e)
        raise e
