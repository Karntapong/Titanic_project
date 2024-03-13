from Titanic_classifier import logger
from Titanic_classifier.pipeline.stage01_data_ingestion import DataingestionTraningPipeline
from Titanic_classifier.pipeline.stage02_data_transform import DatatransformationPipeline


STAGE_NAME = 'Data Training stage'
try:
    logger.info(f'>>>>stage {STAGE_NAME} started<<<<<<')
    data_ingestion = DataingestionTraningPipeline()
    data_ingestion.main()
    logger.info(f'{STAGE_NAME}completed')
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = 'Data Transform stage'
try:
    logger.info(f'>>>>stage {STAGE_NAME} started<<<<<<')
    data_transform = DataingestionTraningPipeline()
    data_transform.main()
    logger.info(f'{STAGE_NAME}completed')
except Exception as e:
    logger.exception(e)
    raise e