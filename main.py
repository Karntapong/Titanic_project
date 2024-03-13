from Titanic_classifier import logger
from Titanic_classifier.pipeline.stage01_data_ingestion import DataingestionTraningPipeline


STAGE_NAME = 'Data Training stage'
try:
    logger.info(f'>>>>stage {STAGE_NAME} started<<<<<<')
    data_ingestion = DataingestionTraningPipeline()
    data_ingestion.main()
    logger.info(f'{STAGE_NAME}completed')
except Exception as e:
    logger.exception(e)
    raise e