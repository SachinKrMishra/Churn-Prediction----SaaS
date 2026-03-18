from churnprediction.components.data_ingestion import DataIngestion
from churnprediction.components.data_validation import DataValidation
from churnprediction.exception.exception import ChurnPredictionException
from churnprediction.logging.logger import logging
from churnprediction.entity.config_entity import DataIngestionConfig, DataValidationConfig
from churnprediction.entity.config_entity import TrainingPipelineConfig
import sys

if __name__ == '__main__':
    try:
        training_pipeline_config = TrainingPipelineConfig()
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config)
        logging.info("Initiate data ingestion")
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        logging.info('Data Initiation Completed')
        print(data_ingestion_artifact)
        
        datavalidationconfig = DataValidationConfig(training_pipeline_config)
        data_validation = DataValidation(data_ingestion_artifact, datavalidationconfig)
        logging.info("Initiate Data Validation")
        data_validation_artifact = data_validation.initiate_data_validation()
        logging.info("Data Validation Completed")
        print(data_validation_artifact)

    except Exception as e:
        raise ChurnPredictionException(e, sys)