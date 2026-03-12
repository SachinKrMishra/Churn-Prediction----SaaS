from churnprediction.exception.exception import ChurnPredictionException
from churnprediction.logging.logger import logging

## Configuration of the data integestion config
from churnprediction.entity.config_entity import DataIngestionConfig
from churnprediction.entity.artifact_entity import DataIngestionArtifact

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise ChurnPredictionException(e, sys)

    def export_data_as_dataframe(self):
        '''
        Read data from local CSV file
        '''
        try:
            file_path = self.data_ingestion_config.dataset_path

            logging.info(f"Reading dataset from path: {file_path}")

            df = pd.read_csv(file_path)

            df = df.drop(['Customer_ID'], axis=1)

            logging.info(f"Dataset loaded successfully with shape {df.shape}")

            return df

        except Exception as e:
            raise ChurnPredictionException(e, sys)

    def export_data_into_feature_store(self, dataframe: pd.DataFrame):
        try:
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path

            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)

            dataframe.to_csv(feature_store_file_path, index=False, header=True)

            logging.info("Data exported to feature store")

            return dataframe

        except Exception as e:
            raise ChurnPredictionException(e, sys)

    def split_data_as_train_test_split(self, dataframe: pd.DataFrame):
        try:
            train_set, test_set = train_test_split(
                dataframe,
                test_size=self.data_ingestion_config.train_test_split_ratio,
                random_state=42
            )

            logging.info("Performed train-test split")

            train_dir = os.path.dirname(self.data_ingestion_config.training_file_path)
            test_dir = os.path.dirname(self.data_ingestion_config.testing_file_path)

            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(test_dir, exist_ok=True)

            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index=False, header=True)

            logging.info("Train and test datasets saved")

        except Exception as e:
            raise ChurnPredictionException(e, sys)

    def initiate_data_ingestion(self):
        try:
            dataframe = self.export_data_as_dataframe()

            dataframe = self.export_data_into_feature_store(dataframe)

            self.split_data_as_train_test_split(dataframe)

            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path
            )

            logging.info("Data ingestion completed")

            return data_ingestion_artifact

        except Exception as e:
            raise ChurnPredictionException(e, sys)