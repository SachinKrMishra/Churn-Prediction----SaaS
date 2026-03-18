import pandas as pd
import os, sys
from scipy.stats import ks_2samp
from churnprediction.logging.logger import logging
from churnprediction.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from churnprediction.entity.config_entity import DataValidationConfig
from churnprediction.exception.exception import ChurnPredictionException
from churnprediction.constant.training_pipeline import SCHEMA_FILE_PATH
from churnprediction.utils.main_utils.utils import real_yaml_file, write_yaml_file


class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_config: DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self.schema_config = real_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise ChurnPredictionException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise ChurnPredictionException(e, sys)

    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        try:
            number_of_columns = len(self.schema_config["columns"])

            logging.info(f"Required number of columns : {number_of_columns}")
            logging.info(f"Data frame has columns :{len(dataframe.columns)}")
            
            if len(dataframe.columns) == number_of_columns:
                return True
            return False
        except Exception as e:
            raise ChurnPredictionException(e, sys)

    def detect_dataset_drift(self, base_df, current_df, threshold=0.05) -> bool:
        try:
            status = True
            report = {}

            for column in base_df.columns:
                d1 = base_df[column]
                d2 = current_df[column]

                test = ks_2samp(d1, d2)

                drift = bool(test.pvalue < threshold)
                if drift:
                    status = False

                report[column] = {
                    "p_value": float(test.pvalue),
                    "drift_status": drift
                }

            os.makedirs(self.data_validation_config.drift_report_dir, exist_ok=True)

            write_yaml_file(
                file_path=self.data_validation_config.drift_report_file_path,
                content=report
            )

            return status

        except Exception as e:
            raise ChurnPredictionException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            train_df = self.read_data(self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(self.data_ingestion_artifact.test_file_path)

            # Column validation
            if not self.validate_number_of_columns(train_df):
                raise Exception("Train dataframe column mismatch")

            if not self.validate_number_of_columns(test_df):
                raise Exception("Test dataframe column mismatch")

            # Drift detection
            drift_status = self.detect_dataset_drift(train_df, test_df)

            # Create directories
            os.makedirs(self.data_validation_config.valid_data_dir, exist_ok=True)
            os.makedirs(self.data_validation_config.invalid_data_dir, exist_ok=True)

            if drift_status:
                # Save valid data
                train_df.to_csv(self.data_validation_config.valid_train_file_path, index=False)
                test_df.to_csv(self.data_validation_config.valid_test_file_path, index=False)

                validation_status = True
                invalid_train = None
                invalid_test = None
            else:
                # Save invalid data
                train_df.to_csv(self.data_validation_config.invalid_train_file_path, index=False)
                test_df.to_csv(self.data_validation_config.invalid_test_file_path, index=False)

                validation_status = False
                invalid_train = self.data_validation_config.invalid_train_file_path
                invalid_test = self.data_validation_config.invalid_test_file_path

            return DataValidationArtifact(
                validation_status=validation_status,
                valid_train_file_path=self.data_validation_config.valid_train_file_path,
                valid_test_file_path=self.data_validation_config.valid_test_file_path,
                invalid_train_file_path=invalid_train,
                invalid_test_file_path=invalid_test,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )

        except Exception as e:
            raise ChurnPredictionException(e, sys)