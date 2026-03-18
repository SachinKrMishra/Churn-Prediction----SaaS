from datetime import datetime
import os
from churnprediction.constant import training_pipeline

print(training_pipeline.PIPELINE_NAME)
print(training_pipeline.ARTIFACT_DIR)

class TrainingPipelineConfig:
    def __init__(self, timestamp=datetime.now()):
        timestamp = timestamp.strftime("%m_%d_%Y_%H_%M_%S")

        self.pipeline_name = training_pipeline.PIPELINE_NAME
        self.artifact_name = training_pipeline.ARTIFACT_DIR
        self.artifact_dir = os.path.join(self.artifact_name, timestamp)

        self.timestamp: str = timestamp


class DataIngestionConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):

        # Root directory for data ingestion artifacts
        self.data_ingestion_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.DATA_INGESTION_DIR_NAME
        )

        # Local dataset path
        self.dataset_path: str = training_pipeline.DATA_INGESTION_DATASET_PATH

        # Feature store path
        self.feature_store_file_path: str = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_FEATURE_STORE_DIR,
            training_pipeline.FILE_NAME
        )

        # Train dataset path
        self.training_file_path: str = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_INGESTED_DIR,
            training_pipeline.TRAIN_FILE_NAME
        )

        # Test dataset path
        self.testing_file_path: str = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_INGESTED_DIR,
            training_pipeline.TEST_FILE_NAME
        )

        # Train test split ratio
        self.train_test_split_ratio: float = training_pipeline.DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO


class DataValidationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):

        # Root directory for data validation
        self.data_validation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.DATA_VALIDATION_DIR_NAME
        )

        # Valid & Invalid directories
        self.valid_data_dir: str = os.path.join(
            self.data_validation_dir,
            training_pipeline.DATA_VALIDATION_VALID_DIR
        )

        self.invalid_data_dir: str = os.path.join(
            self.data_validation_dir,
            training_pipeline.DATA_VALIDATION_INVALID_DIR
        )

        # Valid file paths
        self.valid_train_file_path: str = os.path.join(
            self.valid_data_dir,
            training_pipeline.TRAIN_FILE_NAME
        )

        self.valid_test_file_path: str = os.path.join(
            self.valid_data_dir,
            training_pipeline.TEST_FILE_NAME
        )

        # Invalid file paths
        self.invalid_train_file_path: str = os.path.join(
            self.invalid_data_dir,
            training_pipeline.TRAIN_FILE_NAME
        )

        self.invalid_test_file_path: str = os.path.join(
            self.invalid_data_dir,
            training_pipeline.TEST_FILE_NAME
        )

        # Drift report directory
        self.drift_report_dir: str = os.path.join(
            self.data_validation_dir,
            training_pipeline.DATA_VALIDATION_DRIFT_REPORT_DIR
        )

        # Drift report file path
        self.drift_report_file_path: str = os.path.join(
            self.drift_report_dir,
            training_pipeline.DATA_VALIDATION_DRIFT_REPORT_FILE_NAME
        )

        # Add schema file path (CRITICAL)
        self.schema_file_path: str = training_pipeline.SCHEMA_FILE_PATH