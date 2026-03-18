import os
import sys
import numpy as np
import pandas as pd

'''
Defining common constant variable for training pipeline.
'''

TARGET_COLUMN: str = "Will_Cancel_Next_3_Months"
PIPELINE_NAME: str = "ChurnPrediction"
ARTIFACT_DIR: str = "Artifacts"
FILE_NAME: str = "Subscription Fatigue.csv"

SCHEMA_FILE_PATH : str = os.path.join("data_schema","schema.yaml")

TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"

'''
Data Ingestion related constants start DATA_INGESTION VARIABLE NAME.
'''

DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_DATASET_PATH: str = "SaaSData/Subscription Fatigue.csv"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2

'''
Data Validation related constant start with DATA_VALIDATION VAR NAME.
'''

DATA_VALIDATION_DIR_NAME : str = "data_validation"
DATA_VALIDATION_VALID_DIR : str = "validated"
DATA_VALIDATION_INVALID_DIR : str = "invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR : str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME : str = "report.yaml"

