import os
import sys
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churnprediction.exception.exception import ChurnPredictionException
from churnprediction.logging.logger import logging
from churnprediction.entity.artifact_entity import DataTransformationArtifact, DataValidationArtifact
from churnprediction.constant.training_pipeline import TARGET_COLUMN

from churnprediction.utils.main_utils.utils import save_numpy_array_data, save_object


class DataTransformation:
    def __init__(self, data_validation_artifact, data_transformation_config):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise ChurnPredictionException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise ChurnPredictionException(e, sys)

    def get_data_transformer_object(self) -> ColumnTransformer:
        """
        Creates preprocessing pipeline:
        - Numerical → StandardScaler
        - Categorical → OneHotEncoder
        """

        logging.info("Creating preprocessing pipeline")

        try:
            numerical_columns = [
                "Age",
                "Number_of_Subscriptions",
                "Avg_Usage_Hours_Per_Week",
                "App_Switch_Frequency",
                "Discount_Used",
                "Customer_Support_Interactions",
                "Tenure_Months",
                "Monthly_Total_Spend"
            ]

            categorical_columns = [
                "Income_Level",
                "Payment_Mode",
                "Device_Type"
            ]

            num_pipeline = Pipeline([
                ("scaler", StandardScaler())
            ])

            cat_pipeline = Pipeline([
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ])

            preprocessor = ColumnTransformer([
                ("num", num_pipeline, numerical_columns),
                ("cat", cat_pipeline, categorical_columns)
            ])

            return preprocessor

        except Exception as e:
            raise ChurnPredictionException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        logging.info("Starting data transformation")

        try:
            train_df = self.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = self.read_data(self.data_validation_artifact.valid_test_file_path)

            # Columns to drop (leakage)
            drop_columns = ["Satisfaction_Score"]

            # Drop only if present (safe handling)
            train_df = train_df.drop(columns=[col for col in drop_columns if col in train_df.columns])
            test_df = test_df.drop(columns=[col for col in drop_columns if col in test_df.columns])

            # Split input & target
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]

            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]

            # Preprocessing
            preprocessor = self.get_data_transformer_object()

            transformed_input_train = preprocessor.fit_transform(input_feature_train_df)
            transformed_input_test = preprocessor.transform(input_feature_test_df)

            # Combine input + target
            train_arr = np.c_[transformed_input_train, np.array(target_feature_train_df)]
            test_arr = np.c_[transformed_input_test, np.array(target_feature_test_df)]

            # Create directories
            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_train_file_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_object_file_path), exist_ok=True)

            # Save outputs
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, test_arr)

            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)

            logging.info("Data transformation completed successfully")

            ## Model Pusher
            save_object("final_model/preprocessor.pkl", preprocessor)

            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

        except Exception as e:
            raise ChurnPredictionException(e, sys)