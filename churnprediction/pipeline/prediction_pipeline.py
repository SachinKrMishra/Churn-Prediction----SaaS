import sys
import pandas as pd

from churnprediction.exception.exception import ChurnPredictionException
from churnprediction.logging.logger import logging
from churnprediction.utils.main_utils.utils import load_object


class PredictionPipeline:
    def __init__(self):
        try:
            logging.info("Loading model and preprocessor")

            self.model = load_object("final_model/model.pkl")
            self.preprocessor = load_object("final_model/preprocessor.pkl")

            logging.info("Model and preprocessor loaded successfully")

        except Exception as e:
            raise ChurnPredictionException(e, sys)

    def predict(self, dataframe: pd.DataFrame):
        """
        Performs prediction on input dataframe
        """
        try:
            logging.info("Starting prediction pipeline")

            # Transform input data
            transformed_data = self.preprocessor.transform(dataframe)

            # Predictions
            prediction = self.model.predict(transformed_data)
            probability = self.model.predict_proba(transformed_data)[:, 1]

            logging.info("Prediction completed successfully")

            return prediction, probability

        except Exception as e:
            raise ChurnPredictionException(e, sys)