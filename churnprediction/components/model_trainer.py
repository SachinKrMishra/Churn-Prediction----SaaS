import os
import sys
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

import mlflow
import mlflow.sklearn

from churnprediction.exception.exception import ChurnPredictionException
from churnprediction.logging.logger import logging
from churnprediction.entity.artifact_entity import (
    ModelTrainerArtifact,
    ClassificationMetricArtifact
)
from churnprediction.utils.main_utils.utils import load_numpy_array_data, save_object

class ModelTrainer:
    def __init__(self, model_trainer_config, data_transformation_artifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise ChurnPredictionException(e, sys)

    def evaluate_model(self, y_true, y_pred, y_prob):
        return ClassificationMetricArtifact(
            f1_score=f1_score(y_true, y_pred),
            precision_score=precision_score(y_true, y_pred),
            recall_score=recall_score(y_true, y_pred),
            roc_auc_score=roc_auc_score(y_true, y_prob)
        )

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logging.info("Starting model training with MLflow")

        try:
            # Load data
            train_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_train_file_path
            )
            test_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_test_file_path
            )

            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            models = {
                "LogisticRegression": LogisticRegression(class_weight="balanced", max_iter=1000),
                "RandomForest": RandomForestClassifier(class_weight="balanced"),
                "XGBoost": __import__("xgboost").XGBClassifier(eval_metric="logloss")
            }

            params = {
                "LogisticRegression": {
                    "C": [0.01, 0.1, 1, 10]
                },
                "RandomForest": {
                    "n_estimators": [100, 200],
                    "max_depth": [5, 10, None],
                    "min_samples_split": [2, 5, 10],
                    "class_weight": ["balanced"]
                },
                "XGBoost": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.01, 0.1],
                    "max_depth": [3, 6]
                }
            }

            best_model = None
            best_score = 0
            best_model_name = None
            best_train_metrics = None
            best_test_metrics = None

            mlflow.set_experiment("ChurnPrediction")

            for model_name, model in models.items():
                with mlflow.start_run(run_name=model_name):

                    grid = GridSearchCV(
                        model,
                        params[model_name],
                        cv=3,
                        scoring="f1",
                        n_jobs=-1
                    )

                    grid.fit(X_train, y_train)
                    best_estimator = grid.best_estimator_

                    # Predictions
                    y_train_pred = best_estimator.predict(X_train)
                    y_test_pred = best_estimator.predict(X_test)

                    y_train_prob = best_estimator.predict_proba(X_train)[:, 1]
                    y_test_prob = best_estimator.predict_proba(X_test)[:, 1]

                    # Metrics
                    train_metrics = self.evaluate_model(y_train, y_train_pred, y_train_prob)
                    test_metrics = self.evaluate_model(y_test, y_test_pred, y_test_prob)

                    mlflow.log_params(grid.best_params_)
                    mlflow.log_metric("train_f1", train_metrics.f1_score)
                    mlflow.log_metric("test_f1", test_metrics.f1_score)
                    mlflow.log_metric("test_roc_auc", test_metrics.roc_auc_score)

                    mlflow.sklearn.log_model(best_estimator, model_name)

                    # Best model selection (based on F1)
                    if test_metrics.f1_score > best_score:
                        best_score = test_metrics.f1_score
                        best_model = best_estimator
                        best_model_name = model_name
                        best_train_metrics = train_metrics
                        best_test_metrics = test_metrics

            # Overfitting check
            train_f1 = best_train_metrics.f1_score
            test_f1 = best_test_metrics.f1_score

            if abs(train_f1 - test_f1) > self.model_trainer_config.overfitting_underfitting_threshold:
                logging.warning(
                    f"Model may be overfitting/underfitting | Train F1: {train_f1}, Test F1: {test_f1}"
                )

            # Expected score check
            if best_test_metrics.f1_score < self.model_trainer_config.expected_f1_score:
                raise Exception("Model performance is below expected threshold")

            # Save best model
            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_file_path), exist_ok=True)
            save_object(self.model_trainer_config.trained_model_file_path, best_model)

            logging.info(f"Best Model: {best_model_name}")
            
            ## Model Pusher
            save_object("final_model/model.pkl",best_model)

            return ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=best_train_metrics,
                test_metric_artifact=best_test_metrics,
                best_model_name=best_model_name
            )

        except Exception as e:
            raise ChurnPredictionException(e, sys)