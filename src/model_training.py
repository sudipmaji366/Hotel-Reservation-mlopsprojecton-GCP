import os
import sys
import pandas as pd
import joblib
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import lightgbm as lgb
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from config.model_params import *
from utils.common import load_data,read_yaml_file
from scipy.stats import uniform, randint

import mlflow
import mlflow.sklearn
logger = get_logger(__name__)

class ModelTrainer:

    def __init__(self, train_path,test_path,model_output_path):
        self.train_path = train_path
        self.test_path = test_path
        self.model_output_path = model_output_path
        
        self.param_distributions = LIGHTGBM_PARAMS
        self.random_search_params = RANDOM_SEARCH_PARAMS

    def load_and_split_data(self):
        try:
            logger.info("Loading training and testing data")
            train_df = load_data(self.train_path)
            test_df = load_data(self.test_path)

            X_train = train_df.drop(columns=['booking_status'])
            y_train = train_df['booking_status']

            X_test = test_df.drop(columns=['booking_status'])
            y_test = test_df['booking_status']
            logger.info("Data loading and splitting completed")

            return X_train, y_train, X_test, y_test
        except Exception as e:
            logger.error(f"Error loading and splitting data: {e}")
            raise CustomException(str(e), sys) from e
        
    def train_model(self,X_train,y_train):
        try:
            logger.info("Starting model training")
            lgbm = lgb.LGBMClassifier(random_state=42)

            logger.info("performing hyperparameter tuning")
            random_search = RandomizedSearchCV(
                estimator=lgbm,
                param_distributions=self.param_distributions,
                n_iter=self.random_search_params['n_iter'],
                cv=self.random_search_params['cv'],
                n_jobs=self.random_search_params['n_jobs'],
                verbose=self.random_search_params['verbose'],
                random_state=self.random_search_params['random_state'],
                scoring=self.random_search_params['scoring']
            )

            logger.info("Fitting the model with best parameters")
            random_search.fit(X_train, y_train)
            best_params = random_search.best_params_
            best_model = random_search.best_estimator_
            logger.info(f"Best parameters found: {best_params}")
            logger.info("Model training completed")
            return best_model
        except Exception as e:
            logger.error(f"Error in model training: {e}")
            raise CustomException(str(e), sys) from e
    
    def evaluate_model(self,model,X_test,y_test):
        try:
            logger.info("Evaluating the model'")
            y_pred = model.predict(X_test)
            accuracy=accuracy_score(y_test, y_pred)
            precision=precision_score(y_test, y_pred)
            recall=recall_score(y_test, y_pred)
            f1=f1_score(y_test, y_pred)

            logger.info(f"Model Evaluation Metrics: Accuracy={accuracy}, Precision={precision}, Recall={recall}, F1-Score={f1}")
            return accuracy, precision, recall, f1
        except Exception as e:
            logger.error(f"Error in model evaluation: {e}")
            raise CustomException(str(e), sys) from e
    
    def save_model(self,model):
        try:
            # Create model output directory if it doesn't exist
            os.makedirs(os.path.dirname(self.model_output_path), exist_ok=True)
            joblib.dump(model, self.model_output_path)
            logger.info(f"Model saved at {self.model_output_path}")
        except Exception as e:
            logger.error(f"Error in saving model: {e}")
            raise CustomException(str(e), sys) from e
    
    def run(self):

        try:
            with mlflow.start_run():
                logger.info("Strating mlflow experiment logging")
                logger.info("training and testing dataseet in mlflow")
                mlflow.log_artifact(self.train_path, artifact_path="datasets")
                mlflow.log_artifact(self.test_path, artifact_path="datasets")
                



                logger.info("Model Trainer started")
                X_train, y_train, X_test, y_test = self.load_and_split_data()
                best_model = self.train_model(X_train, y_train)
                metrics=self.evaluate_model(best_model, X_test, y_test)
                self.save_model(best_model)


                logger.info("Logging model to mlflow")
                mlflow.log_artifact(self.model_output_path)

                logger.info("Logging metrics to mlflow")
                # Log each model parameter separately
                logger.info("Logging parameters to mlflow")
                for param_name, param_value in best_model.get_params().items():
                    mlflow.log_param(param_name, param_value)
                
                # Log each metric separately
                logger.info("Logging metrics to mlflow")
                metric_names = ['accuracy', 'precision', 'recall', 'f1_score']
                for metric_name, metric_value in zip(metric_names, metrics):
                    mlflow.log_metric(metric_name, metric_value)
                
                logger.info("Model Trainer finished successfully")
                return metrics
        except Exception as e:
                logger.error(f"Error in Model Trainer: {e}")
                raise CustomException(str(e), sys) from e
        
if __name__ == "__main__":

    model_trainer = ModelTrainer(
        train_path=PROCESSED_TRAIN_DATA_PATH,
        test_path=PROCESSED_TEST_DATA_PATH,
        model_output_path=MODEL_OUTPUT_PATH
    )
    model_trainer.run()
        