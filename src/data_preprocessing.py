import pandas as pd
import os
import sys
import numpy as np
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common import load_data,read_yaml_file
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE


logger=get_logger(__name__)

class DataPreprocessor:

    def __init__(self, train_path, test_path,processed_dir,config_path):
        self.train_path = train_path
        self.test_path = test_path
        self.processed_dir=processed_dir
        self.config = read_yaml_file(config_path)
        os.makedirs(self.processed_dir, exist_ok=True)

    def preprocess_data(self,df):
        try:
            logger.info("Starting data preprocessing")
            logger.info("dropping unnecessary columns")
            df.drop(columns=['Unnamed: 0', 'Booking_ID'],inplace=True)
            df.drop_duplicates(inplace=True)
            cat_coloumns=self.config["data_processing"]["categorical_colomns"]
            num_coloumns=self.config["data_processing"]["numerical_columns"]
            label_encoder = LabelEncoder()

            mappings={}

            for col in cat_coloumns:
              df[col] = label_encoder.fit_transform(df[col])

              mappings[col] = {label:code for label,code in zip(label_encoder.classes_ , label_encoder.transform(label_encoder.classes_))}
            logger.info(f"Label mapping are : ")
            for col,mapping in mappings.items():

              logger.info(f"{col} : {mappings[col]}")
            logger.info("Handling skewing")
            skew_threshold = self.config["data_processing"]["skewness_threshold"]
            skewness = df[num_coloumns].apply(lambda x: x.skew())
            for col in skewness[skewness > skew_threshold].index:
                df[col] = np.log1p(df[col])
            return df
        except Exception as e:
            logger.error(f"Error in data preprocessing: {e}")
            raise CustomException(str(e), sys) from e

    def balance_data(self,df):
        try:
            logger.info("Balancing the dataset using SMOTE")
            X=df.drop(columns=['booking_status'])
            y=df['booking_status']
            smote = SMOTE(random_state=42)
            X_res, y_res = smote.fit_resample(X, y)
            balanced_df = pd.DataFrame(X_res , columns=X.columns)
            balanced_df["booking_status"] = y_res
            logger.info("Data balancing completed")
            return balanced_df
        except Exception as e:
            logger.error(f"Error in balancing data: {e}")
            raise CustomException(str(e), sys) from e
    def feature_selection(self,df):
     try:
        X = df.drop(columns='booking_status')
        y = df["booking_status"]
        model =  RandomForestClassifier(random_state=42)
        model.fit(X,y)
        feature_importance = model.feature_importances_
        feature_importance_df = pd.DataFrame({
          'feature':X.columns,
          'importance':feature_importance
             })
        top_features_importance_df=feature_importance_df.sort_values(by='importance' , ascending=False)
        # config.yaml uses the key `no_of_features` (number of features to select)
        num_of_features_to_select = self.config["data_processing"].get("no_of_features")
        if num_of_features_to_select is None:
            # backward-compatible fallback to older key name
            num_of_features_to_select = self.config["data_processing"].get("num_of_features", 10)
        top_10_features = top_features_importance_df["feature"].head(num_of_features_to_select).values

        top_10_df = df[top_10_features.tolist() + ["booking_status"]]
        logger.info(f"Selected top {num_of_features_to_select} features based on importance")
        return top_10_df
     except Exception as e:
        logger.error(f"Error in feature selection: {e}")
        raise CustomException(str(e), sys) from e
    def save_processed_data(self,df,filepath):
        try:
            df.to_csv(filepath , index=False)
            logger.info(f"Processed data saved at {filepath}")
        except Exception as e:
            logger.error(f"Error in saving processed data: {e}")
            raise CustomException(str(e), sys) from e
    
    def process(self):
       try:
          logger.info("Loading training data")
          train_df=load_data(self.train_path)
          logger.info("Loading testing data")
          test_df=load_data(self.test_path)
          train_df=self.preprocess_data(train_df)
          test_df=self.preprocess_data(test_df)


          train_df=self.balance_data(train_df)
          test_df=self.balance_data(test_df)
          train_df=self.feature_selection(train_df)
          test_df=test_df[train_df.columns]
          
          self.save_processed_data(train_df , PROCESSED_TRAIN_DATA_PATH)
          self.save_processed_data(test_df , PROCESSED_TEST_DATA_PATH)
          logger.info("Data preprocessing completed successfully")
       except Exception as e:
          logger.error(f"Error in data processing pipeline: {e}")
          raise CustomException(str(e), sys) from e
if __name__ == "__main__":
    data_preprocessor = DataPreprocessor(
        train_path=TRAIN_FILE_PATH,
        test_path=TEST_FILE_PATH,
        processed_dir=PROCESSED_DIR,
        config_path=CONFIG_PATH
    )
    data_preprocessor.process()
          
