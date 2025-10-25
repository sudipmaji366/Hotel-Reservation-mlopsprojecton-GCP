from src.data_ingestion import DataIngestion
from src.data_preprocessing import DataPreprocessor
from src.model_training import ModelTrainer
from config.paths_config import *
from utils.common import read_yaml_file

if __name__ == "__main__":

    data_ingestion = DataIngestion(read_yaml_file(CONFIG_PATH))
    data_ingestion.run()

    data_preprocessor = DataPreprocessor(
        train_path=TRAIN_FILE_PATH,
        test_path=TEST_FILE_PATH,
        processed_dir=PROCESSED_DIR,
        config_path=CONFIG_PATH
    )
    data_preprocessor.process()

    model_trainer = ModelTrainer(
        train_path=PROCESSED_TRAIN_DATA_PATH,
        test_path=PROCESSED_TEST_DATA_PATH,
        model_output_path=MODEL_OUTPUT_PATH
    )
    model_trainer.run()