import argparse
import os
import shutil
from tqdm import tqdm
import logging
import tensorflow as tf
import mlflow
from src.utils.common import read_yaml, create_directories
import random


STAGE = "STAGE_NAME" ## <<< change stage name 

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


def main(config_path):
    ## read config files
    config = read_yaml(config_path)
    params = config["params"]
    # get ready the data

    PARENT_DIR = os.path.join(config["DATA"]["unzip_data_dir"],
                            config["DATA"]["parent_data_dir"])
    # creating training and validation dataset
    logging.info(f"Creating training and validation dataset from :: {PARENT_DIR}")
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        PARENT_DIR,
        validation_split =params["validation_split"],
        subset = "training",
        seed=params["seed"],
        image_size=params["img_shape"][:-1],
        batch_size=params["batch_size"]
        )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        PARENT_DIR,
        validation_split = params["validation_split"],
        subset = "validation",
        seed=params["seed"],
        image_size=params["img_shape"][:-1],
        batch_size=params["batch_size"]
        )

    train_ds = train_ds.prefetch(buffer_size=params["buffer_size"])
    val_ds = val_ds.prefetch(buffer_size=params["buffer_size"])
    # load the base model
    logging.info(f"Creating training and validation dataset from :: {PARENT_DIR}")
    path_to_model = os.path.join(
        config["DATA"]["local_dir"],
        config["DATA"]["model_dir"],
        config["DATA"]["initial_model_file"])

    logging.info(f"Loading the base model from :: {path_to_model}")

    classifier = tf.keras.models.load_model(path_to_model)
    # starting the training
    logging.info(f"Starting the training")
    classifier.fit(train_ds,epochs = params["epochs"],validation_data =  val_ds)
    
    # Saving the model file
    trained_model_file = os.path.join(
        config["DATA"]["local_dir"],
        config["DATA"]["model_dir"],
        config["DATA"]["trained_model_file"])
    
    classifier.save(trained_model_file)
    logging.info(f"Trained model saved at :: {trained_model_file}")

    # saving the model in mlflow way
    with mlflow.run() as run:
        mlflow.log_params(params)
        mlflow.keras.log_model(classifier, "model")
    pass


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path=parsed_args.config)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e