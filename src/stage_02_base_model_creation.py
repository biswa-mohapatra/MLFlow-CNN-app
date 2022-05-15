import argparse
import os
import shutil
from tqdm import tqdm
import logging
from src.utils.common import read_yaml, create_directories
from src.utils.model import log_model_summary
import random
import tensorflow as tf


STAGE = "BASE MODEL CREATION" ## <<< change stage name 

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
    logging.info("Layers defined...")
    LAYERS = [
    tf.keras.layers.Input(shape=tuple(params["img_shape"])),
    tf.keras.layers.Conv2D(32, (3,3), activation="relu"),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),
    tf.keras.layers.Conv2D(32, (3,3), activation="relu"),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(2, activation="softmax")
    ]

    classifier = tf.keras.Sequential(LAYERS)

    logging.info(f"Base model summary:\n{log_model_summary(classifier)}")
    
    classifier.compile(
    optimizer=tf.keras.optimizers.Adam(params["learning_rate"]),
    loss=params["loss"],
    metrics=params["metrics"]
    )

    # Saving the model 
    path_to_model_dir = os.path.join(config["Data"]["local_dir"],config["Data"]["model_dir"])

    path_to_model = os.path.join(path_to_model_dir,config["Data"]["initial_model_file"])
    
    create_directories([path_to_model_dir])
    classifier.save(path_to_model)
    logging.info(f"Model is saved at :: {path_to_model}")

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