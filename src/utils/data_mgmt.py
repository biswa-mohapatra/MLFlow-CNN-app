from PIL import Image
import os
import imghdr
import shutil
import numpy as np
import logging
from src.utils.common import create_directories

def validating_img(config:dict) -> None:
    PARENT_DIR = os.path.join(config["DATA"]["unzip_data_dir"],
                            config["DATA"]["parent_data_dir"])
    BAD_DATA_DIR = os.path.join(config["DATA"]["unzip_data_dir"],
                            config["DATA"]["bad_data"])
    create_directories([BAD_DATA_DIR])
    for dirs in os.listdir(PARENT_DIR):
        full_path_data_dir = os.path.join(PARENT_DIR, dirs)
        for imgs in os.listdir(full_path_data_dir):
            path_to_img = os.path.join(full_path_data_dir, imgs)
            try:
                img = Image.open(path_to_img)
                img.verify()
                    
                if len(img.getbands()) !=3 or imghdr.what(path_to_img) not in ['jpeg','png']:
                    bad_data_path = os.path.join(BAD_DATA_DIR, imgs)
                    shutil.move(path_to_img, bad_data_path)
                    logging.info(f"{path_to_img} not of expected format")
                    continue
                    
            except Exception as e:
                logging.info(f"{path_to_img} not of expected format")
                bad_data_path = os.path.join(BAD_DATA_DIR, imgs)
                shutil.move(path_to_img, bad_data_path)
                logging.info(f"moved bad file from {path_to_img} to {bad_data_path}")
                logging.exception(e)