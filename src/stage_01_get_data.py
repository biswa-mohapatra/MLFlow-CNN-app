import argparse
import os
import shutil
from tqdm import tqdm
import logging
from src.utils.common import read_yaml, create_directories,unzip_file
import random
from urllib import request as req


STAGE = "GET_DATA" ## <<< change stage name 

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


def main(config_path):
    ## read config files
    config = read_yaml(config_path)
    #params = read_yaml(params_path)
    URL = config['DATA']["SOURCE_URL"]
    local_dir = config['DATA']["local_dir"]
    ## Creating Directory
    create_directories([local_dir])

    file_name = config['DATA']["data_file_name"]
    ## Creating Data file path
    data_file_path = os.path.join(local_dir,file_name)

    ## Downloading the data file
    
    if not os.path.isfile(data_file_path):
        logging.info("Downloading started......")
        filename, headers = req.urlretrieve(URL,data_file_path)
        ## Logging the file name and headers
        logging.info(f"File :: {filename} is created \n with info :: {headers} \n at {data_file_path}...")
    else:
        logging.info(f"File :: {file_name} is already present.....")
    
    # Uzipping ops
    destination = config["DATA"]["unzip_data_dir"]
    create_directories([destination]) ## creating the destination directory
    unzip_file(data_file_path,destination)


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