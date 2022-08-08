import argparse
import faiss 
import image_embeddings
import os
import pandas as pd
from typing import Text

from src.utils.config import load_config


def build_index(config_path: Text) -> None:
    """Read the embeddings and build an index
    Args:
        config_path {Text}: path to config
        
    Notes:
    - The KNN index is built using https://github.com/facebookresearch/faiss
    """

    config = load_config(config_path)
    BASEDIR = config.data.data_base_dir
    DATASET = config.data.dataset_name
    PATH_EMBEDDINGS = f"{BASEDIR}/{DATASET}/{config.data.embeddings_dir}"
    PATH_INDEX = os.path.join(config.build_index.models_dir, config.build_index.model_name)
    
    [id_to_name, name_to_id, embeddings] = image_embeddings.knn.read_embeddings(
        PATH_EMBEDDINGS
    )
    index = image_embeddings.knn.build_index(embeddings)
    

    faiss.write_index(index, PATH_INDEX)
   

if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    build_index(config_path=args.config)
