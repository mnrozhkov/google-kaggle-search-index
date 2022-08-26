import argparse
import faiss 
import numpy as np
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
    DATASET_NAME = config.data.dataset_name
    BASEDIR = config.data.data_base_dir
    PATH_EMBEDDINGS_DIR = f"{BASEDIR}/{DATASET_NAME}/embeddings"
    PATH_EMBEDDINGS = f"{PATH_EMBEDDINGS_DIR}/embeddings.parquet"
    PATH_INDEX = os.path.join(config.build_index.models_dir, config.build_index.model_name)
    
    
    # Load data 
    emb = pd.read_parquet(PATH_EMBEDDINGS)
    embeddings = np.stack(emb["embedding"].to_numpy()).astype('float32')
    
    # Build index
    index = faiss.IndexFlatL2(embeddings.shape[1])   # build the index object for vectors size 64
    index.add(embeddings)                  # add vectors to the index
    print(f"Index is trained: {index.is_trained}")
    print(f"Index size total: {index.ntotal}")
    
    # Save index
    faiss.write_index(index, PATH_INDEX)
    print(f"Index saved to: {PATH_INDEX}")
   

if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    build_index(config_path=args.config)
