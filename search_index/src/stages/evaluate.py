import argparse
import faiss
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pathlib import Path
import seaborn as sns
from tqdm import tqdm
from typing import Text

from src.utils.config import load_config
from src.utils.metrics import calculate_map_per_query
from search_index.src.utils import index_search, process_search_results


def evaluate(config_path: Text) -> None:
    """Evaluate model.

    Args:
        config_path {Text}: path to config
    """

    config = load_config(config_path)
    DATASET_NAME = config.data.dataset_name
    BASEDIR = config.data.data_base_dir
    PATH_TEST_DATA_EMBEDDINGS = f"{BASEDIR}/{DATASET_NAME}/embeddings/embeddings.parquet"
    PATH_INDEX = os.path.join(config.build_index.models_dir, config.build_index.model_name)
    PATH_DATASET_ANNOTATIONS = Path(f"{BASEDIR}/{DATASET_NAME}/annotations.csv")
    
    # Load data 
    emb = pd.read_parquet(PATH_TEST_DATA_EMBEDDINGS)
    embeddings = np.stack(emb["embedding"].to_numpy()).astype('float32')
    annotations = pd.read_csv(PATH_DATASET_ANNOTATIONS)
    label_map_dict = annotations[['label', 'label_name']].drop_duplicates().set_index('label').to_dict()['label_name']
    
    # Download the index model 
    index = faiss.read_index(PATH_INDEX)
    
    # Build a query 
    # (example: [{'id': 0, 'file': 'artwork/360.png', 'label': 0, 'label_name': 'artwork'}])
    q_df = emb[['file', 'label', 'label_name']].reset_index().rename(columns={'index': 'id'})
    query = [q_df.iloc[id].to_dict() for id in range(0, q_df.shape[0])]

    # Run a search for the query 
    query_results = {}
    for q in tqdm(query): 
        s_dist, s_ids = index_search(index, embeddings[q.get('id')], k=5)
        q_results = process_search_results(q, s_dist,  s_ids, q_df)
        query_results.update({ q.get('id'): q_results})
        
    # Prepare a report 
    frames = []
    for k in range(1, 6):
        frames.append(calculate_map_per_query(query, query_results, k=k))
    report = pd.concat(frames, axis=1)
    report = report.rename(index=label_map_dict)  # rename label to label_name
    report_map_at_5 = report['map@5'].to_dict()
    report_map_at_5['map@5'] = report.mean()['map@5']
    
    print(report)
    print(report_map_at_5)
    
    # Save reports and log metrics  
    PATH_REPORTS_DIR = config.evaluate.reports_dir
    PATH_METRICS_ALL = Path(PATH_REPORTS_DIR, config.evaluate.metrics_all)
    PATH_METRICS_MAP5 = Path(PATH_REPORTS_DIR, config.evaluate.metrics_map5)
    PATH_PLOT_METRICS_ALL = Path(PATH_REPORTS_DIR, config.evaluate.plots_metrics_all)
    
    report.to_csv(PATH_METRICS_ALL, index=True)
    
    with open(PATH_METRICS_MAP5 , 'w') as outfile:
        json.dump(report_map_at_5, outfile)
        
    f, ax = plt.subplots(figsize=(10, 5))
    ax = sns.heatmap(report.T, annot=True, cmap="YlGnBu")
    ax.set(
        title = 'Metrics for each class', 
        xlabel = 'Classes', 
        ylabel = 'Metrics'
    )
    plt.savefig(PATH_PLOT_METRICS_ALL, dpi=400)


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    evaluate(config_path=args.config)
