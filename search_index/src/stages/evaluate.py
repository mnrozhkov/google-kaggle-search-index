import argparse
from tkinter.tix import Tree
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
from src.utils.metrics import calculate_map_per_query, SimilarityType
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

     # Save reports and log metrics  
    PATH_REPORTS_DIR = Path(config.evaluate.reports_dir)
    PATH_METRICS_ALL = PATH_REPORTS_DIR / config.evaluate.metrics_all
    PATH_METRICS_MAP5_DIR = PATH_REPORTS_DIR / config.evaluate.metrics_map5
    PATH_PLOT_METRICS_ALL = PATH_REPORTS_DIR / config.evaluate.plots_metrics_all
    QUERY_RESULTS_PATH = PATH_REPORTS_DIR / config.evaluate.query_results
    
    # Load data 
    emb = pd.read_parquet(PATH_TEST_DATA_EMBEDDINGS)
    embeddings = np.stack(emb["embedding"].to_numpy()).astype('float32')
    annotations = pd.read_csv(PATH_DATASET_ANNOTATIONS)
    label_map_dict = annotations[['label', 'label_name']].drop_duplicates().set_index('label').to_dict()['label_name']
    
    # Download the index model 
    index = faiss.read_index(PATH_INDEX)
    
    # Build a query 
    # (example: [{'id': 0, 'file': 'artwork/360.png', 'label': 0, 'label_name': 'artwork', 'sim_category': 'close_positive'}])
    q_df = emb[['file', 'label', 'label_name', 'sim_category']].reset_index().rename(columns={'index': 'id'})
    queries = [q_df.iloc[id].to_dict() for id in range(0, q_df.shape[0])]

    # Run a search for the query 
    query_results = {}
    for q in tqdm(queries): 
        s_dist, s_ids = index_search(index, embeddings[q.get('id')], k=5)
        q_results = process_search_results(q, s_dist,  s_ids, q_df)
        query_results.update({
            q.get('id'): {
                'label_name': q.get('label_name'),
                'file': q.get('file'),
                'results': q_results
            }
        })

    # Prepare a reports
    os.makedirs(PATH_METRICS_MAP5_DIR, exist_ok=True)

    for sim_type in SimilarityType:
        
        print(f'Similarity type: {sim_type}')
        frames = []
        
        for k in [1, 3, 5]:
            frames.append(calculate_map_per_query(
                queries, query_results, k=k, sim_type=sim_type
            ))
        
        report = pd.concat(frames, axis=1)
        report = report.rename(index=label_map_dict)  # rename label to label_name
        report_map_at_5 = report['map@5'].to_dict()
        report_map_at_5['map@5'] = report.mean()['map@5']

        with open(PATH_METRICS_MAP5_DIR / f'{sim_type.value}.json' , 'w') as outfile:
            json.dump(report_map_at_5, outfile, indent=4)
    
        print(report)
        print(report_map_at_5)
    
    query_items = []

    for query_id, query_info in query_results.items():
        item_df = pd.DataFrame(query_info['results'])
        item_df['query_id'] = query_id
        item_df['query_label_name'] = query_info['label_name']
        item_df['query_file'] = query_info['file']
        query_items.append(item_df)

    query_results_df = pd.concat(query_items, ignore_index=True)
    query_results_df.to_csv(QUERY_RESULTS_PATH, index=False)

    report.to_csv(PATH_METRICS_ALL, index=True)

    f, ax = plt.subplots(figsize=(10, 6))
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
