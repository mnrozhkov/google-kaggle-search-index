import argparse
import faiss
import image_embeddings
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from typing import Text

from src.utils.config import load_config
from src.utils.metrics import overall_precision, map_per_query

def process_results(results):

    results_proc = []
    for item in results:
        img_meta = item[1].split('_')

        results_proc.append({ 
            'id': img_meta[2],
            'label': img_meta[1],
            'd': item[0]
        })
    # return pd.DataFrame(results_proc)
    return results_proc

def evaluate(config_path: Text) -> None:
    """Evaluate model.

    Args:
        config_path {Text}: path to config
    """

    config = load_config(config_path)
    BASEDIR = config.data.data_base_dir
    DATASET = config.data.test_dataset_name
    PATH_TEST_DATA_EMBEDDINGS = f"{BASEDIR}/{DATASET}/{config.data.embeddings_dir}"
    PATH_INDEX = os.path.join(config.build_index.models_dir, config.build_index.model_name)
    
    # Download the index model 
    index = faiss.read_index(PATH_INDEX)
    
    # Run inference for test datasets 
    [id_to_name, name_to_id, embeddings] = image_embeddings.knn.read_embeddings(PATH_TEST_DATA_EMBEDDINGS)
    
    # Build a query 
    query = [{
        'id': i, 
        'name': id_to_name[i],
        'true_label': id_to_name[i].split('_')[1]
        } for i in id_to_name.keys()]
    # print(query)
    
    # Run searching for the query 
    query_results = {}
    for q in query: 
        q_results = image_embeddings.knn.search(index, id_to_name, embeddings[q.get('id')])
        query_results.update({ q.get('id'): process_results(q_results)})
        
    # Prepare a report with metrics and plots 
    frames = []
    for k in range(1, 6):
        frames.append(map_per_query(query, query_results, k=k))
        
    report = pd.concat(frames, axis=1)
    report_map_at_5 = report['map@5'].to_dict()
    report_map_at_5['map@5'] = report.mean()['map@5']
    
    # Save reports and log metrics  
    PATH_REPORTS_DIR = config.base.reports_dir
    PATH_METRICS_ALL = f"{PATH_REPORTS_DIR}/{config.evaluate.metrics_all}"
    PATH_METRICS_MAP5 = f"{PATH_REPORTS_DIR}/{config.evaluate.metrics_map5}"
    PATH_PLOT_METRICS_ALL = f"{PATH_REPORTS_DIR}/{config.evaluate.plots_metrics_all}"
    
    report.to_csv(PATH_METRICS_ALL, index=False)
    with open(PATH_METRICS_MAP5 , 'w') as outfile:
        json.dump(report_map_at_5, outfile)
        
    f, ax = plt.subplots(figsize=(10, 5))
    ax = sns.heatmap(report.T, annot=True,cmap="YlGnBu")
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
