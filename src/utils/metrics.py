from collections import defaultdict
from enum import Enum
from multiprocessing.pool import CLOSE
import numpy as np
import pandas as pd 
from typing import Text, Dict


class SimilarityType(Enum):
    NO_SIMILARITY = 'no_similarity'
    CLOSE_POSITIVE = 'close_positive'
    CLOSE_POSITIVE_OR_HARD_POSITIVE = 'close_positive_or_hard_positive'


def calc_rel(query, query_result, sim_type):

    query_label = query['label']
    query_result_label = query_result['label']
    query_sim_cat = query['sim_category']
    query_result_sim_cat = query_result['sim_category']
    
    labels_equal = (query_label == query_result_label)
    sim_cats_equal = (query_sim_cat and query_result_sim_cat)


    if sim_type == SimilarityType.NO_SIMILARITY:
        return labels_equal
    
    elif sim_type == SimilarityType.CLOSE_POSITIVE:
        return (
            labels_equal and
            sim_cats_equal and
            query_sim_cat == 'close_positive'
        )
    
    elif sim_type == SimilarityType.CLOSE_POSITIVE_OR_HARD_POSITIVE:
        return (
            labels_equal and
            sim_cats_equal and
            (query_sim_cat in ['close_positive', 'hard_positive'])
        )


def overall_precision(
    query, q_result, k, n,
    sim_type: SimilarityType = SimilarityType.NO_SIMILARITY
):
    
    if q_result.shape[0] == 0: 
        return 0.0
    
    j = min(n, k) 
    ap_overall = []
    tp = 0

    for i in range(j):
        # rel = int(y_pred[i] == label)
        rel = calc_rel(query, q_result.iloc[i], sim_type)
        tp += rel
        ap = rel * tp / (i+1)
        ap_overall.append(ap)
    
    return sum(ap_overall) / j


def map_per_query(labels, predictions, k, n):
    # Average precision for all labels in query `@k` predictions (but not more than `n`` ground truth positives)
    return np.mean([overall_precision(l, p, k, n) for l,p in zip(labels, predictions)])


def calculate_map_per_query(
    queries, query_results, k, 
    sim_type: SimilarityType = SimilarityType.NO_SIMILARITY
):
    # Calculate MAP@k for each label in the query 
    
    # Count true labels for the query
    query_n = defaultdict(int)

    for query in queries: 
        true_label = query.get('label')
        query_n[true_label] += 1

    # Calculate metrics for each label
    query_metrics = []
    for query in queries:
        # print(f'query_results: {query_results}')
        # print(query_results[])
        q_result = pd.DataFrame(query_results.get(query.get('id')).get('results'))
        true_label = query.get('label')
        n = query_n.get(true_label, 0)

        q_map = overall_precision(query, q_result, k, n, sim_type)
        query_metrics.append({'label': true_label, f'map@{k}': q_map})
        
    return pd.DataFrame(query_metrics).groupby('label').agg('mean')
