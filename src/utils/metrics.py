import numpy as np
import pandas as pd 
from typing import Text, Dict


def overall_precision(label, y_pred, k, n):
    
    if len(y_pred) == 0: 
        return 0.0
    
    j = min(n, k) 
    # print(j)
    
    ap_overall = []
    tp = 0
    for i in range(j):
        rel = int(y_pred[i] == label)
        tp += rel
        ap = rel * tp / (i+1)
        ap_overall.append(ap)
        # print(tp, rel, ap)
    
    return sum(ap_overall) / j


def map_per_query(labels, predictions, k, n):
    # Average precision for all labels in query `@k` predictions (but not more than `n`` ground truth positives)
    return np.mean([overall_precision(l, p, k, n) for l,p in zip(labels, predictions)])


def calculate_map_per_query(query, query_results, k):
    # Calculate MAP@k for each label in the query 
    
    # Count true labels for the query
    query_n = {}
    for q in query: 
        true_label = q.get('label')
        query_n[true_label] = query_n.get(true_label, 0) + 1

    # Calculate metrics for each label
    query_metrics = []
    for q in query: 
        y_pred = pd.DataFrame(query_results.get(q.get('id'))).label
        true_label = q.get('label')
        n = query_n.get(true_label, 0)

        q_map = overall_precision(true_label, y_pred, k, n)
        query_metrics.append({'label': true_label, f'map@{k}': q_map})
        
    return pd.DataFrame(query_metrics).groupby('label').agg('mean')
