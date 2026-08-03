from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torchmetrics.functional.classification import (
    multiclass_f1_score
)

ROUND_DIGITS = 4


def calculate_batch_normalized_dot_product(
        pred_tensor: torch.Tensor,
        label_tensor: torch.Tensor,
        device: torch.device
) -> float:
    dot_products = (pred_tensor * label_tensor).sum(dim=1)
    k_vals = (label_tensor != 0).sum(dim=1).float()

    # Apply scaling: (dot + k) / (2 * k)
    # Changed: Returns 0.0 if k_vals is 0 (no labels present)
    scores = torch.where(
        k_vals > 0,
        (dot_products + k_vals) / (2 * k_vals),
        torch.tensor(0.0, device=device)
    )
    return round(scores.mean().item(), ROUND_DIGITS)


def calculate_batch_cosine_similarity(
        pred_tensor: torch.Tensor,
        label_tensor: torch.Tensor,
) -> float:
    cos_sims = torch.nn.functional.cosine_similarity(pred_tensor, label_tensor, dim=1)

    # 3. Handle zero-vector edge cases (where norm is 0)
    # torch.nn.functional.cosine_similarity returns 0 by default for zero vectors
    return round(cos_sims.mean().item(), ROUND_DIGITS)


def calculate_mrr(y_pred: list[list[str]], y_true: list[str]) -> float:
    """Calculates Mean Reciprocal Rank for ranked predictions."""
    rr_sum = 0
    for pred_list, true_label in zip(y_pred, y_true):
        if true_label in pred_list:
            rank = pred_list.index(true_label) + 1
            rr_sum += 1.0 / rank
    return rr_sum / len(y_true) if y_true else 0.0


def calculate_accuracy_at_k(y_pred: list[list[str]], y_true: list[str], k: int) -> float:
    """Calculates Hit Rate @ K."""
    hits = 0
    for pred_list, true_label in zip(y_pred, y_true):
        if true_label in pred_list[:k]:
            hits += 1
    return hits / len(y_true) if y_true else 0.0

def calculate_batch_symptom_extraction_metrics(predictions: list, labels: list, device) -> dict:

    # We create a dataframe temporarily to group by length and calculate metrics
    df_temp = pd.DataFrame({'p': predictions, 'l': labels})
    df_temp['len'] = df_temp['l'].map(len)

    group_stats = []

    for length, group in df_temp.groupby('len'):
        # Convert lists to a single numpy array (Rectangular block)
        p_np = np.array(group['p'].tolist())
        l_np = np.array(group['l'].tolist())

        # Convert to Tensor and move to MPS
        p_tensor = torch.from_numpy(p_np).to(dtype=torch.float32, device=device)
        l_tensor = torch.from_numpy(l_np).to(dtype=torch.float32, device=device)

        group_stats.append({
            'count': len(group),
            'cos': calculate_batch_cosine_similarity(p_tensor, l_tensor),
            'dot': calculate_batch_normalized_dot_product(p_tensor, l_tensor, device)
        })

    # Weighted average calculation
    total_samples = sum(s['count'] for s in group_stats)
    avg_cos = sum(s['cos'] * s['count'] for s in group_stats) / total_samples
    avg_dot = sum(s['dot'] * s['count'] for s in group_stats) / total_samples

    return {
        'CosSim': round(avg_cos, ROUND_DIGITS),
        'DotProduct': round(avg_dot, ROUND_DIGITS)
    }


def calculate_disease_metrics(y_pred: list[list[str]], y_true: list[str], device) -> dict:

    y_pred_top1 = [p[0] if (isinstance(p, list) and len(p) > 0) else "" for p in y_pred]

    le = LabelEncoder()
    le.fit(list(y_pred_top1) + list(y_true))

    y_pred_idx = torch.tensor(le.transform(y_pred_top1), dtype=torch.long, device=device)
    y_true_idx = torch.tensor(le.transform(y_true), dtype=torch.long, device=device)
    num_classes = len(le.classes_)

    # Retrieval metrics (hand-rolled)
    mrr = calculate_mrr(y_pred, y_true)

    # Recall@k == Hit Rate@k here, because each row has exactly one gold
    # `disease` label (not a set): "does the true label appear in the top-k
    # predictions" is precision@k, recall@k, and accuracy@k all at once when
    # there's a single relevant item per query. calculate_accuracy_at_k is
    # that same hit-or-miss fraction, just under an older name. V2 Accuracy@1
    # is kept for backward compat with existing wandb history/paper tables;
    # V2 Recall@1 is the identical number under the name we asked for
    # (2026-07-24), added alongside @3/@5/@10 -- v2 prompts ask the model for
    # a ranked top-10 (src/pipeline/verifier_args.py num_choices=10), so @10
    # isn't trivially 100%. Purpose: check whether V2's accuracy drop-off on
    # bigger models is a true miss or a near-miss the model gets right if you
    # look past rank 1.
    recall_at_k = {k: calculate_accuracy_at_k(y_pred, y_true, k=k) for k in (1, 3, 5, 10)}

    return {
        "V2 F1 Micro@1": round(
            multiclass_f1_score(y_pred_idx, y_true_idx, num_classes=num_classes,
                                average='micro').item(), ROUND_DIGITS),
        "V2 F1 Macro@1": round(
            multiclass_f1_score(y_pred_idx, y_true_idx, num_classes=num_classes,
                                average='macro').item(), ROUND_DIGITS),
        "V2 MRR": round(mrr, 4),
        "V2 Accuracy@1": round(recall_at_k[1], 4),
        "V2 Recall@1": round(recall_at_k[1], 4),
        "V2 Recall@3": round(recall_at_k[3], 4),
        "V2 Recall@5": round(recall_at_k[5], 4),
        "V2 Recall@10": round(recall_at_k[10], 4),
    }

def calculate_icd_metrics_cpu(y_pred, y_true):
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for pred, true in zip(y_pred, y_true):
        pred_set = set(pred)
        true_set = set(true)

        for code in pred_set & true_set:
            tp[code] += 1

        for code in pred_set - true_set:
            fp[code] += 1

        for code in true_set - pred_set:
            fn[code] += 1

    labels = set(tp) | set(fp) | set(fn)

    # micro
    TP = sum(tp.values())
    FP = sum(fp.values())
    FN = sum(fn.values())

    precision_micro = TP / (TP + FP) if TP + FP else 0
    recall_micro = TP / (TP + FN) if TP + FN else 0
    f1_micro = (
        2 * precision_micro * recall_micro / (precision_micro + recall_micro)
        if precision_micro + recall_micro else 0
    )

    # macro
    precisions = []
    recalls = []
    f1s = []

    for label in labels:
        p = tp[label] / (tp[label] + fp[label]) if tp[label] + fp[label] else 0
        r = tp[label] / (tp[label] + fn[label]) if tp[label] + fn[label] else 0
        f1 = 2*p*r/(p+r) if p+r else 0

        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

    precision_macro = sum(precisions) / len(precisions)
    recall_macro = sum(recalls) / len(recalls)
    f1_macro = sum(f1s) / len(f1s)

    return {
        "ICD F1 Micro": round(f1_micro, ROUND_DIGITS),
        "ICD F1 Macro": round(f1_macro, ROUND_DIGITS),
        # Added 2026-07-22 (were computed but dropped from the return value
        # before): same tp/fp/fn already computed above, just not previously
        # surfaced. Enables direct precision/recall comparison between
        # encoder classifiers and generative LLMs -- both are scored from
        # discrete predicted-code sets via this same function, unlike AUROC
        # (needs probability scores, only available for encoders) which was
        # dropped from the encoder-vs-LLM table for exactly that reason.
        "ICD Precision Micro": round(precision_micro, ROUND_DIGITS),
        "ICD Recall Micro": round(recall_micro, ROUND_DIGITS),
        "ICD Precision Macro": round(precision_macro, ROUND_DIGITS),
        "ICD Recall Macro": round(recall_macro, ROUND_DIGITS),
    }
