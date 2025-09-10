# utils/metrics.py
import torch
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score
import numpy as np


def calculate_metrics(all_targets, all_predictions, average='weighted'):
    if isinstance(all_targets, torch.Tensor):
        all_targets = all_targets.cpu().numpy()
    if isinstance(all_predictions, torch.Tensor):
        all_predictions = all_predictions.cpu().numpy()

    precision, recall, f1, support = precision_recall_fscore_support(
        all_targets,
        all_predictions,
        average=average,
        zero_division=0
    )

    accuracy = accuracy_score(all_targets, all_predictions)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'support': support
    }


def calculate_per_class_accuracy(all_targets, all_predictions, num_classes: int):
    if isinstance(all_targets, torch.Tensor):
        all_targets = all_targets.cpu().numpy()
    if isinstance(all_predictions, torch.Tensor):
        all_predictions = all_predictions.cpu().numpy()

    cm = confusion_matrix(all_targets, all_predictions, labels=np.arange(num_classes))

    per_class_acc = np.zeros(num_classes, dtype=float)
    for i in range(num_classes):
        tp = cm[i, i]
        class_total_samples = np.sum(cm[i, :])
        if class_total_samples > 0:
            per_class_acc[i] = tp / class_total_samples
        else:
            per_class_acc[i] = 0.0

    return per_class_acc