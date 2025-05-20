#!/usr/bin/env python3
"""
metrics_adni_mri.py

Comprehensive evaluation metrics for ADNI MRI classification and regression tasks.
Supports:
  - Multiclass classification (e.g., CN vs MCI vs AD)
  - Binary classification
  - Regression (e.g., age prediction)

Usage:
    from metrics_adni_mri import Metrics

    # classification
    y_pred = torch.tensor([...])  # logits or probabilities
    y_true = torch.tensor([...])  # integer labels
    acc = Metrics.accuracy(y_pred, y_true)
    prec = Metrics.precision(y_pred, y_true, average='macro')
    f1 = Metrics.f1_score(y_pred, y_true, average='macro')

    # binary AUC
    y_score = torch.sigmoid(logits)[:, 1]
    auc = Metrics.roc_auc(y_score, y_true)

    # regression
    y_reg_pred = torch.tensor([...])
    y_reg_true = torch.tensor([...])
    mae = Metrics.mae(y_reg_pred, y_reg_true)
    r2 = Metrics.r2_score(y_reg_pred, y_reg_true)
"""
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def _to_numpy(x):
    """
    Convert torch Tensor or numpy array to 1d numpy array.
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return x

class Metrics:
    @staticmethod
    def accuracy(y_pred, y_true):
        """
        Compute accuracy for multiclass or binary classification.

        Args:
            y_pred: logits or probabilities, shape (N, C) or (N,)
            y_true: true labels, shape (N,)
        """
        y_true_np = _to_numpy(y_true).astype(int)
        y_pred_np = _to_numpy(y_pred)
        # if logits or probabilities
        if y_pred_np.ndim > 1:
            y_pred_labels = np.argmax(y_pred_np, axis=1)
        else:
            # binary prediction threshold at 0.5
            y_pred_labels = (y_pred_np >= 0.5).astype(int)
        return accuracy_score(y_true_np, y_pred_labels)

    @staticmethod
    def precision(y_pred, y_true, average='binary'):
        """
        Precision score.

        average: 'binary', 'micro', 'macro', or 'weighted'.
        """
        y_true_np = _to_numpy(y_true).astype(int)
        y_pred_np = _to_numpy(y_pred)
        if y_pred_np.ndim > 1:
            y_pred_labels = np.argmax(y_pred_np, axis=1)
        else:
            y_pred_labels = (y_pred_np >= 0.5).astype(int)
        return precision_score(y_true_np, y_pred_labels, average=average, zero_division=0)

    @staticmethod
    def recall(y_pred, y_true, average='binary'):
        """
        Recall score.
        """
        y_true_np = _to_numpy(y_true).astype(int)
        y_pred_np = _to_numpy(y_pred)
        if y_pred_np.ndim > 1:
            y_pred_labels = np.argmax(y_pred_np, axis=1)
        else:
            y_pred_labels = (y_pred_np >= 0.5).astype(int)
        return recall_score(y_true_np, y_pred_labels, average=average, zero_division=0)

    @staticmethod
    def f1_score(y_pred, y_true, average='binary'):
        """
        F1 score.
        """
        y_true_np = _to_numpy(y_true).astype(int)
        y_pred_np = _to_numpy(y_pred)
        if y_pred_np.ndim > 1:
            y_pred_labels = np.argmax(y_pred_np, axis=1)
        else:
            y_pred_labels = (y_pred_np >= 0.5).astype(int)
        return f1_score(y_true_np, y_pred_labels, average=average, zero_division=0)

    @staticmethod
    def roc_auc(y_score, y_true, multi_class='ovr', average='macro'):
        """
        Compute ROC AUC score.

        y_score: probability estimates, shape (N, C) or (N,)
        y_true: true labels, shape (N,)
        """
        y_true_np = _to_numpy(y_true).astype(int)
        y_score_np = _to_numpy(y_score)
        return roc_auc_score(y_true_np, y_score_np, multi_class=multi_class, average=average)

    @staticmethod
    def confusion_matrix(y_pred, y_true, labels=None):
        """
        Confusion matrix.
        """
        y_true_np = _to_numpy(y_true).astype(int)
        y_pred_np = _to_numpy(y_pred)
        if y_pred_np.ndim > 1:
            y_pred_labels = np.argmax(y_pred_np, axis=1)
        else:
            y_pred_labels = (y_pred_np >= 0.5).astype(int)
        return confusion_matrix(y_true_np, y_pred_labels, labels=labels)

    @staticmethod
    def classification_report_dict(y_pred, y_true, target_names=None):
        """
        Return classification report as a dict.
        """
        from sklearn.metrics import classification_report
        y_true_np = _to_numpy(y_true).astype(int)
        y_pred_np = _to_numpy(y_pred)
        if y_pred_np.ndim > 1:
            y_pred_labels = np.argmax(y_pred_np, axis=1)
        else:
            y_pred_labels = (y_pred_np >= 0.5).astype(int)
        return classification_report(
            y_true_np,
            y_pred_labels,
            target_names=target_names,
            output_dict=True,
            zero_division=0
        )

    # Regression metrics
    @staticmethod
    def mse(y_pred, y_true):
        """
        Mean Squared Error.
        """
        y_true_np = _to_numpy(y_true).astype(float)
        y_pred_np = _to_numpy(y_pred).astype(float)
        return mean_squared_error(y_true_np, y_pred_np)

    @staticmethod
    def rmse(y_pred, y_true):
        """
        Root Mean Squared Error.
        """
        return np.sqrt(Metrics.mse(y_pred, y_true))

    @staticmethod
    def mae(y_pred, y_true):
        """
        Mean Absolute Error.
        """
        y_true_np = _to_numpy(y_true).astype(float)
        y_pred_np = _to_numpy(y_pred).astype(float)
        return mean_absolute_error(y_true_np, y_pred_np)

    @staticmethod
    def r2_score(y_pred, y_true):
        """
        R^2 Score.
        """
        y_true_np = _to_numpy(y_true).astype(float)
        y_pred_np = _to_numpy(y_pred).astype(float)
        return r2_score(y_true_np, y_pred_np)
