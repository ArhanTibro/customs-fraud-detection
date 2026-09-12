# evaluation functions will go here
import os
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              average_precision_score)


def evaluate_model(model_name, variant, y_test, y_pred, y_proba,
                    metrics_path="../results/metrics.csv",
                    preds_path=None):
    """
    Computes standard classification metrics, appends them to the shared
    metrics.csv, optionally saves predictions for later statistical tests,
    and returns the metrics as a dict.
    """
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"PR-AUC:    {pr_auc:.4f}")
    print("Confusion Matrix:\n", cm)

    results_row = {
        "model": model_name,
        "variant": variant,
        "accuracy": acc, "precision": prec, "recall": rec,
        "f1": f1, "roc_auc": roc_auc, "pr_auc": pr_auc
    }

    if os.path.exists(metrics_path):
        metrics_df = pd.read_csv(metrics_path)
        metrics_df = pd.concat([metrics_df, pd.DataFrame([results_row])], ignore_index=True)
    else:
        metrics_df = pd.DataFrame([results_row])
    metrics_df.to_csv(metrics_path, index=False)

    if preds_path is not None:
        preds_df = pd.DataFrame({
            "y_true": y_test.values,
            f"y_pred_{model_name}": y_pred,
            f"y_proba_{model_name}": y_proba
        }, index=y_test.index)
        preds_df.to_csv(preds_path, index=False)

    return results_row, cm