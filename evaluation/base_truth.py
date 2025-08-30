def weighted_multilabel_auc(y_true, y_pred):
    """
    y_true: Ground truth (n_samples, 14)
    y_pred: Predicted probabilities (n_samples, 14)
    """
    auc_ap = roc_auc_score(y_true[:, 0], y_pred[:, 0])  # Aneurysm Present (index 0)
    auc_vessels = np.mean([roc_auc_score(y_true[:, i], y_pred[:, i]) for i in range(1, 14)])  # 13 vessel classes
    final_score = 0.5 * (auc_ap + auc_vessels)
    return final_score

# Example usage:
y_true = np.random.randint(0, 2, size=(100, 14))  # Replace with real data
y_pred = np.random.rand(100, 14)                  # Replace with model outputs
print(weighted_multilabel_auc(y_true, y_pred))