import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def compute_metrics(y_true, y_pred, target_names=None):
    """
    Compute MAE, RMSE, and R^2 for each target variable.
    """
    num_targets = y_true.shape[2]
    if target_names is None:
        target_names = [f"target_{i}" for i in range(num_targets)]

    metrics_list = []

    for i, name in enumerate(target_names):
        y_t = y_true[:, :, i].ravel()
        y_p = y_pred[:, :, i].ravel()

        metrics_list.append({
            'target_variable': name,
            'MAE': round(mean_absolute_error(y_t, y_p), 4),
            'RMSE': round(np.sqrt(mean_squared_error(y_t, y_p)), 4),
            'R2': round(r2_score(y_t, y_p), 4)
        })

    return pd.DataFrame(metrics_list)


def save_metrics_to_csv(metrics_df, path):
    """
    Save the computed metrics DataFrame to a CSV file.
    """
    metrics_df.to_csv(path, index=False)
