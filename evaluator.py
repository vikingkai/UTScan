# evaluator.py
# Provides functions for model evaluation metrics calculation
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def compute_metrics(y_true, y_pred):
    """
    Compute basic regression evaluation metrics: RMSE, R2, and Pearson correlation coefficient.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred)) if len(y_true) > 0 else np.nan
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan
    if len(y_true) > 1:
        pearson = np.corrcoef(y_true, y_pred)[0, 1]
    else:
        pearson = np.nan
    return {
        'RMSE': rmse,
        'R2': r2,
        'Pearson': pearson
    }


def compute_metrics2(y_true, y_pred):
    """
    Compute extended regression evaluation metrics: RMSE, MAE, R2, MAPE, and Pearson correlation coefficient.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    # Root Mean Squared Error
    rmse = np.sqrt(mean_squared_error(y_true, y_pred)) if len(y_true) > 0 else np.nan
    # Mean Absolute Error
    mae = mean_absolute_error(y_true, y_pred) if len(y_true) > 0 else np.nan
    # Coefficient of Determination (R2)
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan
    # Mean Absolute Percentage Error
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if len(y_true) > 0 else np.nan
    # Pearson correlation coefficient
    if len(y_true) > 1:
        pearson = np.corrcoef(y_true, y_pred)[0, 1]
    else:
        pearson = np.nan
    return {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape,
        'Pearson': pearson
    }
