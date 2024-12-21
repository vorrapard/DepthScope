import numpy as np

def compute_errors(gt, pred):
    """
    Compute error metrics between predicted and ground truth depth maps.
    Args:
        gt (numpy.ndarray): Ground truth depth map.
        pred (numpy.ndarray): Predicted depth map.

    Returns:
        tuple: A tuple containing:
            - MAE (Mean Absolute Error)
            - RMSE (Root Mean Squared Error)
            - Threshold (δ < 1.25)
    """
    # Ensure no zero or infinite values
    valid_mask = (gt > 0) & (np.isfinite(gt)) & (np.isfinite(pred))
    gt = gt[valid_mask]
    pred = pred[valid_mask]

    # Compute absolute relative difference
    abs_rel = np.mean(np.abs(gt - pred) / gt)

    # Compute RMSE
    rmse = np.sqrt(np.mean((gt - pred) ** 2))

    # Compute threshold accuracy δ < 1.25
    thresh = np.maximum(gt / pred, pred / gt)
    delta_1_25 = np.mean(thresh < 1.25)

    return abs_rel, rmse, delta_1_25
