import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from model import create_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm

# Configuration
KITTI_RGB_DIR = "../KITTI/val_selection_cropped/image"
KITTI_DEPTH_DIR = "../KITTI/val_selection_cropped/groundtruth_depth"
NYU_MAT_FILE = "../nyu_depth_v2_labeled.mat"
MODEL_PATH = "./fine_tuned_kitti/densedepth_kitti.h5"  # Update with your model path
STEREO_SCALE_FACTOR = 5.4  # For KITTI
MAX_IMAGES = 200  # Number of images to evaluate

# Utility Functions
def compute_metrics(gt_depth, pred_depth):
    """Compute MAE, RMSE, and δ1.25 metrics."""
    valid_mask = (gt_depth > 0) & (gt_depth < np.max(gt_depth))  # Ignore invalid depths
    gt_depth = gt_depth[valid_mask]
    pred_depth = pred_depth[valid_mask]

    mae = mean_absolute_error(gt_depth, pred_depth)
    rmse = np.sqrt(mean_squared_error(gt_depth, pred_depth))
    delta_1_25 = np.mean((np.maximum(gt_depth / pred_depth, pred_depth / gt_depth) < 1.25).astype(np.float32))

    return mae, rmse, delta_1_25


def benchmark_kitti(model):
    """Run evaluation on KITTI dataset."""
    print("\n=== Benchmarking on KITTI Dataset ===")

    rgb_files = sorted(os.listdir(KITTI_RGB_DIR))[:MAX_IMAGES]
    depth_files = [f.replace("_image", "_groundtruth_depth") for f in rgb_files]

    total_metrics = []
    for rgb_file, depth_file in tqdm(zip(rgb_files, depth_files), total=len(rgb_files), desc="Running KITTI"):
        # Load RGB image
        rgb_path = os.path.join(KITTI_RGB_DIR, rgb_file)
        depth_path = os.path.join(KITTI_DEPTH_DIR, depth_file)

        rgb = plt.imread(rgb_path) / 255.0  # Normalize RGB
        gt_depth = plt.imread(depth_path).astype(np.float32) / 256.0  # Normalize depth

        # Predict depth
        pred_depth = model.predict(np.expand_dims(rgb, axis=0))[0, :, :, 0]
        pred_depth *= STEREO_SCALE_FACTOR  # Scale to match KITTI metric depth

        # Compute metrics
        metrics = compute_metrics(gt_depth, pred_depth)
        total_metrics.append(metrics)

    # Average metrics
    avg_metrics = np.mean(total_metrics, axis=0)
    print("\nKITTI Results:")
    print(f"MAE: {avg_metrics[0]:.4f}")
    print(f"RMSE: {avg_metrics[1]:.4f}")
    print(f"Threshold (δ=1.25): {avg_metrics[2]:.4f}")


def benchmark_nyu(model):
    """Run evaluation on NYUv2 dataset."""
    print("\n=== Benchmarking on NYUv2 Dataset ===")

    import h5py
    with h5py.File(NYU_MAT_FILE, "r") as f:
        rgb_images = np.array(f["images"])[:, :, :, :MAX_IMAGES]  # Limit to MAX_IMAGES
        depth_maps = np.array(f["depths"])[:, :, :MAX_IMAGES]

    total_metrics = []
    for i in tqdm(range(rgb_images.shape[-1]), desc="Running NYUv2"):
        rgb = np.transpose(rgb_images[:, :, :, i], (1, 2, 0)) / 255.0  # Normalize RGB
        gt_depth = depth_maps[:, :, i]

        # Predict depth
        pred_depth = model.predict(np.expand_dims(rgb, axis=0))[0, :, :, 0]

        # Compute metrics
        metrics = compute_metrics(gt_depth, pred_depth)
        total_metrics.append(metrics)

    # Average metrics
    avg_metrics = np.mean(total_metrics, axis=0)
    print("\nNYUv2 Results:")
    print(f"MAE: {avg_metrics[0]:.4f}")
    print(f"RMSE: {avg_metrics[1]:.4f}")
    print(f"Threshold (δ=1.25): {avg_metrics[2]:.4f}")


if __name__ == "__main__":
    # Load the fine-tuned model
    model = load_model(MODEL_PATH, compile=False)

    # Run benchmarks
    benchmark_kitti(model)
    benchmark_nyu(model)
