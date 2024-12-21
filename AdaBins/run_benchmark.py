import os
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm
from infer import InferenceHelper
from metrics import compute_metrics


def load_image(image_path):
    """Load an image as a numpy array."""
    return np.asarray(Image.open(image_path), dtype='float32') / 255.0


def load_depth(depth_path):
    """Load a depth map as a numpy array."""
    return np.asarray(Image.open(depth_path), dtype='float32')


def run_benchmark(test_dir, gt_dir, dataset='nyu', output_dir='outputs'):
    """
    Run benchmark on a dataset and compute metrics.

    Parameters:
    - test_dir: Directory containing input RGB images.
    - gt_dir: Directory containing ground truth depth maps.
    - dataset: Dataset name ('nyu' or 'kitti').
    - output_dir: Directory to save predicted depth maps.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Initialize the inference helper
    infer_helper = InferenceHelper(dataset=dataset)

    # Collect all test images
    test_images = glob.glob(os.path.join(test_dir, "*"))

    metrics_list = []  # Store metrics for all images
    for image_path in tqdm(test_images, desc=f"Benchmarking on {dataset}"):
        # Load input image and ground truth depth map
        image = load_image(image_path)
        depth_path = os.path.join(gt_dir, os.path.basename(image_path))
        gt_depth = load_depth(depth_path)

        # Predict depth map
        _, pred_depth = infer_helper.predict_pil(Image.fromarray((image * 255).astype('uint8')))

        # Scale prediction to match dataset requirements
        if dataset == 'nyu':
            pred_depth *= 1000  # NYU uses millimeters
        elif dataset == 'kitti':
            pred_depth *= 256  # KITTI scaling factor

        # Compute metrics
        metrics = compute_metrics(pred_depth, gt_depth)
        metrics_list.append(metrics)

        # Save the predicted depth map
        output_path = os.path.join(output_dir, os.path.basename(image_path))
        pred_depth = (pred_depth).astype('uint16')
        Image.fromarray(pred_depth.squeeze()).save(output_path)

    # Compute average metrics
    avg_metrics = {k: np.mean([m[k] for m in metrics_list]) for k in metrics_list[0]}
    print(f"\n=== Benchmark Results on {dataset} ===")
    for k, v in avg_metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    import argparse

    # Argument parser
    parser = argparse.ArgumentParser(description="Run benchmark for AdaBins on NYU or KITTI datasets.")
    parser.add_argument("--test_dir", type=str, required=True, help="Path to the directory containing test RGB images.")
    parser.add_argument("--gt_dir", type=str, required=True, help="Path to the directory containing ground truth depth maps.")
    parser.add_argument("--dataset", type=str, choices=["nyu", "kitti"], default="nyu", help="Dataset name: 'nyu' or 'kitti'.")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save predicted depth maps.")

    args = parser.parse_args()

    # Run benchmark
    run_benchmark(args.test_dir, args.gt_dir, dataset=args.dataset, output_dir=args.output_dir)
