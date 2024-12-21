import os
import torch
import numpy as np
from tqdm import tqdm
from torchvision.transforms.functional import resize, to_tensor
from depth_anything_v2.dpt import DepthAnythingV2
from scipy.io import loadmat
from PIL import Image

# Compute Evaluation Metrics
def compute_metrics(gt, pred):
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)
    rmse = np.sqrt(np.mean((gt - pred) ** 2))
    rmse_log = np.sqrt(np.mean((np.log(gt) - np.log(pred)) ** 2))
    delta_1 = np.mean((np.maximum(gt / pred, pred / gt) < 1.25).astype(np.float32))
    return abs_rel, sq_rel, rmse, rmse_log, delta_1

# Evaluate Depth Anything V2 on KITTI Dataset
def benchmark_kitti(model, image_dir, depth_dir, device, max_images=20):
    print("\n=== Benchmarking on KITTI Dataset ===")
    image_files = sorted(os.listdir(image_dir))[:max_images]
    depth_files = sorted(os.listdir(depth_dir))[:max_images]

    metrics = []
    for img_file, depth_file in tqdm(zip(image_files, depth_files), total=len(image_files), desc="Evaluating KITTI"):
        img_path = os.path.join(image_dir, img_file)
        depth_path = os.path.join(depth_dir, depth_file)

        # Load image and ground truth depth
        img = np.array(Image.open(img_path).convert("RGB"))
        gt_depth = np.array(Image.open(depth_path)).astype(np.float32) / 256.0  # Convert to meters

        # Resize and normalize image
        img_resized = resize(to_tensor(img / 255.0), (384, 384)).unsqueeze(0).to(device)

        # Predict depth
        with torch.no_grad():
            pred_depth = model(img_resized).squeeze(0).squeeze(0).cpu().numpy()
            pred_depth_resized = resize(torch.tensor(pred_depth), gt_depth.shape, antialias=True).numpy()

        # Compute metrics
        valid_mask = gt_depth > 0
        metrics.append(compute_metrics(gt_depth[valid_mask], pred_depth_resized[valid_mask]))

    avg_metrics = np.mean(metrics, axis=0)
    print("\nKITTI Results:")
    print(f"Abs Rel: {avg_metrics[0]:.4f}, Sq Rel: {avg_metrics[1]:.4f}, RMSE: {avg_metrics[2]:.4f}, RMSE Log: {avg_metrics[3]:.4f}, δ < 1.25: {avg_metrics[4]:.4f}")

# Evaluate Depth Anything V2 on NYUv2 Dataset
def benchmark_nyu(model, mat_file, device, max_images=20):
    print("\n=== Benchmarking on NYUv2 Dataset ===")
    data = loadmat(mat_file)
    images = data["images"][:, :, :, :max_images].transpose(3, 0, 1, 2)
    depths = data["depths"][:, :, :max_images]

    metrics = []
    for i in tqdm(range(images.shape[0]), desc="Evaluating NYUv2"):
        img = images[i] / 255.0
        gt_depth = depths[:, :, i]

        # Resize and normalize image
        img_resized = resize(to_tensor(img), (384, 384)).unsqueeze(0).to(device)

        # Predict depth
        with torch.no_grad():
            pred_depth = model(img_resized).squeeze(0).squeeze(0).cpu().numpy()
            pred_depth_resized = resize(torch.tensor(pred_depth), gt_depth.shape, antialias=True).numpy()

        # Compute metrics
        valid_mask = gt_depth > 0
        metrics.append(compute_metrics(gt_depth[valid_mask], pred_depth_resized[valid_mask]))

    avg_metrics = np.mean(metrics, axis=0)
    print("\nNYUv2 Results:")
    print(f"Abs Rel: {avg_metrics[0]:.4f}, Sq Rel: {avg_metrics[1]:.4f}, RMSE: {avg_metrics[2]:.4f}, RMSE Log: {avg_metrics[3]:.4f}, δ < 1.25: {avg_metrics[4]:.4f}")

# Main Function
if __name__ == "__main__":
    # Paths
    KITTI_IMAGE_DIR = "../KITTI/val_selection_cropped/image"
    KITTI_DEPTH_DIR = "../KITTI/val_selection_cropped/groundtruth_depth"
    NYU_MAT_FILE = "../nyu_depth_v2_labeled.mat"

    # Model Configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model = DepthAnythingV2(encoder="vitb", features=128, out_channels=[96, 192, 384, 768])
    model.load_state_dict(torch.load("checkpoints/depth_anything_v2_vitb.pth", map_location=DEVICE))
    model = model.to(DEVICE).eval()

    # Benchmark
    benchmark_kitti(model, KITTI_IMAGE_DIR, KITTI_DEPTH_DIR, DEVICE, max_images=20)
    benchmark_nyu(model, NYU_MAT_FILE, DEVICE, max_images=20)
