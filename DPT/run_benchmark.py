import os
import h5py
import numpy as np
import torch
from tqdm import tqdm
from torchvision.transforms.functional import to_tensor
from dpt.models import DPTDepthModel
from evaluate_depth import compute_errors  # Assuming `compute_errors` function exists
import matplotlib.pyplot as plt

# Configuration
NYU_MAT_FILE = "../nyu_depth_v2_labeled.mat"
MAX_IMAGES = 20
MODEL_PATH = "./fine_tuned_dpt_nyuv2/dpt_nyuv2.pth"

# Dataset Class for Benchmarking
class NYUv2Dataset:
    def __init__(self, mat_file, max_images=20):
        with h5py.File(mat_file, "r") as f:
            self.images = np.array(f["images"])[:, :, :, :max_images] / 255.0
            self.depths = np.array(f["depths"])[:, :, :max_images]
        self.num_samples = self.images.shape[-1]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = self.images[:, :, :, idx].transpose(1, 2, 0)
        depth = self.depths[:, :, idx]
        return to_tensor(image), depth

# Benchmark Function
def benchmark():
    # Dataset
    dataset = NYUv2Dataset(NYU_MAT_FILE, max_images=MAX_IMAGES)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DPTDepthModel(
        path=None,
        backbone="vitb_rn50_384",
        non_negative=True
    ).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # Metrics
    total_metrics = []
    with tqdm(range(len(dataset)), desc="Benchmarking NYUv2") as bar:
        for i in bar:
            image, gt_depth = dataset[i]
            image = image.unsqueeze(0).to(device)  # Add batch dimension
            pred_depth = model(image).squeeze().cpu().numpy()

            # Compute Errors
            valid_mask = (gt_depth > 0) & (gt_depth < 10)
            metrics = compute_errors(gt_depth[valid_mask], pred_depth[valid_mask])
            total_metrics.append(metrics)

    # Average Metrics
    avg_metrics = np.mean(total_metrics, axis=0)
    print("\nNYUv2 Benchmark Results:")
    print(f"MAE: {avg_metrics[0]:.4f}")
    print(f"RMSE: {avg_metrics[1]:.4f}")
    print(f"Threshold (δ=1.25): {avg_metrics[2]:.4f}")

if __name__ == "__main__":
    benchmark()
