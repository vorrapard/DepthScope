import os
import glob
import numpy as np
from PIL import Image
from scipy.io import loadmat
from tqdm import tqdm
import torch
from torchvision import transforms
from torchvision.transforms.functional import resize

from networks import ResnetEncoder, DepthDecoder
from evaluate_depth import compute_errors


class Monodepth2Benchmark:
    def __init__(self, model_name="mono_640x192", device="cuda"):
        """Initialize the Monodepth2 model."""
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.load_model()

    def load_model(self):
        """Load the pre-trained Monodepth2 model."""
        print("-> Loading pre-trained Monodepth2 model...")
        model_dir = os.path.join("models", self.model_name)
        encoder_path = os.path.join(model_dir, "encoder.pth")
        decoder_path = os.path.join(model_dir, "depth.pth")

        # Load encoder
        self.encoder = ResnetEncoder(18, False)
        loaded_dict_enc = torch.load(encoder_path, map_location=self.device)
        self.encoder.load_state_dict({k: v for k, v in loaded_dict_enc.items() if k in self.encoder.state_dict()})
        self.encoder.to(self.device).eval()

        # Load decoder
        self.depth_decoder = DepthDecoder(num_ch_enc=self.encoder.num_ch_enc, scales=range(4))
        loaded_dict = torch.load(decoder_path, map_location=self.device)
        self.depth_decoder.load_state_dict(loaded_dict)
        self.depth_decoder.to(self.device).eval()

        print("-> Model Loaded Successfully")

    def predict_depth(self, image, stereo_scale_factor=1.0):
        """Predict depth for a single image and scale it correctly."""
        input_image = image.resize((640, 192), Image.LANCZOS)
        input_tensor = transforms.ToTensor()(input_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.encoder(input_tensor)
            outputs = self.depth_decoder(features)
            disp = outputs[("disp", 0)].squeeze().cpu().numpy()
            pred_depth = 1 / disp  # Convert disparity to depth

        # Apply stereo scaling factor for KITTI (or set it to 1.0 for other datasets)
        return pred_depth * stereo_scale_factor


    def benchmark_nyu(self, mat_file, max_images=200):
        """Run benchmark on NYUv2 dataset."""
        print("\n=== Benchmarking on NYUv2 Dataset ===")
        print("Loading NYUv2 .mat file...")
        with h5py.File(mat_file, 'r') as f:
            rgb_images = np.array(f['images'])  # Shape: (3, H, W, N)
            depth_maps = np.array(f['depths'])  # Shape: (H, W, N)

        total_metrics = []

        for i in tqdm(range(min(max_images, rgb_images.shape[-1])), desc="Running NYUv2"):
            # Extract and reshape RGB image
            rgb = np.transpose(rgb_images[:, :, :, i], (1, 2, 0))  # (H, W, 3)
            rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)  # Normalize and scale
            rgb_image = Image.fromarray(rgb, mode='RGB')  # Convert to PIL Image

            # Extract ground truth depth
            gt_depth = depth_maps[:, :, i] * 1000.0  # Ground truth depth in mm

            # Predict depth
            pred_depth = self.predict_depth(rgb_image) * 1000.0  # Scale to millimeters

            # Resize prediction to match ground truth size
            pred_depth_resized = resize(torch.tensor(pred_depth).unsqueeze(0), gt_depth.shape, antialias=True)
            pred_depth_resized = pred_depth_resized.squeeze().numpy()

            # Compute metrics
            valid_mask = (gt_depth > 0) & (gt_depth < 10000)
            metrics = compute_errors(gt_depth[valid_mask] / 1000.0, pred_depth_resized[valid_mask] / 1000.0)
            mae, rmse, thresh_1_25 = metrics[:3]  # Extract the first 3 metrics
            total_metrics.append((mae, rmse, thresh_1_25))

        # Average metrics
        avg_mae = np.mean([m[0] for m in total_metrics])
        avg_rmse = np.mean([m[1] for m in total_metrics])
        avg_thresh_1_25 = np.mean([m[2] for m in total_metrics])

        print("\nNYUv2 Results (20 images):")
        print(f"MAE: {avg_mae:.4f}")
        print(f"RMSE: {avg_rmse:.4f}")
        print(f"Threshold (δ=1.25): {avg_thresh_1_25:.4f}")




    def benchmark_kitti(self, rgb_dir, depth_dir, max_images=200):
        """Run benchmark on KITTI dataset."""
        print("\n=== Benchmarking on KITTI Dataset ===")
        rgb_files = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))[:max_images]
        depth_files = sorted(glob.glob(os.path.join(depth_dir, "*.png")))[:max_images]

        STEREO_SCALE_FACTOR = 5.4  # Scaling factor for KITTI stereo-trained models
        total_metrics = []

        for rgb_file, depth_file in tqdm(zip(rgb_files, depth_files), total=len(rgb_files), desc="Running KITTI"):
            # Load RGB image and ground truth depth map
            rgb_image = Image.open(rgb_file).convert("RGB")
            gt_depth = np.array(Image.open(depth_file)).astype(np.float32) / 256.0  # Ground truth depth in meters

            # Predict depth
            pred_depth = self.predict_depth(rgb_image)
            pred_depth = pred_depth * STEREO_SCALE_FACTOR  # Scale predicted depth to meters

            # Resize prediction to match ground truth size
            pred_depth_resized = resize(torch.tensor(pred_depth).unsqueeze(0), gt_depth.shape, antialias=True)
            pred_depth_resized = pred_depth_resized.squeeze().numpy()

            # Compute metrics
            valid_mask = (gt_depth > 0) & (gt_depth < 80)  # Valid KITTI depth range
            metrics = compute_errors(gt_depth[valid_mask], pred_depth_resized[valid_mask])
            mae, rmse, thresh_1_25 = metrics[:3]  # Extract the first 3 metrics
            total_metrics.append((mae, rmse, thresh_1_25))

        # Average metrics
        avg_mae = np.mean([m[0] for m in total_metrics])
        avg_rmse = np.mean([m[1] for m in total_metrics])
        avg_thresh_1_25 = np.mean([m[2] for m in total_metrics])

        print("\nKITTI Results (200 images):")
        print(f"MAE: {avg_mae:.4f}")
        print(f"RMSE: {avg_rmse:.4f}")
        print(f"Threshold (δ=1.25): {avg_thresh_1_25:.4f}")




if __name__ == "__main__":
    import h5py

    # Paths
    NYU_MAT_FILE = "../nyu_depth_v2_labeled.mat"  # Path to NYUv2 .mat file
    KITTI_RGB_DIR = "../KITTI/val_selection_cropped/image"  # KITTI RGB images
    KITTI_DEPTH_DIR = "../KITTI/val_selection_cropped/groundtruth_depth"  # KITTI Depth maps

    # Initialize and Run Benchmark
    benchmark = Monodepth2Benchmark()
    benchmark.benchmark_nyu(NYU_MAT_FILE, max_images=20)
    benchmark.benchmark_kitti(KITTI_RGB_DIR, KITTI_DEPTH_DIR, max_images=20)
