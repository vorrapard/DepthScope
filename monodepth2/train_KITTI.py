import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, Dataset, random_split
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm  # Progress bar
import matplotlib.pyplot as plt  # For plotting graphs
from networks import ResnetEncoder, DepthDecoder

# Configuration
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
EPOCHS = 50
DATA_PATH = "../KITTI/val_selection_cropped/"
MODEL_DIR = "./models/mono_640x192"
SAVE_DIR = "./fine_tuned_kitti/"
MAX_BATCHES = 20  # Limit to 20 batches per epoch
VALID_SPLIT = 0.2  # 20% of the dataset for validation


# Set Seed for Reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Hybrid Loss: SSIM + L1 Loss
class SSIM(nn.Module):
    def __init__(self):
        super().__init__()
        self.window_size = 3
        self.window = self.create_window()

    def create_window(self):
        gaussian = torch.tensor([0.0113, 0.0838, 0.0113]).float()
        return torch.outer(gaussian, gaussian).unsqueeze(0).unsqueeze(0)

    def forward(self, x, y):
        window = self.window.to(x.device).type_as(x)
        mu_x = F.conv2d(x, window, padding=self.window_size // 2, groups=1)
        mu_y = F.conv2d(y, window, padding=self.window_size // 2, groups=1)
        sigma_x = F.conv2d(x ** 2, window, padding=self.window_size // 2, groups=1) - mu_x ** 2
        sigma_y = F.conv2d(y ** 2, window, padding=self.window_size // 2, groups=1) - mu_y ** 2
        sigma_xy = F.conv2d(x * y, window, padding=self.window_size // 2, groups=1) - mu_x * mu_y

        C1, C2 = 0.01 ** 2, 0.03 ** 2
        ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))
        return 1 - ssim.mean()


class HybridLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ssim = SSIM()
        self.l1_loss = nn.L1Loss()

    def forward(self, pred, gt):
        return 1.0 * self.ssim(pred, gt) + 0.1 * self.l1_loss(pred, gt)


# Custom Dataset for KITTI Depth
class KITTIDepthDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.image_dir = os.path.join(data_path, "image")
        self.depth_dir = os.path.join(data_path, "groundtruth_depth")
        self.image_files = sorted(os.listdir(self.image_dir))
        self.transform = transform
        self.depth_files = [
            f.replace("_image", "_groundtruth_depth", 1) for f in self.image_files
        ]

        self.resize = transforms.Resize((192, 640), antialias=True)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        depth_path = os.path.join(self.depth_dir, self.depth_files[idx])

        image = Image.open(image_path).convert("RGB")
        depth = Image.open(depth_path)
        depth = self.resize(depth)

        if self.transform:
            image = self.transform(image)
            depth = torch.from_numpy(np.array(depth)).unsqueeze(0).float() / 256.0

        return {"image": image, "depth": depth}


# Load Monodepth2 Pre-trained Model
def load_model(device):
    encoder = ResnetEncoder(18, False)
    encoder_dict = torch.load(os.path.join(MODEL_DIR, "encoder.pth"), map_location=device)
    filtered_dict = {k: v for k, v in encoder_dict.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict)
    encoder.to(device)

    decoder = DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))
    decoder_dict = torch.load(os.path.join(MODEL_DIR, "depth.pth"), map_location=device)
    decoder.load_state_dict(decoder_dict)
    decoder.to(device)

    return encoder, decoder


def train():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((192, 640)),
        transforms.ToTensor()
    ])
    print("-> Loading KITTI Dataset...")
    full_dataset = KITTIDepthDataset(DATA_PATH, transform=transform)

    # Ensure MAX_BATCHES does not exceed dataset size
    total_samples = min(MAX_BATCHES * BATCH_SIZE, len(full_dataset))
    train_size = int((1 - VALID_SPLIT) * total_samples)
    val_size = total_samples - train_size

    subset_indices = list(range(total_samples))
    limited_dataset = Subset(full_dataset, subset_indices)
    train_dataset, val_dataset = random_split(limited_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print(f"Training Samples: {len(train_dataset)}, Validation Samples: {len(val_dataset)}")

    encoder, decoder = load_model(device)
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=LEARNING_RATE)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.1)
    criterion = HybridLoss()

    train_losses, val_losses = [], []

    for epoch in range(EPOCHS):
        encoder.train()
        decoder.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{EPOCHS}"):
            optimizer.zero_grad()
            images = batch["image"].to(device)
            depths = batch["depth"].to(device)

            features = encoder(images)
            outputs = decoder(features)
            pred_depth = outputs[("disp", 0)]

            loss = criterion(pred_depth, depths)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation Loop
        encoder.eval()
        decoder.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                depths = batch["depth"].to(device)

                features = encoder(images)
                outputs = decoder(features)
                pred_depth = outputs[("disp", 0)]

                loss = criterion(pred_depth, depths)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        scheduler.step()
        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

     # Plot Loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, EPOCHS + 1), train_losses, label="Train Loss", color="b")
    plt.plot(range(1, EPOCHS + 1), val_losses, label="Validation Loss", color="r")
    plt.xlabel("Epochs", fontsize=16)  # Larger font for x-axis label
    plt.ylabel("Loss", fontsize=16)    # Larger font for y-axis label
    plt.title("Training and Validation Loss", fontsize=18)  # Larger font for title
    plt.legend(fontsize=14)  # Larger font for legend
    plt.xticks(fontsize=12)  # Larger font for x-axis ticks
    plt.yticks(fontsize=12)  # Larger font for y-axis ticks
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "train_val_loss_plot.png"))
    plt.show()


if __name__ == "__main__":
    train()
