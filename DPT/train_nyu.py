import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm
import matplotlib.pyplot as plt
from dpt.models import DPTDepthModel
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

# Configuration
DATA_PATH = "../nyu_depth_v2_labeled.mat"
SAVE_DIR = "./fine_tuned_dpt_nyuv2"
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
EPOCHS = 100
VALID_SPLIT = 0.2
MAX_IMAGES = 200

# Dataset Class for NYUv2
class NYUv2Dataset(Dataset):
    def __init__(self, mat_file, max_images=200):
        with h5py.File(mat_file, "r") as f:
            self.images = np.array(f["images"])[:, :, :, :max_images] / 255.0  # Normalize RGB
            self.depths = np.array(f["depths"])[:, :, :max_images]
        self.num_samples = self.images.shape[-1]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = self.images[:, :, :, idx].transpose(1, 2, 0)
        depth = self.depths[:, :, idx]
        return to_tensor(image), torch.tensor(depth).unsqueeze(0)

# Training Function
def train():
    # Dataset and DataLoader
    dataset = NYUv2Dataset(DATA_PATH, max_images=MAX_IMAGES)
    train_size = int(len(dataset) * (1 - VALID_SPLIT))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Model, Optimizer, and Scheduler
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DPTDepthModel(
        path=None,  # Initialize without pre-trained weights
        backbone="vitb_rn50_384",  # Hybrid Vision Transformer (ViT + ResNet)
        non_negative=True
    ).to(device)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.1)
    criterion = torch.nn.L1Loss()

    # Training Loop
    os.makedirs(SAVE_DIR, exist_ok=True)
    train_losses, val_losses = [], []
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for images, depths in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} - Training", leave=False):
            images, depths = images.to(device), depths.to(device)
            optimizer.zero_grad()
            preds = model(images)
            loss = criterion(preds, depths)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_losses.append(train_loss / len(train_loader))

        # Validation Loop
        model.eval()
        val_loss = 0
        with tqdm(val_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} - Validating", leave=False) as val_bar:
            with torch.no_grad():
                for images, depths in val_bar:
                    images, depths = images.to(device), depths.to(device)
                    preds = model(images)
                    loss = criterion(preds, depths)
                    val_loss += loss.item()
        val_losses.append(val_loss / len(val_loader))

        # Log Epoch Results
        print(f"Epoch {epoch + 1}/{EPOCHS}: Train Loss = {train_losses[-1]:.4f}, Val Loss = {val_losses[-1]:.4f}")
        scheduler.step()

    # Save Model
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "dpt_nyuv2.pth"))

    # Plot Training and Validation Loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss", color="blue")
    plt.plot(val_losses, label="Validation Loss", color="red")
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.title("Training and Validation Loss (NYUv2)", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "train_val_loss_plot.png"))
    plt.show()

if __name__ == "__main__":
    train()
