import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import cv2
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm
from depth_anything_v2.dpt import DepthAnythingV2

# Dataset Class for NYUv2
class NYUDepthDataset(Dataset):
    def __init__(self, images, depths, input_size=(224, 224)):
        self.images = images
        self.depths = depths
        self.input_size = input_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
        depth = self.depths[idx]

        img_resized = cv2.resize(img, self.input_size, interpolation=cv2.INTER_LINEAR)
        depth_resized = cv2.resize(depth, self.input_size, interpolation=cv2.INTER_NEAREST)

        img_resized = img_resized / 255.0  # Normalize to [0, 1]
        img_tensor = to_tensor(img_resized)

        depth_tensor = torch.tensor(depth_resized, dtype=torch.float32)
        return img_tensor, depth_tensor

# Fine-tuning Function
def fine_tune_nyu():
    MAT_FILE = "../nyu_depth_v2_labeled.mat"
    MODEL_SAVE_PATH = "./fine_tuned_depth_anything_nyu.pth"

    # Load Data
    with h5py.File(MAT_FILE, "r") as f:
        images = f["images"][:500]
        depths = f["depths"][:500]

    # Datasets and DataLoader
    dataset = NYUDepthDataset(images, depths)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # Model Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DepthAnythingV2(encoder="vitb", features=128, out_channels=[96, 192, 384, 768])
    model.load_state_dict(torch.load("checkpoints/depth_anything_v2_vitb.pth", map_location=device))
    model.to(device)

    # Freeze encoder
    for name, param in model.named_parameters():
        if "pretrained" in name:
            param.requires_grad = False

    # Loss and Optimizer
    criterion = nn.L1Loss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    # Fine-tuning Loop
    epochs = 100
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, depths in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            images, depths = images.to(device), depths.to(device)

            preds = model(images).squeeze(1)
            loss = criterion(preds, depths)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation Loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, depths in val_loader:
                images, depths = images.to(device), depths.to(device)
                preds = model(images).squeeze(1)
                loss = criterion(preds, depths)
                val_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}: Train Loss = {running_loss / len(train_loader):.4f}, Val Loss = {val_loss / len(val_loader):.4f}")

    # Save the Model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved at {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    fine_tune_nyu()
