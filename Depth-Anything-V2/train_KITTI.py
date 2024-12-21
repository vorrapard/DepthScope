import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm
from depth_anything_v2.dpt import DepthAnythingV2

# Dataset Class for KITTI
class KITTIDataset(Dataset):
    def __init__(self, image_dir, depth_dir, input_size=(224, 224)):
        self.image_dir = image_dir
        self.depth_dir = depth_dir
        self.input_size = input_size
        self.image_files = sorted(os.listdir(image_dir))
        self.depth_files = sorted(os.listdir(depth_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        depth_path = os.path.join(self.depth_dir, self.depth_files[idx])

        img = cv2.imread(img_path)[:, :, ::-1]  # Convert BGR to RGB
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

        # Resize images and depths
        img_resized = cv2.resize(img, self.input_size, interpolation=cv2.INTER_LINEAR)
        depth_resized = cv2.resize(depth, self.input_size, interpolation=cv2.INTER_NEAREST)

        # Normalize image
        img_resized = img_resized / 255.0
        img_tensor = to_tensor(img_resized)

        # Convert depth to tensor
        depth_tensor = torch.tensor(depth_resized / 256.0, dtype=torch.float32)

        return img_tensor, depth_tensor

# Fine-tuning Function
def fine_tune_kitti():
    IMAGE_DIR = "../KITTI/val_selection_cropped/image"
    DEPTH_DIR = "../KITTI/val_selection_cropped/groundtruth_depth"
    MODEL_SAVE_PATH = "./fine_tuned_depth_anything_kitti.pth"

    # Dataset and DataLoader
    dataset = KITTIDataset(IMAGE_DIR, DEPTH_DIR)
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
    epochs = 50
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, depths in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            images, depths = images.to(device), depths.to(device)

            # Forward
            preds = model(images).squeeze(1)
            loss = criterion(preds, depths)

            # Backward
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
    fine_tune_kitti()
