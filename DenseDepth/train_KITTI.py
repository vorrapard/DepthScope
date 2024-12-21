import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import Sequence
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.models import load_model
from model import create_model  # Import DenseDepth model

# Configuration
DATA_PATH = "../KITTI/val_selection_cropped/"
SAVE_DIR = "./fine_tuned_kitti/"
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
EPOCHS = 100
VALID_SPLIT = 0.2
MAX_IMAGES = 200  # Limit images to speed up training

# Dataset
class KITTIDataset(Sequence):
    def __init__(self, image_dir, depth_dir, batch_size, split="train", valid_split=0.2):
        self.image_dir = image_dir
        self.depth_dir = depth_dir
        self.image_files = sorted(os.listdir(image_dir))[:MAX_IMAGES]
        self.depth_files = [f.replace("_image", "_groundtruth_depth") for f in self.image_files]
        self.batch_size = batch_size
        self.split = split
        split_idx = int(len(self.image_files) * (1 - valid_split))
        if split == "train":
            self.image_files = self.image_files[:split_idx]
            self.depth_files = self.depth_files[:split_idx]
        elif split == "val":
            self.image_files = self.image_files[split_idx:]
            self.depth_files = self.depth_files[split_idx:]

    def __len__(self):
        return len(self.image_files) // self.batch_size

    def __getitem__(self, idx):
        batch_images = self.image_files[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_depths = self.depth_files[idx * self.batch_size:(idx + 1) * self.batch_size]

        images = []
        depths = []
        for img_file, depth_file in zip(batch_images, batch_depths):
            img = plt.imread(os.path.join(self.image_dir, img_file))
            depth = plt.imread(os.path.join(self.depth_dir, depth_file)) / 256.0  # Normalize depth

            images.append(img)
            depths.append(depth[:, :, np.newaxis])  # Add channel dimension

        images = np.array(images, dtype=np.float32) / 255.0  # Normalize RGB images
        depths = np.array(depths, dtype=np.float32)

        return images, depths


def lr_scheduler(epoch, lr):
    """Learning rate scheduler: decay by 0.1 every 20 epochs."""
    if epoch % 20 == 0 and epoch != 0:
        return lr * 0.1
    return lr


def train():
    # Prepare dataset
    train_dataset = KITTIDataset(
        image_dir=os.path.join(DATA_PATH, "image"),
        depth_dir=os.path.join(DATA_PATH, "groundtruth_depth"),
        batch_size=BATCH_SIZE,
        split="train",
    )
    val_dataset = KITTIDataset(
        image_dir=os.path.join(DATA_PATH, "image"),
        depth_dir=os.path.join(DATA_PATH, "groundtruth_depth"),
        batch_size=BATCH_SIZE,
        split="val",
    )

    # Create model
    model = create_model()
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss=MeanAbsoluteError())

    # Callbacks
    os.makedirs(SAVE_DIR, exist_ok=True)
    checkpoint = ModelCheckpoint(
        filepath=os.path.join(SAVE_DIR, "densedepth_kitti.h5"),
        save_best_only=True,
        monitor="val_loss",
        mode="min",
    )
    lr_schedule = LearningRateScheduler(lr_scheduler)

    # Train model
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=[checkpoint, lr_schedule],
        verbose=1,
    )

    # Plot training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(history.history["loss"], label="Train Loss", color="b")
    plt.plot(history.history["val_loss"], label="Validation Loss", color="r")
    plt.xlabel("Epochs", fontsize=16)
    plt.ylabel("Loss", fontsize=16)
    plt.title("Training and Validation Loss (KITTI)", fontsize=18)
    plt.legend(fontsize=14)
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "train_val_loss_plot.png"))
    plt.show()


if __name__ == "__main__":
    train()
