import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.utils import Sequence
from model import create_model  # Import DenseDepth model

# Configuration
DATA_PATH = "../nyu_depth_v2_labeled.mat"
SAVE_DIR = "./fine_tuned_nyu/"
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
EPOCHS = 100
VALID_SPLIT = 0.2
MAX_IMAGES = 200  # Limit images to speed up training

# Dataset
class NYUv2Dataset(Sequence):
    def __init__(self, mat_file, batch_size, split="train", valid_split=0.2):
        self.batch_size = batch_size
        self.split = split
        with h5py.File(mat_file, "r") as f:
            self.images = np.array(f["images"])  # (3, H, W, N)
            self.depths = np.array(f["depths"])  # (H, W, N)
        self.num_samples = self.images.shape[-1]
        split_idx = int(self.num_samples * (1 - valid_split))
        if split == "train":
            self.indices = np.arange(split_idx)
        elif split == "val":
            self.indices = np.arange(split_idx, self.num_samples)

    def __len__(self):
        return len(self.indices) // self.batch_size

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]

        images = []
        depths = []
        for i in batch_indices:
            img = np.transpose(self.images[:, :, :, i], (1, 2, 0)) / 255.0  # Normalize RGB
            depth = self.depths[:, :, i][:, :, np.newaxis]  # Add channel dimension
            images.append(img)
            depths.append(depth)

        return np.array(images, dtype=np.float32), np.array(depths, dtype=np.float32)


def lr_scheduler(epoch, lr):
    """Learning rate scheduler: decay by 0.1 every 20 epochs."""
    if epoch % 20 == 0 and epoch != 0:
        return lr * 0.1
    return lr


def train():
    # Prepare dataset
    train_dataset = NYUv2Dataset(DATA_PATH, batch_size=BATCH_SIZE, split="train")
    val_dataset = NYUv2Dataset(DATA_PATH, batch_size=BATCH_SIZE, split="val")

    # Create model
    model = create_model()
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss=MeanAbsoluteError())

    # Callbacks
    os.makedirs(SAVE_DIR, exist_ok=True)
    checkpoint = ModelCheckpoint(
        filepath=os.path.join(SAVE_DIR, "densedepth_nyu.h5"),
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
    plt.title("Training and Validation Loss (NYUv2)", fontsize=18)
    plt.legend(fontsize=14)
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "train_val_loss_plot.png"))
    plt.show()


if __name__ == "__main__":
    train()
