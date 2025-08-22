import torch

from src.model import AntBeeClassifier
from src.dataset import ClassDirectoryDataset
from torch.optim import Adam
from torch.utils.data import DataLoader,random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import autocast, GradScaler
from torch.amp.autocast_mode import is_autocast_available

from tqdm import tqdm
from tqdm import trange
import numpy as np
import os
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

# ==== Training Configuration ====
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
experiment_name = f"exp-{current_time}"

checkpoints_dir = f"checkpoints/{experiment_name}"
os.makedirs(checkpoints_dir, exist_ok=True)

writer = SummaryWriter(f"runs/{experiment_name}")

# ==== Parameter Settings ====
num_epochs = 100
batch_size = 32
num_workers = 2
learning_rate = 1e-3

patience = 30
best_val_loss = np.inf
epochs_without_improvement = 0 # Record how many epochs val_loss hasn't decreased

# ==== Device Selection ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== Create dataset and dataloader ====
dataset = ClassDirectoryDataset("../data/hymenoptera_data/train", ["jpg"])
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size],
                                  generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_set,
                          batch_size=batch_size,
                          num_workers=num_workers,
                          shuffle=True,
                          pin_memory=True  # Enable this option to accelerate data transfer if using GPU
)
val_loader = DataLoader(val_set,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        shuffle=True,
                        pin_memory=True
)

# ==== Model Instance ====
model = AntBeeClassifier(dropout_rate=0.2).to(device) # 1. build a model
# ==== Loss Function & Optimizer ====
criterion = torch.nn.CrossEntropyLoss() # 2. define the loss
optimizer = Adam(model.parameters(), lr=learning_rate , weight_decay=1e-4) # 3. do the optimize work, add weight decay to prevent overfitting
scheduler = ReduceLROnPlateau(optimizer, verbose=True)
scaler = GradScaler()  # Used to scale loss to prevent underflow

if __name__ == '__main__':
    print(f"Using device: {device}")
    
    if is_autocast_available(str(device)):
        print("Autocast available")
    else:
        print("Autocast not available")
    
    # ==== Training + Validation loop ====
    for epoch in range(num_epochs):
        # --- Training ---
        model.train()
        train_loss = 0.0 # Sum of losses from all batches in 1 epoch
        correct, total = 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()

            # Enables autocasting for the forward pass (model + loss)
            with autocast(device_type=str(device)):
                outputs = model(inputs)
                loss = criterion(outputs, labels) # Calculate cross entropy loss

            # Scales the loss, and calls backward() to create scaled gradients
            scaler.scale(loss).backward()

            # scaler.step() first unscales the gradients of the optimizer's assigned params.
            # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
            # otherwise, optimizer.step() is skipped.
            scaler.step(optimizer)

            # Updates the scale for next iteration
            scaler.update()

            train_loss += loss.item()
            predicted = outputs.argmax(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_train_loss = train_loss / len(train_loader) # Divide total loss by number of batches to get average loss per batch
        train_acc = correct / total

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        correct, total = 0, 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                predicted = outputs.argmax(1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total

        scheduler.step(avg_val_loss)

        # --- Model Saving ---
        # Save latest model
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
        }
        torch.save(checkpoint, os.path.join(checkpoints_dir, "latest_model.pth"))

        # Save best model (based on validation loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(checkpoint, os.path.join(checkpoints_dir, "best_model.pth"))
            epochs_without_improvement = 0
            print(f"New best model saved with val_loss: {avg_val_loss:.4f}")
        else:
            epochs_without_improvement += 1

        # --- Log ---
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

        # --- Early Stopping Check ---
        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs!")
            break

    writer.close()