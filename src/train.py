import torch
import argparse

from model import AntBeeClassifier
from dataset import ClassDirectoryDataset
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


def get_args():
    parser = argparse.ArgumentParser(description="Training script for AntBeeClassifier")
    # Basic training params
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.01, help="learning rate", metavar="")
    parser.add_argument("-bs", "--batch_size", type=int, default=16, help="batch size", metavar="")
    parser.add_argument("-e", "--epochs", type=int, default=1000, help="number of epochs", metavar="")
    parser.add_argument("-nw", "--num_workers", type=int, default=0, help="dataloader num_workers", metavar="")

    # Data and save paths
    parser.add_argument("-dp", "--data_path", type=str, default="data/hymenoptera_data/train", help="training dataset path", metavar="")
    parser.add_argument("-sd", "--save_dir", type=str, default="experiments/checkpoints", help="checkpoints base directory", metavar="")
    parser.add_argument("-rd", "--runs_dir", type=str, default="experiments/runs", help="tensorboard runs base directory", metavar="")

    # Early stopping/scheduler knobs
    parser.add_argument("--patience", type=int, default=200, help="early stopping patience", metavar="")

    # Device
    parser.add_argument("--use_gpu", action="store_true", help="use GPU if available")

    return parser.parse_args()

def main():
    args = get_args()

    # ==== Training Configuration ====
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_name = f"exp-{current_time}"

    checkpoints_dir = os.path.join(args.save_dir, experiment_name)
    os.makedirs(checkpoints_dir, exist_ok=True)

    writer = SummaryWriter(os.path.join(args.runs_dir, experiment_name))

    # ==== Parameter Settings ====
    num_epochs = args.epochs
    batch_size = args.batch_size
    num_workers = args.num_workers
    learning_rate = args.learning_rate

    patience = args.patience
    best_val_loss = np.inf # Initialize best validation loss to infinity
    epochs_without_improvement = 0 # Record how many epochs val_loss hasn't decreased

    # ==== Device Selection ====
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # ==== Create dataset and dataloader ====
    dataset = ClassDirectoryDataset(args.data_path, ["jpg"])
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
    criterion = torch.nn.CrossEntropyLoss(reduction="sum") # 2. define the loss
    optimizer = Adam(model.parameters(), lr=learning_rate , weight_decay=1e-4) # 3. do the optimize work, add weight decay to prevent overfitting
    scheduler = ReduceLROnPlateau(optimizer)
    scaler = GradScaler()  # Used to scale loss to prevent underflow

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

        # avg_train_loss = train_loss / len(train_loader) # Divide total loss by number of batches to get average loss per batch
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

        # avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total

        scheduler.step(val_loss)

        # --- Model Saving ---
        # Save latest model
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
        }
        torch.save(checkpoint, os.path.join(checkpoints_dir, "latest_model.pth"))

        # Save best model (based on validation loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, os.path.join(checkpoints_dir, "best_model.pth"))
            epochs_without_improvement = 0
            print(f"New best model saved with val_loss: {val_loss:.4f}")
        else:
            epochs_without_improvement += 1

        # --- Log ---
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

        # --- Early Stopping Check ---
        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs!")
            break

    writer.close()

if __name__ == "__main__":
    main()