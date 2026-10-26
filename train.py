"""
Training Pipeline for Human Activity Recognition (LRCN: CNN + BiLSTM).
Trains on video clips, logs accuracy/loss/F1 metrics per epoch,
saves the best model checkpoint, and outputs training loss & accuracy curves.
"""

import os
import argparse
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from model_architecture import LRCNActivityModel, save_checkpoint, DEFAULT_CLASSES, DEFAULT_IMAGE_SIZE, DEFAULT_SEQUENCE_LENGTH
from preprocess import get_data_loaders


def train_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n==========================================")
    print(f" Training Human Activity Recognition Model")
    print(f" Using Device: {device}")
    print(f" Architecture: LRCN (CNN + Bi-LSTM)")
    print(f" Epochs: {args.epochs} | Batch Size: {args.batch_size} | LR: {args.lr}")
    print(f"==========================================\n")

    train_loader, val_loader, test_loader, class_names = get_data_loaders(
        dataset_root=args.dataset_root,
        class_names=DEFAULT_CLASSES,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        img_size=(args.img_size, args.img_size)
    )

    num_classes = len(class_names)
    print(f"Loaded {len(train_loader.dataset)} training samples, {len(val_loader.dataset)} validation samples.")
    print(f"Classes ({num_classes}): {class_names}\n")

    if len(train_loader.dataset) == 0:
        print("ERROR: No training samples found. Run 'py create_sample_data.py' first!")
        return

    model = LRCNActivityModel(num_classes=num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_f1": []
    }

    best_val_loss = float("inf")
    best_val_acc = 0.0

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        # Training Phase
        model.train()
        running_loss = 0.0
        train_preds, train_targets = [], []

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_x.size(0)
            preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            train_preds.extend(preds)
            train_targets.extend(batch_y.detach().cpu().numpy())

        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = accuracy_score(train_targets, train_preds) * 100.0

        # Validation Phase
        model.eval()
        val_running_loss = 0.0
        val_preds, val_targets = [], []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                val_running_loss += loss.item() * batch_x.size(0)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(batch_y.cpu().numpy())

        epoch_val_loss = val_running_loss / len(val_loader.dataset)
        epoch_val_acc = accuracy_score(val_targets, val_preds) * 100.0
        _, _, f1, _ = precision_recall_fscore_support(val_targets, val_preds, average='macro', zero_division=0)
        epoch_val_f1 = f1 * 100.0

        scheduler.step(epoch_val_loss)

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_acc"].append(epoch_val_acc)
        history["val_f1"].append(epoch_val_f1)

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] "
              f"| Train Loss: {epoch_train_loss:.4f} - Acc: {epoch_train_acc:.2f}% "
              f"| Val Loss: {epoch_val_loss:.4f} - Acc: {epoch_val_acc:.2f}% - F1: {epoch_val_f1:.2f}%")

        # Save Best Model Checkpoint
        if epoch_val_loss < best_val_loss or epoch_val_acc > best_val_acc:
            best_val_loss = min(best_val_loss, epoch_val_loss)
            best_val_acc = max(best_val_acc, epoch_val_acc)
            save_checkpoint(
                model=model,
                class_names=class_names,
                file_path=args.model_path,
                label_map_path=args.label_map_path
            )

    elapsed_time = time.time() - start_time
    print(f"\nTraining completed in {elapsed_time:.1f}s. Best Val Acc: {best_val_acc:.2f}%")

    # Generate and Save Training Curves
    os.makedirs(args.output_dir, exist_ok=True)
    history_plot_path = os.path.join(args.output_dir, "training_history.png")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs_range = range(1, args.epochs + 1)

    # Loss plot
    ax1.plot(epochs_range, history["train_loss"], 'o-', label="Training Loss", color="#1f77b4", linewidth=2)
    ax1.plot(epochs_range, history["val_loss"], 's--', label="Validation Loss", color="#ff7f0e", linewidth=2)
    ax1.set_title("Training & Validation Loss Curve", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Cross Entropy Loss", fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=11)

    # Accuracy plot
    ax2.plot(epochs_range, history["train_acc"], 'o-', label="Training Accuracy", color="#2ca02c", linewidth=2)
    ax2.plot(epochs_range, history["val_acc"], 's--', label="Validation Accuracy", color="#d62728", linewidth=2)
    ax2.set_title("Training & Validation Accuracy Curve", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Accuracy (%)", fontsize=12)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(history_plot_path, dpi=200)
    plt.close()
    print(f"Training history curves saved to {history_plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Human Activity Recognition LRCN Model")
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=6, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seq_len", type=int, default=DEFAULT_SEQUENCE_LENGTH, help="Frames per video clip")
    parser.add_argument("--img_size", type=int, default=DEFAULT_IMAGE_SIZE[0], help="Height/width of input frames")
    parser.add_argument("--dataset_root", type=str, default="dataset", help="Dataset directory")
    parser.add_argument("--model_path", type=str, default="models/activity_model.pt", help="Path to save best model")
    parser.add_argument("--label_map_path", type=str, default="models/label_map.json", help="Path to save label map")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory for plots")

    cli_args = parser.parse_args()
    train_model(cli_args)
