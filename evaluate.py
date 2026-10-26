"""
Evaluation Module for Human Activity Recognition.
Calculates test accuracy, per-class Precision/Recall/F1-score,
generates annotated Confusion Matrix heatmap, and exports detailed test predictions.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from model_architecture import load_model, DEFAULT_IMAGE_SIZE, DEFAULT_SEQUENCE_LENGTH
from preprocess import get_data_loaders


def evaluate_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n==========================================")
    print(f" Evaluating Human Activity Recognition Model")
    print(f" Using Device: {device}")
    print(f" Loading checkpoint: {args.model_path}")
    print(f"==========================================\n")

    if not os.path.exists(args.model_path):
        print(f"Error: Model not found at {args.model_path}. Please train the model first.")
        return

    model, class_names = load_model(args.model_path, args.label_map_path, device=device)
    model.eval()

    _, _, test_loader, _ = get_data_loaders(
        dataset_root=args.dataset_root,
        class_names=class_names,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        img_size=(args.img_size, args.img_size)
    )

    print(f"Loaded {len(test_loader.dataset)} test samples across {len(class_names)} classes.")

    all_preds = []
    all_targets = []
    all_confs = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            probs = torch.softmax(outputs, dim=1)
            confs, preds = torch.max(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch_y.numpy())
            all_confs.extend(confs.cpu().numpy())

    # Overall metrics
    overall_acc = accuracy_score(all_targets, all_preds) * 100.0
    print(f"\n================ Overall Test Accuracy: {overall_acc:.2f}% ================\n")

    # Classification report
    report_text = classification_report(
        all_targets,
        all_preds,
        target_names=class_names,
        digits=4,
        zero_division=0
    )
    print("Classification Report:\n")
    print(report_text)

    os.makedirs(args.output_dir, exist_ok=True)
    report_path = os.path.join(args.output_dir, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Human Activity Recognition - Test Evaluation Report\n")
        f.write(f"Overall Accuracy: {overall_acc:.2f}%\n\n")
        f.write(report_text)
    print(f"Classification report saved to {report_path}")

    # Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds, labels=range(len(class_names)))
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-7)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_norm * 100,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Percentage (%)'},
        annot_kws={'size': 12, 'weight': 'bold'}
    )
    plt.title(f"Confusion Matrix (Normalized %) - Test Accuracy: {overall_acc:.1f}%", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Predicted Class", fontsize=12, fontweight="bold")
    plt.ylabel("True Class", fontsize=12, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"Confusion matrix heatmap saved to {cm_path}")

    # Detailed Predictions DataFrame
    results_list = []
    for idx, (vid_path, target) in enumerate(test_loader.dataset.samples):
        results_list.append({
            "video_path": os.path.basename(vid_path),
            "true_label": class_names[target],
            "predicted_label": class_names[all_preds[idx]],
            "confidence_pct": round(float(all_confs[idx] * 100), 2),
            "is_correct": bool(target == all_preds[idx]),
            "is_fall_detected": bool(class_names[all_preds[idx]] == "Falling")
        })

    results_df = pd.DataFrame(results_list)
    csv_path = os.path.join(args.output_dir, "test_predictions.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Detailed predictions saved to {csv_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Human Activity Recognition Model")
    parser.add_argument("--dataset_root", type=str, default="dataset", help="Dataset directory")
    parser.add_argument("--model_path", type=str, default="models/activity_model.pt", help="Path to saved model")
    parser.add_argument("--label_map_path", type=str, default="models/label_map.json", help="Path to label map")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=6, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=DEFAULT_SEQUENCE_LENGTH, help="Frames per sequence")
    parser.add_argument("--img_size", type=int, default=DEFAULT_IMAGE_SIZE[0], help="Image dimension")

    cli_args = parser.parse_args()
    evaluate_model(cli_args)
