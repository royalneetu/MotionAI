"""
Deep Learning Model Architecture: LRCN (Long-term Recurrent Convolutional Network)
Combines Time-Distributed CNNs for spatial feature extraction with Bidirectional LSTMs
for temporal motion analysis across video frame sequences.
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_CLASSES = [
    "Walking",
    "Running",
    "Sitting",
    "Standing",
    "Lying_Down",
    "Falling"
]

DEFAULT_IMAGE_SIZE = (64, 64)
DEFAULT_SEQUENCE_LENGTH = 16


class CNNFeatureExtractor(nn.Module):
    """
    Extracts spatial features from individual video frames.
    Input: (B, 3, H, W) -> Output: (B, feature_dim)
    """
    def __init__(self, feature_dim=256):
        super(CNNFeatureExtractor, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 2))
        )
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256 * 2 * 2, feature_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class LRCNActivityModel(nn.Module):
    """
    LRCN (Long-term Recurrent Convolutional Network)
    Accepts video sequences: (Batch, Sequence_Length, Channels, Height, Width)
    Extracts frame features via Time-Distributed CNN and analyzes temporal transitions via Bi-LSTM.
    """
    def __init__(self, num_classes=6, feature_dim=256, hidden_dim=128, num_lstm_layers=2, bidirectional=True):
        super(LRCNActivityModel, self).__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

        # Spatial CNN
        self.cnn = CNNFeatureExtractor(feature_dim=feature_dim)

        # Temporal Bi-LSTM
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=0.3 if num_lstm_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)

        # Dense Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        """
        x shape: (batch_size, seq_len, 3, H, W)
        """
        b, seq_len, c, h, w = x.shape

        # Fold seq_len into batch dimension to process frames in parallel through CNN
        x_reshaped = x.view(b * seq_len, c, h, w)
        features = self.cnn(x_reshaped)  # (b * seq_len, feature_dim)

        # Unfold back to (b, seq_len, feature_dim)
        features = features.view(b, seq_len, self.feature_dim)

        # Pass through temporal LSTM
        lstm_out, _ = self.lstm(features)  # (b, seq_len, lstm_out_dim)

        # Temporal Mean-Pooling to summarize whole sequence representation
        pooled = torch.mean(lstm_out, dim=1)  # (b, lstm_out_dim)

        # Output logits
        logits = self.classifier(pooled)
        return logits


def save_checkpoint(model, class_names, file_path="models/activity_model.pt", label_map_path="models/label_map.json"):
    """Saves model weights and label mapping dictionary."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    checkpoint = {
        "num_classes": len(class_names),
        "state_dict": model.state_dict(),
        "class_names": class_names
    }
    torch.save(checkpoint, file_path)

    label_map = {idx: cls_name for idx, cls_name in enumerate(class_names)}
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=4)
    print(f"Checkpoint saved to {file_path}")
    print(f"Label map saved to {label_map_path}")


def load_model(file_path="models/activity_model.pt", label_map_path="models/label_map.json", device="cpu"):
    """Loads trained model weights and class labels."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model checkpoint not found at {file_path}. Train the model first using train.py.")

    if os.path.exists(label_map_path):
        with open(label_map_path, "r", encoding="utf-8") as f:
            raw_map = json.load(f)
            label_map = {int(k): v for k, v in raw_map.items()}
            class_names = [label_map[i] for i in range(len(label_map))]
    else:
        class_names = DEFAULT_CLASSES

    model = LRCNActivityModel(num_classes=len(class_names))
    checkpoint = torch.load(file_path, map_location=device)
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model, class_names


if __name__ == "__main__":
    # Test model shape consistency
    test_model = LRCNActivityModel(num_classes=6)
    dummy_input = torch.randn(2, 16, 3, 64, 64)
    out = test_model(dummy_input)
    print(f"Test model instantiated successfully. Output shape: {out.shape} (Expected: [2, 6])")
