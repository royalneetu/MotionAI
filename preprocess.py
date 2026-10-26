"""
Preprocessing Pipeline for Human Activity Recognition.
Handles video frame decoding, uniform temporal sampling, spatial resizing,
channel normalization, data augmentation, and PyTorch Dataset loading.
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

DEFAULT_IMG_SIZE = (64, 64)
DEFAULT_SEQ_LEN = 16


def extract_frames(video_path: str, seq_len: int = DEFAULT_SEQ_LEN, img_size: tuple = DEFAULT_IMG_SIZE, augment: bool = False) -> np.ndarray:
    """Sample a fixed-length sequence without loading the entire video into memory.

    Returns float32 RGB frames in CHW layout with values in [0, 1]. Invalid
    videos raise ValueError rather than silently producing a black-frame prediction.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file at {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0:
        cap.release()
        raise ValueError("Video contains no readable frames or does not report a frame count")
    if total_frames >= seq_len:
        indices = np.linspace(0, total_frames - 1, seq_len, dtype=int).tolist()
    else:
        indices = list(range(total_frames)) + [total_frames - 1] * (seq_len - total_frames)

    processed_frames = []
    flip_horiz = augment and (np.random.rand() > 0.5)
    brightness_factor = 1.0 + (np.random.uniform(-0.15, 0.15) if augment else 0.0)
    last_valid = None
    try:
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                if last_valid is None:
                    continue
                frame = last_valid.copy()
            else:
                last_valid = frame.copy()
            frame = cv2.resize(frame, img_size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if flip_horiz:
                frame = cv2.flip(frame, 1)
            frame = np.clip(frame.astype(np.float32) * brightness_factor, 0.0, 255.0) / 255.0
            processed_frames.append(np.transpose(frame, (2, 0, 1)))
    finally:
        cap.release()
    if not processed_frames:
        raise ValueError("Could not decode frames from this video")
    while len(processed_frames) < seq_len:
        processed_frames.append(processed_frames[-1].copy())
    return np.asarray(processed_frames[:seq_len], dtype=np.float32)


def preprocess_frame_buffer(frames_list, img_size: tuple = DEFAULT_IMG_SIZE, seq_len: int = DEFAULT_SEQ_LEN) -> torch.Tensor:
    """
    Takes a list of OpenCV BGR frames from live webcam or video stream,
    extracts/samples seq_len frames, normalizes, and packages as a batch tensor (1, seq_len, 3, H, W).
    """
    n = len(frames_list)
    if n == 0:
        return torch.zeros((1, seq_len, 3, img_size[0], img_size[1]), dtype=torch.float32)

    if n >= seq_len:
        indices = np.linspace(0, n - 1, seq_len, dtype=int)
    else:
        indices = list(range(n)) + [n - 1] * (seq_len - n)

    processed = []
    for idx in indices:
        f = frames_list[idx]
        f_resized = cv2.resize(f, img_size)
        f_rgb = cv2.cvtColor(f_resized, cv2.COLOR_BGR2RGB)
        f_norm = f_rgb.astype(np.float32) / 255.0
        f_trans = np.transpose(f_norm, (2, 0, 1))
        processed.append(f_trans)

    arr = np.array(processed, dtype=np.float32)  # (seq_len, 3, H, W)
    tensor = torch.from_numpy(arr).unsqueeze(0)  # (1, seq_len, 3, H, W)
    return tensor


class ActivityVideoDataset(Dataset):
    """
    PyTorch Dataset that loads video clips organized by class directory:
    root_dir/
      Walking/
        vid1.mp4, vid2.mp4...
      Running/
      Sitting/
      Standing/
      Lying_Down/
      Falling/
    """
    def __init__(self, root_dir: str, class_names: list, seq_len: int = DEFAULT_SEQ_LEN, img_size: tuple = DEFAULT_IMG_SIZE, augment: bool = False):
        self.root_dir = root_dir
        self.class_names = class_names
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        self.seq_len = seq_len
        self.img_size = img_size
        self.augment = augment

        self.samples = []
        valid_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

        if os.path.exists(root_dir):
            for cls_name in class_names:
                cls_folder = os.path.join(root_dir, cls_name)
                if os.path.isdir(cls_folder):
                    for file in os.listdir(cls_folder):
                        _, ext = os.path.splitext(file)
                        if ext.lower() in valid_exts:
                            video_path = os.path.join(cls_folder, file)
                            self.samples.append((video_path, self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        try:
            frames = extract_frames(video_path, seq_len=self.seq_len, img_size=self.img_size, augment=self.augment)
        except Exception:
            # Fallback zero tensor if video decoding encounters corrupt frame
            frames = np.zeros((self.seq_len, 3, self.img_size[0], self.img_size[1]), dtype=np.float32)

        return torch.tensor(frames, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


def get_data_loaders(dataset_root="dataset", class_names=None, batch_size=8, seq_len=DEFAULT_SEQ_LEN, img_size=DEFAULT_IMG_SIZE):
    """
    Convenience function returning DataLoader instances for train, validation, and test splits.
    """
    if class_names is None:
        from model_architecture import DEFAULT_CLASSES
        class_names = DEFAULT_CLASSES

    train_dir = os.path.join(dataset_root, "train")
    val_dir = os.path.join(dataset_root, "validation")
    test_dir = os.path.join(dataset_root, "test")

    train_ds = ActivityVideoDataset(train_dir, class_names, seq_len=seq_len, img_size=img_size, augment=True)
    val_ds = ActivityVideoDataset(val_dir, class_names, seq_len=seq_len, img_size=img_size, augment=False)
    test_ds = ActivityVideoDataset(test_dir, class_names, seq_len=seq_len, img_size=img_size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, class_names


if __name__ == "__main__":
    print("Preprocessing module ready.")
