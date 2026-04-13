import torch
import random
import numpy as np
from datetime import datetime
import csv
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score
from scipy.signal import medfilt, iirnotch, filtfilt, butter, resample


def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def set_seed(seed: int = 42, deterministic: bool = True):
    """
    Set seed for reproducibility.

    Args:
        seed (int): random seed
        deterministic (bool): enforce deterministic behavior (slower)
    """
    random.seed(seed)
    np.random.seed(seed)

    # ---- PyTorch ----
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)

    # ---- cuDNN ----
    torch.backends.cudnn.benchmark = False
    if deterministic:
        torch.backends.cudnn.deterministic = True

    print(f"Seed set to {seed} (deterministic={deterministic})")


class MetricsLogger:
    def __init__(self, save_dir):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": []
        }

    def log(self, epoch, train_loss, val_loss, **metrics):
        self.history["epoch"].append(epoch)
        self.history["train_loss"].append(train_loss)
        self.history["val_loss"].append(val_loss)

        for key, value in metrics.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)

    def save_csv(self):
        path = self.save_dir / "metrics.csv"

        keys = self.history.keys()

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(keys)

            for i in range(len(self.history["epoch"])):
                row = [self.history[k][i] for k in keys]
                writer.writerow(row)


def classification_metrics(preds, targets):
    
    preds = preds.cpu().numpy()
    targets = targets.cpu().numpy()

    return {
        "precision_macro": precision_score(targets, preds, average="macro"),
        "recall_macro": recall_score(targets, preds, average="macro"),
        "f1_macro": f1_score(targets, preds, average="macro"),

        "precision_weighted": precision_score(targets, preds, average="weighted"),
        "recall_weighted": recall_score(targets, preds, average="weighted"),
        "f1_weighted": f1_score(targets, preds, average="weighted"),
    }


def filter_bandpass(signal, fs):
    """
    Bandpass filter

    Args:
        signal: 2D numpy array of shape (channels, time)
        fs: sampling frequency

    Return:
    - filtered_signal: 2D numpy array of shape (channels, time)
    """
    # Remove power-line interference
    b, a = iirnotch(50, 30, fs)
    filtered_signal = np.zeros_like(signal)
    for c in range(signal.shape[0]):
        filtered_signal[c] = filtfilt(b, a, signal[c])

    # Simple bandpass filter
    b, a = butter(N=4, Wn=[0.67, 40], btype='bandpass', fs=fs)
    for c in range(signal.shape[0]):
        filtered_signal[c] = filtfilt(b, a, filtered_signal[c])

    # Remove baseline wander
    baseline = np.zeros_like(filtered_signal)
    for c in range(filtered_signal.shape[0]):
        kernel_size = int(0.4 * fs) + 1
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure kernel size is odd
        baseline[c] = medfilt(filtered_signal[c], kernel_size=kernel_size)
    filter_ecg = filtered_signal - baseline

    return filter_ecg