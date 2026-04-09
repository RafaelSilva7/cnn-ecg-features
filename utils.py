import torch
import random
import numpy as np
from datetime import datetime
import csv
from pathlib import Path


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
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

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