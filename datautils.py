import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from utils import filter_bandpass


class NPZDataset(Dataset):
    def __init__(self, npz_path, x_key="arr_0", y_key="arr_1", dtype=torch.float32):
        data = np.load(npz_path, mmap_mode='r')

        self.X = data[x_key]
        self.y = data[y_key]

        if self.y.ndim == 2 and self.y.shape[1] == 1:
            self.y = self.y.squeeze(1)  # Convert shape (N, 1) to (N,)
            print(f"\n!Warning: Labels had shape {data[y_key].shape}, squeezed to {self.y.shape}.")

        self.dtype = dtype

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx]).to(self.dtype)
        y = torch.tensor(self.y[idx])

        return x, y


class HDF5LazyDataset(Dataset):
    def __init__(self, path, task, transform=None):
        self.path = path
        self.task = task
        self.transform = transform if transform else self._basic_transform

        # DO NOT open file here (important for multiprocessing)
        self._file = None

    def _get_file(self):
        """Lazy file opener (one per worker)."""
        if self._file is None:
            self._file = h5py.File(self.path, "r")
        return self._file

    def __len__(self):
        f = self._get_file()
        return len(f["signals"])

    def _basic_transform(self, x, hash_file_name):

        # Replace NaN and inf with 0
        data = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)  
        if np.isnan(data).any():
            print(f"Warning: NaN values still present after nan_to_num, sample {hash_file_name}.")
        
        data = filter_bandpass(data, fs=500)       # Bandpass filter
        data =  (x - x.mean()) / (x.std() + 1e-8)  # Z-score normalization

        return data

    def __getitem__(self, idx):
        f = self._get_file()

        # ---- load signal ----
        x = f["signals"][idx]  # numpy array (lazy read)

        # ---- load label ----
        if f"labels/{self.task}" in f:
            y = f[f"labels/{self.task}"][idx]
            y = torch.tensor(y)
        else:
            y = None

        # ---- optional transform ----
        if self.transform:
            x = self.transform(x, hash_file_name=f["ecg_ids"][idx])
        x = torch.from_numpy(x).float()

        return x, y

    def __del__(self):
        if self._file is not None:
            self._file.close()


def create_dataloaders(
    train_path,
    val_path,
    test_path,
    task="main",
    batch_size=32,
    num_workers=4,
    pin_memory=True,
    seed=42,
    transform=None
):
    """
    Returns train, val, test DataLoaders
    """

    # ---- datasets ----
    train_dataset = HDF5LazyDataset(train_path, task, transform)
    print("Train dataset loaded with {} samples.".format(len(train_dataset)))
    
    val_dataset = HDF5LazyDataset(val_path, task, transform)
    print("Validation dataset loaded with {} samples.".format(len(val_dataset)))
    
    test_dataset = HDF5LazyDataset(test_path, task, transform)
    print("Test dataset loaded with {} samples.".format(len(test_dataset)))

    # ---- generator (for reproducibility) ----
    g = torch.Generator()
    g.manual_seed(seed)

    # ---- loaders ----
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=g,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader