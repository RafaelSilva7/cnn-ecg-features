import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


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



def get_dataloaders(train_path, val_path, batch_size=32, num_workers=4, pin_memory=True):
    train_dataset = NPZDataset(train_path)
    print("Train dataset loaded with {} samples.".format(len(train_dataset)))

    val_dataset = NPZDataset(val_path)
    print("Validation dataset loaded with {} samples.".format(len(train_dataset)))

    train_loader = DataLoader(
        train_dataset,
        batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader