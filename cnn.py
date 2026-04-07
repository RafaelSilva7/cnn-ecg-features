import torch
import torch.nn as nn
import torch.nn.functional as F


# ---- Basic Conv Block ----
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_factor):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(pool_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


# ---- Feature Extractor ----
class FeatureExtractor(nn.Module):
    def __init__(self, kernels, out_channels, max_poolings):
        super().__init__()

        # ---- Temporal ----
        layers = []
        in_ch = 12

        for k, n, mp in zip(kernels, out_channels, max_poolings):
            layers.append(ConvBlock(in_ch, n, k, mp))
            in_ch = n

        self.temporal = nn.Sequential(*layers)

        # ---- Spatial ----
        self.spatial = nn.Sequential(
            nn.Conv1d(in_ch, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.flatten(x)
        return x


# ---- ECG Classifier  ----
class ECGClassifier(nn.Module):
    def __init__(self, n_classes: int=2, task="multiclass"):
        """
        task:
            'multiclass' → CrossEntropyLoss
            'multilabel' → BCEWithLogitsLoss
        """
        super().__init__()
        self.n_classes = n_classes

        self.feature_extractor = FeatureExtractor(
            
            kernels=[7, 5, 5, 5, 5, 3, 3, 3],
            out_channels=[16, 16, 32, 32, 64, 64, 64, 64], 
            max_poolings=[2, 4, 2, 4, 2, 2, 2, 2]
        )

        self.classifier = None  # lazy init

    def _init_head(self, x):
        in_features = x.shape[1]

        self.classifier = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(64, self.n_classes) 
        )

    def forward(self, x):
        self._feat = self.feature_extractor(x)

        if self.classifier is None:
            self._init_head(self._feat)

        logits = self.classifier(self._feat)

        return logits


# ---- ECG Regressor  ----
class ECGRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = FeatureExtractor(
            kernels=[7, 5, 5, 5, 5, 3, 3, 3],
            out_channels=[16, 16, 32, 32, 64, 64, 64, 64], 
            max_poolings=[2, 4, 2, 4, 2, 2, 2, 2]
        )

        self.regressor = None  # lazy init

    def _init_head(self, x):
        in_features = x.shape[1]

        self.regressor = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(64, 1)
        )

    def forward(self, x):
        self._feat = self.feature_extractor(x)

        if self.regressor is None:
            self._init_head(self._feat)

        return self.regressor(self._feat)