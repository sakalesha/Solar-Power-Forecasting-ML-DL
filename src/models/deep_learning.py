"""
deep_learning.py — PyTorch deep learning models for solar power forecasting.

Models implemented:
  1. LSTMForecaster    — stacked LSTM with dropout
  2. CNNForecaster     — 1-D convolutional network for sequence forecasting
  3. CNNLSTMForecaster — hybrid: CNN extracts features, LSTM learns temporal deps

All models expect input tensors of shape (batch, lookback, n_features)
and output shape (batch, 1) — single-step ahead prediction.

Also provides:
  • SolarDataset        — PyTorch Dataset wrapping numpy arrays
  • Trainer             — training loop with early stopping & loss logging
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm

from src.config import (
    LSTM_PARAMS, CNN_PARAMS, CNN_LSTM_PARAMS,
    BATCH_SIZE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    PATIENCE, DEVICE, MODELS_DIR, LOOKBACK,
)

# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────

class SolarDataset(Dataset):
    """
    PyTorch Dataset wrapper for pre-built sequence arrays.

    Parameters
    ----------
    X_seq : np.ndarray  shape (n, lookback, n_features) — float32
    y_seq : np.ndarray  shape (n,)                      — float32
    """

    def __init__(self, X_seq: np.ndarray, y_seq: np.ndarray):
        self.X = torch.tensor(X_seq, dtype=torch.float32)
        self.y = torch.tensor(y_seq, dtype=torch.float32).unsqueeze(-1)  # (n, 1)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def make_dataloader(X_seq, y_seq, batch_size: int = BATCH_SIZE,
                    shuffle: bool = True) -> DataLoader:
    dataset = SolarDataset(X_seq, y_seq)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=False)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 1: LSTM
# ─────────────────────────────────────────────────────────────────────────────

class LSTMForecaster(nn.Module):
    """
    Stacked LSTM for time-series regression.

    Architecture:
      [Input (lookback, n_features)]
        → LSTM (num_layers, hidden_size, dropout)
        → Last hidden state
        → Dropout
        → FC Linear → scalar output
    """

    def __init__(
        self,
        n_features:  int,
        hidden_size: int = LSTM_PARAMS["hidden_size"],
        num_layers:  int = LSTM_PARAMS["num_layers"],
        dropout:     float = LSTM_PARAMS["dropout"],
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = n_features,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            dropout     = dropout if num_layers > 1 else 0.0,
            batch_first = True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, lookback, n_features)
        out, _  = self.lstm(x)          # (batch, lookback, hidden)
        last    = out[:, -1, :]         # take last time step
        out     = self.dropout(last)
        return self.fc(out)             # (batch, 1)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 2: 1D-CNN
# ─────────────────────────────────────────────────────────────────────────────

class CNNForecaster(nn.Module):
    """
    1-D Convolutional Neural Network for sequence regression.

    Architecture:
      [Input (batch, lookback, n_features)]
        → Transpose to (batch, n_features, lookback)   [Conv1d expects channel-first]
        → Conv1d → BatchNorm → ReLU → MaxPool  (repeated per layer)
        → Flatten
        → Dropout → FC → scalar output
    """

    def __init__(
        self,
        n_features:  int,
        lookback:    int  = LOOKBACK,
        num_filters: list = CNN_PARAMS["num_filters"],
        kernel_sizes:list = CNN_PARAMS["kernel_sizes"],
        dropout:     float= CNN_PARAMS["dropout"],
    ):
        super().__init__()

        layers = []
        in_ch  = n_features
        seq_len = lookback

        for out_ch, ks in zip(num_filters, kernel_sizes):
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=ks, padding=ks // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.MaxPool1d(2),
            ]
            in_ch   = out_ch
            seq_len = seq_len // 2

        self.conv_block = nn.Sequential(*layers)
        self.flatten    = nn.Flatten()
        self.dropout    = nn.Dropout(dropout)
        flat_size       = in_ch * max(seq_len, 1)
        self.fc         = nn.Linear(flat_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, lookback, n_features)
        x = x.permute(0, 2, 1)         # → (batch, n_features, lookback)
        x = self.conv_block(x)
        x = self.flatten(x)
        x = self.dropout(x)
        return self.fc(x)               # (batch, 1)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 3: CNN-LSTM (Hybrid)
# ─────────────────────────────────────────────────────────────────────────────

class CNNLSTMForecaster(nn.Module):
    """
    Hybrid CNN-LSTM model.

    Architecture:
      [Input (batch, lookback, n_features)]
        → Permute → Conv1d → ReLU → MaxPool → Permute back
        → LSTM (learns temporal dependencies on CNN features)
        → Last hidden state → Dropout → FC → scalar output

    The CNN acts as a local feature extractor, reducing dimensionality
    before the LSTM processes the temporal structure.
    """

    def __init__(
        self,
        n_features:  int,
        lookback:    int   = LOOKBACK,
        cnn_filters: int   = CNN_LSTM_PARAMS["cnn_filters"],
        cnn_kernel:  int   = CNN_LSTM_PARAMS["cnn_kernel"],
        lstm_hidden: int   = CNN_LSTM_PARAMS["lstm_hidden"],
        lstm_layers: int   = CNN_LSTM_PARAMS["lstm_layers"],
        dropout:     float = CNN_LSTM_PARAMS["dropout"],
    ):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(n_features, cnn_filters, kernel_size=cnn_kernel,
                      padding=cnn_kernel // 2),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(
            input_size  = cnn_filters,
            hidden_size = lstm_hidden,
            num_layers  = lstm_layers,
            dropout     = dropout if lstm_layers > 1 else 0.0,
            batch_first = True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(lstm_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, lookback, n_features)
        x = x.permute(0, 2, 1)          # → (batch, n_features, lookback)
        x = self.conv(x)                 # → (batch, cnn_filters, lookback//2)
        x = x.permute(0, 2, 1)          # → (batch, lookback//2, cnn_filters)
        out, _ = self.lstm(x)
        last   = out[:, -1, :]
        out    = self.dropout(last)
        return self.fc(out)              # (batch, 1)


# ─────────────────────────────────────────────────────────────────────────────
# TRAINER
# ─────────────────────────────────────────────────────────────────────────────

class Trainer:
    """
    Generic PyTorch training loop with:
      • MSE loss (regression)
      • Adam optimiser with weight decay
      • ReduceLROnPlateau scheduler
      • Early stopping
      • Loss history for plotting
    """

    def __init__(
        self,
        model:          nn.Module,
        lr:             float = LEARNING_RATE,
        weight_decay:   float = WEIGHT_DECAY,
        patience:       int   = PATIENCE,
        device:         str   = DEVICE,
    ):
        self.device  = torch.device(device)
        self.model   = model.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=patience // 2,
        )
        self.patience        = patience
        self.train_losses:   list[float] = []
        self.val_losses:     list[float] = []

    def _run_epoch(self, loader: DataLoader, train: bool) -> float:
        self.model.train(train)
        total_loss = 0.0

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                pred = self.model(X_batch)
                loss = self.criterion(pred, y_batch)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                total_loss += loss.item() * len(y_batch)

        return total_loss / len(loader.dataset)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        epochs:       int = EPOCHS,
        verbose:      bool = True,
    ) -> "Trainer":
        best_val  = float("inf")
        no_improve = 0

        for epoch in range(1, epochs + 1):
            tr_loss = self._run_epoch(train_loader, train=True)
            va_loss = self._run_epoch(val_loader,   train=False)

            self.train_losses.append(tr_loss)
            self.val_losses.append(va_loss)
            self.scheduler.step(va_loss)

            if verbose and (epoch % 5 == 0 or epoch == 1):
                print(f"  Epoch {epoch:>3}/{epochs} | "
                      f"Train: {tr_loss:.4f} | Val: {va_loss:.4f}")

            if va_loss < best_val:
                best_val   = va_loss
                no_improve = 0
                self._save_checkpoint()          # save best weights
            else:
                no_improve += 1

            if no_improve >= self.patience:
                print(f"⏹  Early stopping at epoch {epoch} "
                      f"(no improvement for {self.patience} epochs)")
                break

        self._load_checkpoint()   # restore best weights
        return self

    def predict(self, loader: DataLoader) -> np.ndarray:
        """Run inference and return numpy array of predictions."""
        self.model.eval()
        preds = []
        with torch.no_grad():
            for X_batch, _ in loader:
                X_batch = X_batch.to(self.device)
                preds.append(self.model(X_batch).cpu().numpy())
        return np.concatenate(preds, axis=0).flatten()

    def _checkpoint_path(self) -> Path:
        return MODELS_DIR / f"{self.model.__class__.__name__}_best.pt"

    def _save_checkpoint(self) -> None:
        path = self._checkpoint_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def _load_checkpoint(self) -> None:
        path = self._checkpoint_path()
        if path.exists():
            self.model.load_state_dict(torch.load(path, map_location=self.device))

    def save_model(self, path: Path | None = None) -> Path:
        if path is None:
            path = self._checkpoint_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"💾 Model saved → {path}")
        return path

    @staticmethod
    def load_model(model: nn.Module, path: Path, device: str = DEVICE) -> nn.Module:
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        return model


# ─────────────────────────────────────────────────────────────────────────────
# FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def get_dl_model(name: str, n_features: int, lookback: int = LOOKBACK) -> nn.Module:
    """
    Instantiate a deep-learning model by name.

    Parameters
    ----------
    name       : "lstm" | "cnn" | "cnn_lstm"
    n_features : number of input features
    lookback   : sequence window length

    Returns
    -------
    Instantiated (un-trained) PyTorch Module.
    """
    registry = {
        "lstm":     LSTMForecaster,
        "cnn":      CNNForecaster,
        "cnn_lstm": CNNLSTMForecaster,
    }
    key = name.lower().strip()
    if key not in registry:
        raise ValueError(f"Unknown DL model '{name}'. "
                         f"Choose from: {list(registry.keys())}")

    ModelClass = registry[key]
    if key == "lstm":
        return ModelClass(n_features=n_features)
    else:
        return ModelClass(n_features=n_features, lookback=lookback)
