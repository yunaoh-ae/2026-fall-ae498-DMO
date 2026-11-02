# Train - Validation Original MLP Algorithm: Constant frequency MLP
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os

# Constant frequency signals
# Input File
train_file = 'train_signals_const.csv'
val_file = 'val_signals_const.csv'

# Output Folder
save_folder = 'results_const_mlp'
os.makedirs(save_folder, exist_ok=True)

# Output File
model_file = os.path.join(save_folder, 'best_frequency_const_mlp.pth')
history_file = os.path.join(save_folder, 'training_history_const_mlp.csv')
prediction_file = os.path.join(save_folder, 'validation_results_const_mlp.csv')
loss_plot_file = os.path.join(save_folder, 'training_validation_loss_const_mlp.png')


# Input Settings
N = 2000 # number of samples per signal
freq_scale = 10000.0 # max frequency range = 10,000 Hz (normalized 0-1)

# User Settings
batch_size = 128
lr = 0.001 # learning rate
max_epochs = 500
patience = 30 # for early stopping
seed = 100

torch.manual_seed(seed)
np.random.seed(seed)

# Load CSV files
def load_data(file):
    data = pd.read_csv(file)
    X = data.iloc[1:, 1:].to_numpy(dtype=np.float32).T.copy()
    y = data.iloc[0, 1:].to_numpy(dtype=np.float32).reshape(-1, 1).copy()

    assert X.shape[1] == N, f'{file}: expected {N} samples, got {X.shape[1]}'
    
    return torch.from_numpy(X), torch.from_numpy(y / freq_scale)

X_train, y_train = load_data(train_file)
X_val, y_val = load_data(val_file)

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=batch_size,
    shuffle=True
)
val_loader = DataLoader(
    TensorDataset(X_val, y_val),
    batch_size=batch_size,
    shuffle=False
)

# NN Structure
# N(= 2000) - 2 - 2 - 3 - 1 (Output: single frequency estimation)
class ConstFrequencyNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N, 2), nn.ReLU(),
            nn.Linear(2, 2), nn.ReLU(),
            nn.Linear(2, 3), nn.ReLU(),
            nn.Linear(3, 1)
        )

    def forward(self, x):
        return self.net(x)

model = ConstFrequencyNN()
criterion = nn.MSELoss() # loss function: MSE
optimizer = optim.NAdam(model.parameters(), lr=lr) # optimizer: Nesterov Adam

train_losses = []
val_losses = []
best_val_loss = float('inf')
best_epoch = 0
wait = 0

# Training - Validation Loop
for epoch in range(1, max_epochs + 1):
    model.train()
    train_sum = 0.0

    # Training - Batch
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad() # initialization
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        optimizer.step()

        train_sum += loss.item() * X_batch.size(0)

    train_loss = train_sum / len(train_loader.dataset)

    # Validation - Batch
    model.eval()
    val_sum = 0.0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            pred = model(X_batch)
            loss = criterion(pred, y_batch)

            val_sum += loss.item() * X_batch.size(0)

    val_loss = val_sum / len(val_loader.dataset)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    # Find Best Model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        wait = 0
        torch.save(model.state_dict(), model_file)
    else:
        wait += 1

    if epoch == 1 or epoch % 10 == 0 or wait == patience:
        print(
            f'Epoch {epoch:4d} | '
            f'Train MSE: {train_loss:.8f} | '
            f'Val MSE: {val_loss:.8f}'
        )

    if wait >= patience:
        print(f'Early stopping at epoch {epoch}')
        break

# Reload Best Model
model.load_state_dict(torch.load(model_file, map_location='cpu', weights_only=True))
model.eval()

with torch.no_grad():
    pred_val = model(X_val).cpu().numpy().reshape(-1)
    pred_val = pred_val * freq_scale

true_val = y_val.numpy().reshape(-1) * freq_scale

mae_hz = np.mean(np.abs(pred_val - true_val))
rmse_hz = np.sqrt(np.mean((pred_val - true_val) ** 2))

# Show Results
print(f'Best epoch: {best_epoch}')
print(f'Best validation MSE (normalized): {best_val_loss:.8f}')
print(f'Validation MAE: {mae_hz:.3f} Hz')
print(f'Validation RMSE: {rmse_hz:.3f} Hz')
print(f'Saved model: {model_file}')

# Save Results
pd.DataFrame({
    'epoch': np.arange(1, len(train_losses) + 1),
    'train_mse': train_losses,
    'val_mse': val_losses
}).to_csv(history_file, index=False)

pd.DataFrame({
    'true_frequency_hz': true_val,
    'predicted_frequency_hz': pred_val,
    'absolute_error_hz': np.abs(pred_val - true_val)
}).to_csv(prediction_file, index=False)

# Plot Results
plt.figure(figsize=(8, 5))
plt.semilogy(range(1, len(train_losses) + 1), train_losses, label='Training MSE')
plt.semilogy(range(1, len(val_losses) + 1), val_losses, label='Validation MSE')
plt.axvline(best_epoch, color='gray', linestyle='--', label=f'Best epoch = {best_epoch}')
plt.xlabel('Epoch')
plt.ylabel('MSE (normalized frequency)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(loss_plot_file, dpi=200)
plt.close()
