# Train - Validation Original MLP Algorithm: Time-Varying Frequency MLP
# Original MLP: 2000 - 2 - 2 - 3 - 20

import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Time-varying frequency signals
# Input File
train_file = 'train_signals_varying.csv'
val_file = 'val_signals_varying.csv'

# Output Folder
save_folder = 'results_varying_mlp'
os.makedirs(save_folder, exist_ok=True)

# Output File
model_file = os.path.join(save_folder, 'best_frequency_varying_mlp.pth')
history_file = os.path.join(save_folder, 'training_history_varying_mlp.csv')
prediction_file = os.path.join(save_folder, 'validation_results_varying_mlp.csv')
loss_plot_file = os.path.join(save_folder, 'training_validation_loss_varying_mlp.png')
example_plot_file = os.path.join(save_folder, 'frequency_predictions_examples_varying_mlp.png')

# Input Settings
N = 2000
num_outputs = 20
freq_scale = 10000.0

# User Settings
batch_size = 128
lr = 0.001
max_epochs = 500
patience = 30
seed = 100

torch.manual_seed(seed)
np.random.seed(seed)

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print('Device:', device)

# Load CSV Files
def load_data(file):
    data = pd.read_csv(file)

    # First data row: Number of output points
    output_numbers = data.iloc[0, 1:].to_numpy(dtype=np.int32)

    assert np.all(output_numbers == num_outputs), (
        f'{file}: Output_Points does not match num_outputs={num_outputs}'
    )

    # Next num_outputs rows: True frequencies
    y = data.iloc[1:1+num_outputs, 1:].to_numpy(
        dtype=np.float32
    ).T.copy()

    # Remaining N rows: Input signals
    X = data.iloc[1+num_outputs:, 1:].to_numpy(
        dtype=np.float32
    ).T.copy()

    assert X.shape[1] == N, (
        f'{file}: expected {N} samples, got {X.shape[1]}'
    )
    assert y.shape[1] == num_outputs, (
        f'{file}: expected {num_outputs} outputs, got {y.shape[1]}'
    )

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
# N(=2000) - 2 - 2 - 3 - 20 (Output: varying frequency profile estimation)
class VaryingFrequencyNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N, 2), nn.ReLU(),
            nn.Linear(2, 2), nn.ReLU(),
            nn.Linear(2, 3), nn.ReLU(),
            nn.Linear(3, num_outputs)
        )

    def forward(self, x):
        return self.net(x)

model = VaryingFrequencyNN().to(device)
criterion = nn.MSELoss() # loss function: MSE
optimizer = optim.NAdam(model.parameters(), lr=lr) # optimizer: Nesterov Adam

train_losses = []
val_losses = []
best_val_loss = float('inf')
best_epoch = 0
wait = 0

# Training - Validation Loop
start_time = time.perf_counter()

for epoch in range(1, max_epochs + 1):
    model.train()
    train_sum = 0.0

    # Training - Batch
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
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
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

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

# Total Training - Validataion Time
total_time = time.perf_counter() - start_time
average_epoch_time = total_time / len(train_losses)

print(f'Total Training - Validation Time: {total_time:.2f} s')
print(f'Average Time per Epoch: {average_epoch_time:.4f} s')

# Reload Best Model
model.load_state_dict(torch.load(model_file, map_location=device, weights_only=True))
model.eval()

with torch.no_grad():
    pred_val = model(X_val.to(device)).cpu().numpy() * freq_scale

true_val = y_val.numpy() * freq_scale

absolute_errors = np.abs(pred_val - true_val)
mae_hz = np.mean(absolute_errors)
rmse_hz = np.sqrt(np.mean((pred_val - true_val) ** 2))

mae_per_output = np.mean(absolute_errors, axis=0)
rmse_per_output = np.sqrt(np.mean((pred_val - true_val) ** 2, axis=0))

# Show Results
print('\n========================================')
print('MLP Validation Results')
print('========================================')

print(f'Best epoch: {best_epoch}')
print(f'Best validation MSE (normalized): {best_val_loss:.8f}')
print(f'Validation MAE: {mae_hz:.3f} Hz')
print(f'Validation RMSE: {rmse_hz:.3f} Hz')
print(f'Saved model: {model_file}')

# Save Training History
pd.DataFrame({
    'epoch': np.arange(1, len(train_losses) + 1),
    'train_mse': train_losses,
    'val_mse': val_losses
}).to_csv(history_file, index=False)

# Save Validation Predictions
prediction_data = {}
for j in range(num_outputs):
    prediction_data[f'true_freq_{j+1}_hz'] = true_val[:, j]
    prediction_data[f'pred_freq_{j+1}_hz'] = pred_val[:, j]
    prediction_data[f'abs_error_{j+1}_hz'] = absolute_errors[:, j]

pd.DataFrame(prediction_data).to_csv(
    prediction_file, index=False
)

# Training - Validation Loss Plot
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

# Prediction Standard Deviation
print('\nPrediction std per output:')
print(np.std(pred_val, axis=0))

print('\nTrue frequency std per output:')
print(np.std(true_val, axis=0))

print('\nFirst 3 predictions:')
print(pred_val[:3, :])

print('\nFirst 3 true frequencies:')
print(true_val[:3, :])

# Plot Example Frequency Predictions
time_data = pd.read_csv(
    val_file,
    usecols=[0],
    low_memory=False
)

label_time = pd.to_numeric(
    time_data.iloc[1:1+num_outputs, 0],
    errors='raise'
).to_numpy(dtype=np.float64)

fig, axes = plt.subplots(3, 1, figsize=(9, 10))

for i in range(3):
    axes[i].plot(
        label_time * 1000,
        true_val[i],
        'o-',
        label='True Frequency'
    )
    axes[i].plot(
        label_time * 1000,
        pred_val[i],
        's--',
        label='MLP Prediction'
    )
    axes[i].set_xlabel('Time (ms)')
    axes[i].set_ylabel('Frequency (Hz)')
    axes[i].set_title(f'Validation Signal {i+1}')
    axes[i].legend()
    axes[i].grid(True)

plt.tight_layout()
plt.savefig(example_plot_file, dpi=200)
plt.close()

print(f'\nSaved example plot: {example_plot_file}')
