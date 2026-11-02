# Test Three Algorithms: Time-Varying Frequency Signals
# Models:
#   1. Original MLP
#   2. Modified MLP
#   3. 1D CNN

import os
import random
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

test_file = 'test_signals_varying.csv'
# test_file = 'test_real_dataset.csv'

N = 2000
num_outputs = 20
freq_scale = 10000.0

batch_size = 128
seed = 100

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if not torch.backends.mps.is_available():
    raise RuntimeError('MPS is not available.')
device = torch.device('mps')
print('Device:', device)

model_paths = {
    'MLP':
            'results_varying_mlp/'
            'best_frequency_varying_mlp.pth',
    'mMLP':
        'results_varying_modified_mlp/'
        'best_frequency_varying_modified_mlp.pth',
    'CNN':
        'results_varying_cnn/'
        'best_frequency_varying_cnn.pth'
}

save_root = 'test_results_varying'
os.makedirs(save_root, exist_ok=True)

# Test Dataset
def load_data(file):
    data = pd.read_csv(file, low_memory=False)

    output_numbers = data.iloc[0, 1:].to_numpy(
        dtype=np.int32
    )

    assert np.all(output_numbers == num_outputs), (
        f'{file}: Output_Points mismatch'
    )

    # True frequency
    y = data.iloc[
        1:1+num_outputs, 1:
    ].to_numpy(dtype=np.float32).T.copy()

    # Input signal
    X = data.iloc[
        1+num_outputs:, 1:
    ].to_numpy(dtype=np.float32).T.copy()

    assert X.shape[1] == N
    assert y.shape[1] == num_outputs

    return (torch.from_numpy(X), torch.from_numpy(y / freq_scale))

X_test, y_test = load_data(test_file)

test_loader = DataLoader(
    TensorDataset(X_test, y_test),
    batch_size=batch_size,
    shuffle=False
)

time_data = pd.read_csv(
    test_file,
    usecols=[0],
    low_memory=False
)

label_time = pd.to_numeric(
    time_data.iloc[1:1+num_outputs, 0],
    errors='raise'
).to_numpy(dtype=np.float64)


# Model Architectures

# Original MLP: 2000-2-2-3-20
class OriginalMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N, 2),
            nn.ReLU(),

            nn.Linear(2, 2),
            nn.ReLU(),

            nn.Linear(2, 3),
            nn.ReLU(),

            nn.Linear(3, num_outputs)
        )

    def forward(self, x):
        return self.net(x)

# Modified MLP: 2000-128-64-32-20
class ModifiedMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, num_outputs)
        )

    def forward(self, x):
        return self.net(x)

# 1D CNN: Conv 8-24-8
class VaryingFrequencyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(

            nn.Conv1d(
                1, 8,
                kernel_size=7,
                padding=3
            ),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(
                8, 24,
                kernel_size=7,
                padding=3
            ),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(
                24, 8,
                kernel_size=7,
                padding=3
            ),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * 250, 64),
            nn.ReLU(),
            nn.Linear(64, num_outputs)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.features(x)
        return self.regressor(x)

model_classes = {
    'MLP': OriginalMLP,
    'mMLP': ModifiedMLP,
    'CNN': VaryingFrequencyCNN
}

# Test Evaluation
def evaluate_model(model_name, model_path, model_class):

    print('\n========================================')
    print('Evaluating:', model_name)
    print('========================================')

    if not os.path.isfile(model_path):
        print('Model file not found:', model_path)
        print('Skipping this model.')
        return None

    model = model_class().to(device)

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device,
            weights_only=True
        )
    )

    model.eval()
    pred_batches = []

    # Test Inference Time
    torch.mps.synchronize()
    start_time = time.perf_counter()

    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)

            pred = model(X_batch)

            pred_batches.append(
                pred.cpu().numpy()
            )

    torch.mps.synchronize()
    test_time = time.perf_counter() - start_time

    print(f'Total Test Time: {test_time:.4f} s')

    pred_test = np.concatenate(
        pred_batches, axis=0
    ) * freq_scale

    true_test = y_test.numpy() * freq_scale

    errors = pred_test - true_test
    abs_errors = np.abs(errors)

    mse_hz2 = np.mean(errors ** 2)
    mae_hz = np.mean(abs_errors)
    rmse_hz = np.sqrt(mse_hz2)

    mse_normalized = mse_hz2 / (freq_scale ** 2)

    mae_per_output = np.mean(
        abs_errors, axis=0
    )

    rmse_per_output = np.sqrt(
        np.mean(errors ** 2, axis=0)
    )

    print(f'Test MSE (normalized): {mse_normalized:.8f}')
    print(f'Test MAE: {mae_hz:.3f} Hz')
    print(f'Test RMSE: {rmse_hz:.3f} Hz')

    print('\nPrediction std per output:')
    print(np.std(pred_test, axis=0))

    print('\nTrue frequency std per output:')
    print(np.std(true_test, axis=0))

    print('\nFirst 3 predictions:')
    print(pred_test[:3, :])

    print('\nFirst 3 true frequencies:')
    print(true_test[:3, :])

    # Save Results
    model_save_folder = os.path.join(
        save_root,
        model_name
    )

    os.makedirs(
        model_save_folder,
        exist_ok=True
    )

    prediction_data = {}

    for j in range(num_outputs):

        prediction_data[
            f'true_freq_{j+1}_hz'
        ] = true_test[:, j]

        prediction_data[
            f'pred_freq_{j+1}_hz'
        ] = pred_test[:, j]

        prediction_data[
            f'abs_error_{j+1}_hz'
        ] = abs_errors[:, j]

    pd.DataFrame(
        prediction_data
    ).to_csv(
        os.path.join(
            model_save_folder,
            'test_predictions_varying.csv'
        ),
        index=False
    )

    pd.DataFrame({

        'output_point': np.arange(
            1, num_outputs + 1
        ),

        'time_ms': label_time * 1000,

        'mae_hz': mae_per_output,

        'rmse_hz': rmse_per_output

    }).to_csv(
        os.path.join(
            model_save_folder,
            'test_metrics_per_output.csv'
        ),
        index=False
    )

    fig, axes = plt.subplots(
        3, 1,
        figsize=(9, 10)
    )

    for i in range(min(3, len(pred_test))):

        axes[i].plot(
            label_time * 1000,
            true_test[i],
            'o-',
            label='True Frequency'
        )

        axes[i].plot(
            label_time * 1000,
            pred_test[i],
            's--',
            label=model_name
        )

        axes[i].set_xlabel('Time (ms)')
        axes[i].set_ylabel('Frequency (Hz)')

        axes[i].set_title(
            f'Test Signal {i+1}'
        )

        axes[i].legend()
        axes[i].grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            model_save_folder,
            'test_frequency_examples.png'
        ),
        dpi=200
    )

    plt.close()

    return {
        'model': model_name,
        'test_mse_normalized': mse_normalized,
        'test_mae_hz': mae_hz,
        'test_rmse_hz': rmse_hz,
        'total_test_time_s': test_time
    }


all_results = []

for model_name, model_path in model_paths.items():

    result = evaluate_model(
        model_name,
        model_path,
        model_classes[model_name]
    )

    if result is not None:
        all_results.append(result)


if all_results:

    comparison_df = pd.DataFrame(all_results)

    comparison_df.to_csv(
        os.path.join(
            save_root,
            'test_model_comparison.csv'
        ),
        index=False
    )

    print('\n========================================')
    print('FINAL TEST COMPARISON')
    print('========================================')

    print(
        comparison_df.to_string(
            index=False
        )
    )

    print(
        '\nSaved comparison:',
        os.path.join(
            save_root,
            'test_model_comparison.csv'
        )
    )

else:
    print('\nNo model files were found.')
