%% Sinusoidal signals with constant frequency
% CSV output:
%   row 1: header
%   row 2: true frequency [Hz]
%   column 1: time [s]
%   column 2~: signals w/ noise
% train_signals_const.csv
% val_signals_const.csv
% test_signals_const.csv

clear; clc; close all;

%% INPUT SETTINGS
num_signal = 100000;             % number of signals
N = 2000;                        % samples per signal (= NN input dimension)
dt = 1e-6;                       % sampling interval [s]

freq_list = [1000, 10000];       % constant frequency range [Hz]
amp = 1;                         % amplitude, normalized
phase = 0;                       % initial phase [rad]
snr_list = [10, 40];             % SNR range [dB] ([25, 25], [10, 40])

t = (0:N-1) * dt;

train_ratio = 0.70;
val_ratio = 0.15;
test_ratio = 0.15;

rng(100, 'twister');

train_file = 'train_signals_const.csv';
val_file   = 'val_signals_const.csv';
test_file  = 'test_signals_const.csv';

% Nyquist
assert(freq_list(2) < 1/(2*dt), 'Error: Nyquist');

% Split dataset
num_train = floor(num_signal * train_ratio);
num_val = floor(num_signal * val_ratio);
num_test = num_signal - num_train - num_val;

%% GENERATE CONSTANT-FREQUENCY SIGNALS
freq_Hz = freq_list(1) + diff(freq_list)*rand(num_signal,1);
snr_dB = snr_list(1) + diff(snr_list)*rand(num_signal,1);

signals = zeros(num_signal, N, 'single');

fprintf('Generating %d signals...\n', num_signal);

for i = 1:num_signal

    clean = amp * sin(2*pi*freq_Hz(i)*t + phase);

    sigma = amp / sqrt(10^(snr_dB(i)/10));

    noisy = clean + sigma*randn(1,N);

    signals(i,:) = single(noisy);

    if mod(i, 10000) == 0
        fprintf('Generated %d / %d signals\n', i, num_signal);
    end

end

fprintf('Signal generation completed.\n');

%% SPLIT DATASET
order = randperm(num_signal);

train_idx = order(1:num_train);
val_idx = order(num_train+1:num_train+num_val);
test_idx = order(num_train+num_val+1:end);

%% SAVE CSV FILES
fprintf('Saving training dataset...\n');
writeSignalDataset(train_file, t, freq_Hz(train_idx), signals(train_idx,:));

fprintf('Saving validation dataset...\n');
writeSignalDataset(val_file, t, freq_Hz(val_idx), signals(val_idx,:));

fprintf('Saving test dataset...\n');
writeSignalDataset(test_file, t, freq_Hz(test_idx), signals(test_idx,:));

fprintf('Dataset generation completed.\n');
fprintf('Total: %d signals, %d samples/signal, dt = %.9g s\n', num_signal, N, dt);
fprintf('Training:   %5d -> %s\n', num_train, train_file);
fprintf('Validation: %5d -> %s\n', num_val, val_file);
fprintf('Test:       %5d -> %s\n', num_test, test_file);

%% FUNCTION
function writeSignalDataset(file_name, time_sec, frequencies_Hz, signal_rows)
 
    num_cols = numel(frequencies_Hz);
    N = numel(time_sec);

    % Write CSV file
    fid = fopen(file_name, 'w');

    if fid == -1
        error('Cannot open file: %s', file_name);
    end

    % Header
    fprintf(fid, 'Time');

    for i = 1:num_cols
        fprintf(fid, ',Signal%d', i);
    end

    fprintf(fid, '\n');

    % Write data
    fprintf(fid, 'Frequency_Hz');
    fprintf(fid, ',%.9g', frequencies_Hz(:));
    fprintf(fid, '\n');

    for i = 1:N

        % Time [s]
        fprintf(fid, '%.9g', time_sec(i));

        % Signals
        fprintf(fid, ',%.9g', signal_rows(:,i));

        fprintf(fid, '\n');

    end

    fclose(fid);

    fprintf('Saved: %s (%d signals)\n', file_name, num_cols);

end
