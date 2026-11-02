%% Sinusoidal signals with time-varying frequency
% CSV output:
%   row 1: header
%   row 2~(# output+1): true frequency list [Hz]
%   column 1: time [s]
%   column 2~: signals w/ noise
% train_signals_varying.csv
% val_signals_varying.csv
% test_signals_varying.csv

clear; clc; close all;

%% INPUT SETTINGS
num_signal = 100000;
N = 2000;
dt = 1e-6;

num_outputs = 20; % Number of output frequency points (= Output layers)

freq_list = [1000, 10000/1.5];     % Initial frequency (Hz)
amp = 1;
phase = 0;

snr_list = [10, 40];               % SNR (dB)
k_list = [0.3, 0.9];               % Transition duration ratio
dfreq_list = [0.05, 0.5];          % Frequency change ratio

t = (0:N-1)' * dt;

train_ratio = 0.70;
val_ratio = 0.15;
test_ratio = 0.15;

rng(100, 'twister');

train_file = 'train_signals_varying.csv';
val_file   = 'val_signals_varying.csv';
test_file  = 'test_signals_varying.csv';

train_meta_varying = 'train_metadata.csv';
val_meta_varying = 'val_metadata.csv';
test_meta_varying = 'test_metadata.csv';

% Split dataset
num_train = floor(num_signal * train_ratio);
num_val = floor(num_signal * val_ratio);
num_test = num_signal - num_train - num_val;

%% INITIALIZATION
label_idx = round(((1:num_outputs) - 0.5) * N / num_outputs);
label_idx = max(1, min(N, label_idx));

signals = zeros(N, num_signal, 'single');

freq_labels = zeros(num_outputs, num_signal, 'single');

f_initial_all = zeros(num_signal, 1);
f_final_all   = zeros(num_signal, 1);
snr_all       = zeros(num_signal, 1);
k_all         = zeros(num_signal, 1);
dfreq_all     = zeros(num_signal, 1);
direction_all = zeros(num_signal, 1);

%% GENERATE TIME-VARYING SIGNALS
fprintf('Generating %d signals...\n', num_signal);

for i = 1:num_signal
    
    f1 = freq_list(1) + (freq_list(2) - freq_list(1)) * rand;
    snr_dB = snr_list(1) + (snr_list(2) - snr_list(1)) * rand;
    k = k_list(1) + (k_list(2) - k_list(1)) * rand;
    dfreq = dfreq_list(1) + (dfreq_list(2) - dfreq_list(1)) * rand;
    if rand < 0.5
        direction = 1;
    else
        direction = -1;
    end

    f2 = f1 * (1 + direction * dfreq);

    duration = round(N * k);

    start_idx = round((N - duration) / 2) + 1;
    end_idx = start_idx + duration;

    u = ((1:N)' - start_idx) / duration;
    u = max(0, min(1, u));

    f_t = f1 + (f2 - f1) * u;

    phase_t = phase + 2*pi*cumtrapz(t, f_t);

    clean = amp * sin(phase_t);

    sigma = amp / sqrt(10^(snr_dB/10));
    noisy = clean + sigma * randn(N, 1);

    signals(:, i) = single(noisy);

    freq_labels(:, i) = single(f_t(label_idx));

    % Metadata
    f_initial_all(i) = f1;
    f_final_all(i) = f2;
    snr_all(i) = snr_dB;
    k_all(i) = k;
    dfreq_all(i) = dfreq;
    direction_all(i) = direction;

    % Progress
    if mod(i, 10000) == 0
        fprintf('Generated %d / %d signals\n', i, num_signal);
    end

end

fprintf('Signal generation completed.\n');

%% SPLIT DATASET
idx = randperm(num_signal);

train_idx = idx(1:num_train);
val_idx = idx(num_train+1 : num_train+num_val);
test_idx = idx(num_train+num_val+1 : end);

%% SAVE CSV FILES
fprintf('Saving training dataset...\n');
save_dataset(train_file, t, label_idx, signals(:, train_idx), freq_labels(:, train_idx), num_outputs);

fprintf('Saving validation dataset...\n');
save_dataset(val_file, t, label_idx, signals(:, val_idx), freq_labels(:, val_idx), num_outputs);

fprintf('Saving test dataset...\n');
save_dataset(test_file, t, label_idx, signals(:, test_idx), freq_labels(:, test_idx), num_outputs);

fprintf('All CSV files saved.\n');

%% Save Metadata CSV
metadata = table( ...
    f_initial_all, ...
    f_final_all, ...
    snr_all, ...
    k_all, ...
    dfreq_all, ...
    direction_all, ...
    'VariableNames', { ...
    'f_initial', ...
    'f_final', ...
    'snr_dB', ...
    'k', ...
    'dfreq', ...
    'direction' ...
    });

writetable(metadata(train_idx, :), train_meta_varying);
writetable(metadata(val_idx, :), val_meta_varying);
writetable(metadata(test_idx, :), test_meta_varying);

%% Plot Example Signals
example_idx = [1, 2, 3];

figure('Position', [100, 100, 1200, 500]);

for j = 1:length(example_idx)

    i = example_idx(j);

    f1 = f_initial_all(i);
    f2 = f_final_all(i);

    duration = round(N * k_all(i));

    start_idx = round((N - duration) / 2) + 1;

    u = ((1:N)' - start_idx) / duration;
    u = max(0, min(1, u));

    f_t = f1 + (f2 - f1) * u;

    subplot(2, length(example_idx), j);

    plot(t*1e3, f_t, 'LineWidth', 2);
    hold on;
    plot(t(label_idx)*1e3, f_t(label_idx), 'o', 'MarkerSize', 5, 'LineWidth',1.5);

    xlabel('Time, ms');
    ylabel('Frequency, Hz');
    title(sprintf('Signal %d: k=%.2f, df=%.2f', i, k_all(i), dfreq_all(i)));
    grid on;

    subplot(2, length(example_idx), j+3);

    plot(t*1e3, signals(:,i), 'k');

    xlabel('Time, ms');
    ylabel('Amplitude');
    title(sprintf('Noisy Signal (SNR=%.1f dB)', snr_all(i)));
    grid on;

end

%% FUNCTIONS
function save_dataset(filename, t, label_idx, X, Y, num_outputs)

    num_signals = size(X, 2);
    N = size(X, 1);

    assert(length(t) == N);
    assert(length(label_idx) == num_outputs);
    assert(size(Y, 1) == num_outputs);
    assert(size(Y, 2) == num_signals);

    headers = ["Time", "Signal" + string(1:num_signals)];

    % First column: Time
    label_times = t(label_idx);

    row_labels = [
        "Output_Points";
        string(label_times(:));
        string(t(:))
    ];

    % Row 1: Output number for every signal
    output_row = num_outputs * ones(1, num_signals);

    % Next num_outputs rows: True frequencies
    % Remaining N rows: Noisy signals
    numeric_data = [output_row; double(Y); double(X)];

    % Write CSV file
    fid = fopen(filename, 'w');

    if fid == -1
        error('Cannot open file: %s', filename);
    end

    % Header
    fprintf(fid, '%s\n', strjoin(headers, ','));

    % Write data
    for r = 1:size(numeric_data, 1)

        % Time [s]
        fprintf(fid, '%s', row_labels(r));

        % Signals
        fprintf(fid, ',%.9g', numeric_data(r,:));

        fprintf(fid, '\n');

    end

    fclose(fid);

    fprintf('Saved: %s (%d signals)\n', filename, num_signals);

end
