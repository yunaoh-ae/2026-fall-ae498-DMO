%% Real Bearing Dataset Preprocessing
clear; clc;

%% Settings
folder = fileparts(mfilename('fullpath'));
if isempty(folder), folder = pwd; end

fileNames = {
    'H-A-1.mat'
    'H-A-2.mat'
    'H-A-3.mat'
    'H-B-1.mat'
    'H-B-2.mat'
    'H-B-3.mat'
};

fsReal = 200000;
fsNN = 1000000;
M = 12;
N = 2000;
numOutputs = 20;

PPR = 1024;
encoderThreshold = 2.0;
rateWindowSec = 0.020;

%% Initialize
K = numel(fileNames);
t = (0:N-1)'/fsNN;
label_idx = round(((1:numOutputs)'-0.5)*N/numOutputs);
labelTime = t(label_idx);
freqScale = M*fsNN/fsReal;

X = zeros(N,K);
Y = zeros(numOutputs,K);

%% Process MAT files
for k = 1:K
    S = load(fullfile(folder,fileNames{k}), ...
             'Channel_1','Channel_2');

    vibration = double(S.Channel_1(:));
    encoder = double(S.Channel_2(:));

    L = numel(vibration);

    % Center extraction
    startIdx = floor((L-((N-1)*M+1))/2)+1;
    idx = startIdx+(0:N-1)'*M;

    X(:,k) = vibration(idx);

    % Encoder rising edges
    high = encoder > encoderThreshold;
    risingIdx = find(~high(1:end-1) & high(2:end))+1;
    risingTime = (risingIdx-1)/fsReal;

    % Frequency labels
    labelRealTime = (idx(label_idx)-1)/fsReal;

    for j = 1:numOutputs
        left = max(0,labelRealTime(j)-rateWindowSec/2);
        right = min((L-1)/fsReal, ...
                    labelRealTime(j)+rateWindowSec/2);

        if right > left
            pulseHz = sum(risingTime>=left & ...
                          risingTime<right)/(right-left);
            Y(j,k) = pulseHz/PPR*freqScale;
        else
            Y(j,k) = NaN;
        end
    end

    fprintf('Processed: %s\n',fileNames{k});
end

%% Save CSV
outputCSV = fullfile(folder,'test_real_dataset.csv');
fid = fopen(outputCSV,'w');

% Header
fprintf(fid,'Time');
for k = 1:K
    fprintf(fid,',Signal%d',k);
end

% Output points
fprintf(fid,'\nOutput_Points');
fprintf(fid,',%d',repmat(numOutputs,1,K));
fprintf(fid,'\n');

% Frequency labels
for j = 1:numOutputs
    fprintf(fid,'%.12g',labelTime(j));
    fprintf(fid,',%.12g',Y(j,:));
    fprintf(fid,'\n');
end

% Vibration signals
for i = 1:N
    fprintf(fid,'%.12g',t(i));
    fprintf(fid,',%.12g',X(i,:));
    fprintf(fid,'\n');
end

fclose(fid);
