%% =========================================================
%  spectralAnalysis_v2.m
%
%  Computes power spectral density (PSD) and time-frequency
%  representations from raw LFP electrode files.
% =========================================================

clearvars; close all; clc;

%% 0. PARAMETERS 

dataDir       = '.';
outDir        = 'spectral_out_v2';

nElec         = 90;
Fs            = 2000;              % Hz

% Epoch & baseline windows
epochStart    = -0.5;              % s
epochEnd      =  1.0;              % s
baselineStart = -0.5;
baselineEnd   = -0.05;
stimStart     =  0.0;              % stimulus-period PSD window

hardAmpLimit_uV = 2000;           % exclude electrode from CAR if any
                                   % trial exceeds this p-t-p (uV)

% Artifact rejection (trial-level)
ampThresh_uV  = 1200;             % uV peak-to-peak per trial
zscoreThresh  = 5;

% Welch PSD settings
welchWin_s    = 0.5;              % window length (s)
welchOverlap  = 0.5;              % 50 % overlap
nfft          = 4096;

% Spectrogram settings
tfWin_s       = 0.2;
tfOverlap     = 0.95;
tfNfft        = 4096;

% Display frequency range
freqRange     = [1 150];

% CHANGE 1: unified gamma band definitions
%    All figures and analyses use these boundaries.
slowGamma     = [20  35];         % Hz
fastGamma     = [40  65];         % Hz

% Additional standard bands for the summary table
bands.delta   = [1   4];
bands.theta   = [4   8];
bands.alpha   = [8  13];
bands.beta    = [13 30];
bands.slowG   = slowGamma;
bands.fastG   = fastGamma;
bandNames     = fieldnames(bands);

% CHANGE 7: edge-artefact trim 
edgeTrim_s    = 0.05;             % trim last 50 ms of envelope

%% 1. LOAD METADATA 

fprintf('Loading lfpInfo...\n');
info        = load(fullfile(dataDir, 'lfpInfo.mat'));
timeVals    = info.timeVals(1,:);
goodStimPos = info.goodStimPos(1,:);
nTrials     = length(goodStimPos);

epochIdx    = timeVals >= epochStart & timeVals <= epochEnd;
timeEpoch   = timeVals(epochIdx);
nTime       = sum(epochIdx);

baseIdx     = timeEpoch >= baselineStart & timeEpoch <= baselineEnd;
stimIdx     = timeEpoch >= stimStart     & timeEpoch <= epochEnd;

fprintf('  Epoch: %.2f to %.2f s | %d samples | Fs=%d Hz\n', ...
        epochStart, epochEnd, nTime, Fs);

%% 2. LOAD ALL ELECTRODES 

fprintf('\nLoading %d electrodes...\n', nElec);
lfpAll = zeros(nElec, nTrials, nTime, 'single');

for e = 1:nElec
    fname = fullfile(dataDir, sprintf('elec%d.mat', e));
    if ~isfile(fname); lfpAll(e,:,:) = NaN; continue; end
    tmp = load(fname, 'analogData');
    lfpAll(e,:,:) = single(tmp.analogData(:, epochIdx));
    if mod(e,20)==0; fprintf('  %d / %d\n', e, nElec); end
end

%% 3. PRE-CAR ELECTRODE SCREENING
%
%  Identify electrodes whose peak-to-peak amplitude exceeds
%  hardAmpLimit_uV in ANY trial.  These are excluded from the
%  reference mean so their artefacts are not broadcast to all
%  other channels.  They are still processed individually
%  (artefact trials will be rejected in step 4).
%
% -----------------------------------------------------------

ptp_elec = squeeze(max(lfpAll,[],3) - min(lfpAll,[],3));  % [nElec x nTrials]
badElecForCAR = any(ptp_elec > hardAmpLimit_uV, 2);        % [nElec x 1]
nBadCAR = sum(badElecForCAR);
fprintf('\nPre-CAR screening: %d / %d electrodes excluded from reference (p-t-p > %d uV)\n', ...
        nBadCAR, nElec, hardAmpLimit_uV);

goodElecIdx = ~badElecForCAR;
carMean     = squeeze(mean(lfpAll(goodElecIdx,:,:), 1, 'omitnan'));  % [nTrials x nTime]
lfpCAR      = lfpAll - reshape(carMean, [1, nTrials, nTime]);

%% 4. TRIAL-LEVEL ARTIFACT REJECTION

ptp      = squeeze(max(lfpCAR,[],3) - min(lfpCAR,[],3));   % [nElec x nTrials]
trialPow = squeeze(mean(double(lfpCAR).^2, 3));
zPow     = zscore(trialPow, 0, 2);

badTrials  = any(ptp > ampThresh_uV, 1) | any(abs(zPow) > zscoreThresh, 1);
goodTrials = ~badTrials;
nGood      = sum(goodTrials);

fprintf('Artifact rejection: %d bad / %d total (%.1f%% retained)\n', ...
        sum(badTrials), nTrials, 100*nGood/nTrials);
if nGood == 0
    error('All trials rejected — loosen ampThresh_uV (%d) or zscoreThresh (%d)', ...
          ampThresh_uV, zscoreThresh);
end

lfpGood = lfpCAR(:, goodTrials, :);    % [nElec x nGood x nTime]

%% 5. BASELINE Z-SCORE CORRECTION 

lfpBC = zeros(size(lfpGood), 'single');
for e = 1:nElec
    sig = double(squeeze(lfpGood(e,:,:)));
    mu  = mean(sig(:, baseIdx), 2);
    sd  = std( sig(:, baseIdx), 0, 2); sd(sd==0) = 1;
    lfpBC(e,:,:) = single((sig - mu) ./ sd);
end
fprintf('Baseline z-score correction done.\n');

%% 6. WELCH PSD — PER TRIAL THEN AVERAGE 
%
%  Run pwelch on each individual trial, then average.
%  Done for two windows:
%    (a) Full epoch  (-0.5 – 1.0 s)  — matches v1 for comparison
%    (b) Stimulus    ( 0.0 – 1.0 s)  — used for baseline-normalised PSD
%    (c) Baseline    (-0.5 – -0.05 s)— denominator for (b)
%
% -----------------------------------------------------------

fprintf('\nComputing per-trial Welch PSD...\n');

% Adaptive Welch windows per segment 
%
%  pwelch requires the window to be <= the signal length.
%  The three segments have different lengths:
%    full epoch : epochEnd - epochStart = 1.5 s  → 3000 samples
%    stimulus   : epochEnd - stimStart  = 1.0 s  → 2000 samples
%    baseline   : baselineEnd-baselineStart=0.45s →  900 samples
%
%  We cap each window at floor(segmentLength / 2) so at least
%  two non-overlapping segments exist (minimum for Welch).
%  The nfft is kept constant so all three PSDs share the same
%  frequency axis — shorter windows simply have lower resolution
%  but land on the same frequency bins after zero-padding.
%
% -----------------------------------------------------------

nFull = nTime;
nStim = sum(stimIdx);
nBase = sum(baseIdx);

% Window lengths: desired 0.5 s, capped at floor(segLen/2)
makeWin = @(nSeg) hann(min(round(welchWin_s * Fs), floor(nSeg/2)));

win_full = makeWin(nFull);
win_stim = makeWin(nStim);
win_base = makeWin(nBase);

nov_full = round(length(win_full) * welchOverlap);
nov_stim = round(length(win_stim) * welchOverlap);
nov_base = round(length(win_base) * welchOverlap);

fprintf('  Window lengths — full: %d, stim: %d, base: %d samples\n', ...
        length(win_full), length(win_stim), length(win_base));

% Frequency axis (common nfft → identical freq bins for all three)
[~, freqPSD] = pwelch(randn(nFull,1), win_full, nov_full, nfft, Fs);
nFreq = length(freqPSD);

pxx_full_dB  = zeros(nElec, nFreq);
pxx_stim_dB  = zeros(nElec, nFreq);
pxx_base_dB  = zeros(nElec, nFreq);

for e = 1:nElec
    sigMat = double(squeeze(lfpBC(e,:,:)));     % [nGood x nTime]

    pxx_full_tr  = zeros(nGood, nFreq);
    pxx_stim_tr  = zeros(nGood, nFreq);
    pxx_base_tr  = zeros(nGood, nFreq);

    for tr = 1:nGood
        [px_f, ~] = pwelch(sigMat(tr,:)',           win_full, nov_full, nfft, Fs);
        [px_s, ~] = pwelch(sigMat(tr, stimIdx)',    win_stim, nov_stim, nfft, Fs);
        [px_b, ~] = pwelch(sigMat(tr, baseIdx)',    win_base, nov_base, nfft, Fs);

        pxx_full_tr(tr,:) = 10*log10(px_f' + eps);
        pxx_stim_tr(tr,:) = 10*log10(px_s' + eps);
        pxx_base_tr(tr,:) = 10*log10(px_b' + eps);
    end

    pxx_full_dB(e,:) = mean(pxx_full_tr, 1);
    pxx_stim_dB(e,:) = mean(pxx_stim_tr, 1);
    pxx_base_dB(e,:) = mean(pxx_base_tr, 1);

    if mod(e,20)==0
        fprintf('  Electrode %d / %d done\n', e, nElec);
    end
end

% Full-epoch mean/SEM across electrodes
meanPSD_full = mean(pxx_full_dB, 1);
semPSD_full  = std(pxx_full_dB,  0, 1) / sqrt(nElec);

% Baseline-normalised stimulus PSD 
%    Express stimulus-period power in dB relative to baseline.
%    Positive values = power increase above baseline.
pxx_norm_dB  = pxx_stim_dB - pxx_base_dB;    % [nElec x nFreq]
meanPSD_norm = mean(pxx_norm_dB, 1);
semPSD_norm  = std(pxx_norm_dB,  0, 1) / sqrt(nElec);

fprintf('  Done. Freq resolution: %.3f Hz\n', freqPSD(2)-freqPSD(1));

%% 7. BAND POWER TABLE 

fprintf('\n--- Band Power (mean +/- SEM across electrodes, full epoch) ---\n');
fprintf('  %-22s  %10s  %10s\n', 'Band', 'Mean (dB)', 'SEM (dB)');
fprintf('  %s\n', repmat('-',1,46));

bandPower = struct();
for b = 1:length(bandNames)
    bn   = bandNames{b};
    fIdx = freqPSD >= bands.(bn)(1) & freqPSD <= bands.(bn)(2);
    bp   = mean(pxx_full_dB(:, fIdx), 2);
    bandPower.(bn) = bp;
    fprintf('  %-22s  %10.2f  %10.2f\n', ...
        sprintf('%s (%d-%d Hz)', bn, bands.(bn)(1), bands.(bn)(2)), ...
        mean(bp), std(bp)/sqrt(nElec));
end

% Also print baseline-normalised values for gamma bands
fprintf('\n--- Baseline-normalised band power (stimulus - baseline, dB) ---\n');
fprintf('  %-22s  %10s  %10s\n', 'Band', 'Mean (dB)', 'SEM (dB)');
fprintf('  %s\n', repmat('-',1,46));
for bn = {'slowG','fastG'}
    bn = bn{1};
    fIdx = freqPSD >= bands.(bn)(1) & freqPSD <= bands.(bn)(2);
    bp   = mean(pxx_norm_dB(:, fIdx), 2);
    fprintf('  %-22s  %10.2f  %10.2f\n', ...
        sprintf('%s (%d-%d Hz)', bn, bands.(bn)(1), bands.(bn)(2)), ...
        mean(bp), std(bp)/sqrt(nElec));
end

%% 8. SPECTROGRAM — AVERAGED ACROSS TRIALS & ELECTRODES 
%
%  CHANGE 4: Average over ALL electrodes, not just electrode 1.
%  abs(spectrogram)^2 is computed per trial per electrode;
%  powers are averaged first across trials, then across electrodes.
%
% -----------------------------------------------------------

fprintf('\nComputing trial- and electrode-averaged spectrogram...\n');

tfWinLen   = round(tfWin_s * Fs);
tfNoverlap = round(tfWinLen * tfOverlap);

% Get axis sizes from a dummy call
sig0 = double(squeeze(lfpBC(1,1,:)))';
[~, F_tf, T_tf] = spectrogram(sig0, hann(tfWinLen), tfNoverlap, tfNfft, Fs);
nF_tf = length(F_tf);
nT_tf = length(T_tf);

S_allElec = zeros(nF_tf, nT_tf);   % accumulator across electrodes

for e = 1:nElec
    S_elec = zeros(nF_tf, nT_tf);
    for tr = 1:nGood
        sig_tr = double(squeeze(lfpBC(e, tr, :)))';
        [S_tr, ~, ~] = spectrogram(sig_tr, hann(tfWinLen), tfNoverlap, tfNfft, Fs);
        S_elec = S_elec + abs(S_tr).^2;
    end
    S_allElec = S_allElec + S_elec / nGood;   % trial-averaged for this elec
    if mod(e,20)==0
        fprintf('  Electrode %d / %d done\n', e, nElec);
    end
end

S_avg  = S_allElec / nElec;          % average across electrodes
S_dB   = 10*log10(S_avg + eps);
T_axis = T_tf + timeEpoch(1);        % shift to stimulus-relative time

fprintf('  Done. Time bins: %d | Freq bins: %d\n', nT_tf, nF_tf);

%% 9. GAMMA POWER TIME COURSES 
%
%  (unified bands): both Hilbert envelopes use
%  slowGamma / fastGamma defined in section 0.
%
%  we compute two SEM measures —
%    semTC_trials  : uncertainty of the mean power over time
%                    (SEM across trials, for each electrode,
%                     then mean across electrodes)
%    semTC_elec    : variability across electrodes
%                    (SEM across electrodes of the trial mean)
%
%  (edge trim): last edgeTrim_s seconds zeroed out.
%
% -----------------------------------------------------------

fprintf('\nComputing gamma power time courses...\n');

gammaBands = {slowGamma, sprintf('Slow gamma (%d-%d Hz)', slowGamma(1), slowGamma(2)), [0.1 0.6 0.1];
              fastGamma, sprintf('Fast gamma (%d-%d Hz)', fastGamma(1), fastGamma(2)), [0.8 0.1 0.1]};
nGB = size(gammaBands, 1);

% gammaPow: [nGB x nElec x nGood x nTime]
gammaPow = zeros(nGB, nElec, nGood, nTime);

for g = 1:nGB
    [bCoeff, aCoeff] = butter(4, gammaBands{g,1}/(Fs/2), 'bandpass');
    for e = 1:nElec
        sig  = double(squeeze(lfpBC(e,:,:)));       % [nGood x nTime]
        sigF = filtfilt(bCoeff, aCoeff, sig')';     % [nGood x nTime]
        gammaPow(g,e,:,:) = sigF .^ 2;
    end
    fprintf('  %s done.\n', gammaBands{g,2});
end

% two SEM variants

% (A) Mean and SEM across ELECTRODES (primary plot metric)
%     trialMean_elec: [nGB x nElec x nTime]  — trial mean per electrode
trialMean_elec = squeeze(mean(gammaPow, 3));       % avg over trials
gammaMeanTC    = squeeze(mean(trialMean_elec, 2)); % [nGB x nTime] mean across elec
gammaSEM_elec  = squeeze(std(trialMean_elec, 0, 2)) / sqrt(nElec); % [nGB x nTime]

% (B) SEM across TRIALS (of the electrode-mean)
%     elecMean_trial: [nGB x nGood x nTime]  — electrode mean per trial
elecMean_trial  = squeeze(mean(gammaPow, 2));      % avg over electrodes
gammaSEM_trials = squeeze(std(elecMean_trial, 0, 2)) / sqrt(nGood); % [nGB x nTime]

% edge-artefact trim 
edgeSamples                     = round(edgeTrim_s * Fs);
gammaMeanTC(:, end-edgeSamples+1:end)    = NaN;
gammaSEM_elec(:, end-edgeSamples+1:end) = NaN;
gammaSEM_trials(:, end-edgeSamples+1:end)= NaN;

% percent-change from baseline
baseMean = mean(gammaMeanTC(:, baseIdx), 2, 'omitnan');  % [nGB x 1]
gammaPctChange = (gammaMeanTC - baseMean) ./ baseMean * 100;

%% 10. PLOTS

if ~exist(outDir,'dir'); mkdir(outDir); end

fMask    = freqPSD >= freqRange(1) & freqPSD <= freqRange(2);
f_plot   = freqPSD(fMask);
sMask    = freqPSD >= slowGamma(1) & freqPSD <= slowGamma(2);
fgMask   = freqPSD >= fastGamma(1) & freqPSD <= fastGamma(2);

% Fig 1a: Mean PSD +/- SEM (full epoch, raw dB)

fig1a = figure('Name','Mean PSD – full epoch','Color','w','Position',[50 50 950 500]);
patch([f_plot; flipud(f_plot)], ...
      [(meanPSD_full(fMask)-semPSD_full(fMask))'; ...
       flipud((meanPSD_full(fMask)+semPSD_full(fMask))')], ...
      [0.65 0.8 1],'EdgeColor','none','FaceAlpha',0.5);
hold on;
plot(f_plot, meanPSD_full(fMask), 'b', 'LineWidth', 1.8);

yL = ylim;
patch([slowGamma(1) slowGamma(2) slowGamma(2) slowGamma(1)], ...
      [yL(1) yL(1) yL(2) yL(2)], [0.1 0.65 0.1], ...
      'FaceAlpha',0.12,'EdgeColor','none');
text(mean(slowGamma), yL(1)+0.9*(yL(2)-yL(1)), 'Slow \gamma', ...
     'Color',[0.05 0.4 0.05],'FontSize',10,'HorizontalAlignment','center');

patch([fastGamma(1) fastGamma(2) fastGamma(2) fastGamma(1)], ...
      [yL(1) yL(1) yL(2) yL(2)], [0.8 0.1 0.1], ...
      'FaceAlpha',0.12,'EdgeColor','none');
text(mean(fastGamma), yL(1)+0.9*(yL(2)-yL(1)), 'Fast \gamma', ...
     'Color',[0.6 0.05 0.05],'FontSize',10,'HorizontalAlignment','center');

xlabel('Frequency (Hz)'); ylabel('Power (dB)');
title(sprintf('Mean PSD \\pm SEM — full epoch — %d electrodes, %d trials', nElec, nGood));
xlim(freqRange); grid on; box off;
saveas(fig1a, fullfile(outDir,'PSD_fullEpoch.png'));

% Fig 1b: Baseline-normalised stimulus PSD 

fig1b = figure('Name','Baseline-normalised PSD','Color','w','Position',[50 50 950 500]);
patch([f_plot; flipud(f_plot)], ...
      [(meanPSD_norm(fMask)-semPSD_norm(fMask))'; ...
       flipud((meanPSD_norm(fMask)+semPSD_norm(fMask))')], ...
      [0.65 0.8 1],'EdgeColor','none','FaceAlpha',0.5);
hold on;
plot(f_plot, meanPSD_norm(fMask), 'b', 'LineWidth', 1.8);
yline(0,'--k','Baseline','LabelVerticalAlignment','bottom','LineWidth',1);

yL = ylim;
patch([slowGamma(1) slowGamma(2) slowGamma(2) slowGamma(1)], ...
      [yL(1) yL(1) yL(2) yL(2)], [0.1 0.65 0.1], ...
      'FaceAlpha',0.12,'EdgeColor','none');
text(mean(slowGamma), yL(1)+0.9*(yL(2)-yL(1)), 'Slow \gamma', ...
     'Color',[0.05 0.4 0.05],'FontSize',10,'HorizontalAlignment','center');

patch([fastGamma(1) fastGamma(2) fastGamma(2) fastGamma(1)], ...
      [yL(1) yL(1) yL(2) yL(2)], [0.8 0.1 0.1], ...
      'FaceAlpha',0.12,'EdgeColor','none');
text(mean(fastGamma), yL(1)+0.9*(yL(2)-yL(1)), 'Fast \gamma', ...
     'Color',[0.6 0.05 0.05],'FontSize',10,'HorizontalAlignment','center');

xlabel('Frequency (Hz)');
ylabel('Power change re baseline (dB)');
title(sprintf('Baseline-normalised PSD (stimulus – baseline) — %d electrodes, %d trials', nElec, nGood));
xlim(freqRange); grid on; box off;
saveas(fig1b, fullfile(outDir,'PSD_baselineNorm.png'));

% Fig 2: PSD heatmap (all electrodes, full epoch)

fig2 = figure('Name','PSD Heatmap','Color','w','Position',[50 50 1000 500]);
imagesc(f_plot, 1:nElec, pxx_full_dB(:,fMask));
axis xy; colormap(jet);
cb = colorbar; cb.Label.String = 'Power (dB)';
xline(slowGamma(1),'w--','LineWidth',1.2);
xline(slowGamma(2),'w--','LineWidth',1.2);
xline(fastGamma(1),'r--','LineWidth',1.2);
xline(fastGamma(2),'r--','LineWidth',1.2);
xlabel('Frequency (Hz)'); ylabel('Electrode #');
title(sprintf('PSD per electrode — full epoch (per-trial Welch, n=%d trials)', nGood));
xlim(freqRange);
saveas(fig2, fullfile(outDir,'PSD_heatmap.png'));

% Fig 3: Spectrogram (all electrodes averaged)

fig3 = figure('Name','Spectrogram','Color','w','Position',[50 50 1050 480]);
fMask_tf = F_tf >= freqRange(1) & F_tf <= freqRange(2);
tMask_tf = T_axis >= epochStart & T_axis <= epochEnd;

imagesc(T_axis(tMask_tf), F_tf(fMask_tf), S_dB(fMask_tf, tMask_tf));
axis xy; colormap(jet);
cb = colorbar; cb.Label.String = 'Power (dB)';
xline(0,'w--','LineWidth',1.8);
yline(slowGamma(1),'w:','LineWidth',1.2); yline(slowGamma(2),'w:','LineWidth',1.2);
yline(fastGamma(1),'r:','LineWidth',1.2); yline(fastGamma(2),'r:','LineWidth',1.2);
xlabel('Time (s)'); ylabel('Frequency (Hz)');
title(sprintf('Trial- & electrode-averaged spectrogram — all %d electrodes (n=%d trials)', nElec, nGood));
saveas(fig3, fullfile(outDir,'TF_spectrogram.png'));

% Fig 4: Gamma power time courses — SEM across electrodes 
%           (primary SEM is across electrodes)

fig4 = figure('Name','Gamma time courses','Color','w','Position',[50 50 1050 600]);
for g = 1:nGB
    subplot(nGB, 1, g);
    col = gammaBands{g,3};
    tc  = gammaMeanTC(g,:);
    sem = gammaSEM_elec(g,:);

    patch([timeEpoch fliplr(timeEpoch)], [tc-sem, fliplr(tc+sem)], col, ...
          'FaceAlpha',0.25,'EdgeColor','none');
    hold on;
    plot(timeEpoch, tc, 'Color', col, 'LineWidth', 1.8);
    xline(0,'--k','Stimulus onset','LabelVerticalAlignment','bottom','LineWidth',1.2);
    xline(baselineEnd,':k','LineWidth',0.8);
    ylabel('Power (z^2)');
    title(sprintf('%s | mean \\pm SEM across electrodes (%d elec, %d trials)', ...
          gammaBands{g,2}, nElec, nGood));
    grid on; box off;
end
xlabel('Time (s)');
sgtitle('Instantaneous gamma band power (Hilbert envelope²) — shaded: SEM across electrodes');
saveas(fig4, fullfile(outDir,'gammaPower_timecourse.png'));

% Fig 5a: Slow vs Fast absolute power overlay

fig5a = figure('Name','Slow vs Fast – absolute','Color','w','Position',[50 50 1000 400]);
hold on;
for g = 1:nGB
    plot(timeEpoch, gammaMeanTC(g,:), 'Color', gammaBands{g,3}, ...
         'LineWidth', 2, 'DisplayName', gammaBands{g,2});
end
xline(0,'--k','Stimulus onset','LabelVerticalAlignment','bottom','LineWidth',1.2);
xlabel('Time (s)'); ylabel('Mean power (z^2)');
title('Slow vs Fast gamma — absolute power (mean across all electrodes & trials)');
legend('Location','northwest'); grid on; box off;
saveas(fig5a, fullfile(outDir,'slowVsFast_absolute.png'));

% Fig 5b: Slow vs Fast PERCENT CHANGE overlay 

fig5b = figure('Name','Slow vs Fast – % change','Color','w','Position',[50 50 1000 400]);
hold on;
for g = 1:nGB
    plot(timeEpoch, gammaPctChange(g,:), 'Color', gammaBands{g,3}, ...
         'LineWidth', 2, 'DisplayName', gammaBands{g,2});
end
yline(0,'--k','Baseline','LabelVerticalAlignment','bottom','LineWidth',1);
xline(0,'--k','Stimulus onset','LabelVerticalAlignment','bottom','LineWidth',1.2);
xlabel('Time (s)'); ylabel('Power change from baseline (%)');
title('Slow vs Fast gamma — % change from pre-stimulus baseline');
legend('Location','northwest'); grid on; box off;
saveas(fig5b, fullfile(outDir,'slowVsFast_pctChange.png'));

%% 11. SAVE RESULTS

save(fullfile(outDir,'spectralResults_v2.mat'), ...
    'pxx_full_dB',  'pxx_norm_dB',   ...
    'freqPSD',      'meanPSD_full',   'semPSD_full', ...
    'meanPSD_norm', 'semPSD_norm',    ...
    'bandPower',    ...
    'S_dB', 'F_tf', 'T_axis',        ...
    'gammaMeanTC',  'gammaPctChange', ...
    'gammaSEM_elec','gammaSEM_trials',...
    'timeEpoch',    'goodTrials',     'nGood', 'Fs', ...
    'slowGamma',    'fastGamma',      ...
    '-v7.3');

fprintf('\nAll figures and results saved to: %s/\n', outDir);
fprintf('Done.\n');