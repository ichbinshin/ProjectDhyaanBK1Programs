%% =========================================================
%  cfc.m
%
%  AIM 2 - Part A: Cross-Frequency Coupling (CFC)
%
%  Tests whether slow gamma phase (20-35 Hz) entrains fast
%  gamma amplitude (40-65 Hz) in macaque (AlpaH) V1 LFP.
%
%  The comodulogram is the slowest section. With the current
%  settings (90 electrodes, ~625 trials, 8x15 freq grid, 19
%  surrogates), expect roughly tens of minutes on a single core
%  and lesser with parfor on multiple cores.
%
%  References
%    Tort et al. (2010) J Neurosci Methods 199:299-312
%    Canolty et al. (2006) Science 313:1626-1628
%% =========================================================

clearvars; close all; clc;

%% 0. PARAMETERS

dataDir   = '.';
outDir    = 'aim2_out';

nElec     = 90;
Fs        = 2000;

epochStart    = -0.5;
epochEnd      =  1.0;
baselineStart = -0.5;
baselineEnd   = -0.05;

% CFC frequency bands
slowGamma  = [20 35];   % Hz - phase signal
fastGamma  = [40 65];   % Hz - amplitude signal

% Comodulogram sweep ranges
comod_phaseRange = [5 40];    % phase frequency sweep (Hz)
comod_ampRange   = [30 100];  % amplitude frequency sweep (Hz)
comod_phaseStep  = 5;         % Hz
comod_ampStep    = 5;         % Hz
comod_bandwidth  = 5;         % full bandwidth (Hz)

% MI settings
nBins_phase = 18;   % 18 bins = 20 degrees each
nSurr       = 200;  % surrogates per electrode
alpha       = 0.05;

% Artifact rejection
hardAmpLimit_uV = 2000;
ampThresh_uV    = 1200;
zscoreThresh    = 5;

%% 1. LOAD AND PREPROCESS LFP

fprintf('Loading raw LFP...\n');
info      = load(fullfile(dataDir, 'lfpInfo.mat'));
timeVals  = info.timeVals(1, :);
nTrials   = size(info.goodStimPos, 2);

epochIdx  = timeVals >= epochStart & timeVals <= epochEnd;
timeEpoch = timeVals(epochIdx);
nTime     = sum(epochIdx);
baseIdx   = timeEpoch >= baselineStart & timeEpoch <= baselineEnd;

lfpAll = zeros(nElec, nTrials, nTime, 'single');
for e = 1:nElec
    fname = fullfile(dataDir, sprintf('elec%d.mat', e));
    if ~isfile(fname)
        lfpAll(e, :, :) = NaN;
        continue;
    end
    tmp = load(fname, 'analogData');
    lfpAll(e, :, :) = single(tmp.analogData(:, epochIdx));
end

% Pre-CAR screening
ptp_elec    = squeeze(max(lfpAll, [], 3) - min(lfpAll, [], 3));
goodElecIdx = ~any(ptp_elec > hardAmpLimit_uV, 2);
carMean     = squeeze(mean(lfpAll(goodElecIdx, :, :), 1, 'omitnan'));
lfpCAR      = lfpAll - reshape(carMean, [1, nTrials, nTime]);

% Trial rejection
ptp      = squeeze(max(lfpCAR, [], 3) - min(lfpCAR, [], 3));
zPow     = zscore(squeeze(mean(double(lfpCAR).^2, 3)), 0, 2);
badTrials= any(ptp > ampThresh_uV, 1) | any(abs(zPow) > zscoreThresh, 1);
lfpGood  = lfpCAR(:, ~badTrials, :);
nGood    = sum(~badTrials);

% Baseline z-score
lfpBC = zeros(size(lfpGood), 'single');
for e = 1:nElec
    sig = double(squeeze(lfpGood(e, :, :)));
    mu  = mean(sig(:, baseIdx), 2);
    sd  = std(sig(:, baseIdx), 0, 2);
    sd(sd == 0) = 1;
    lfpBC(e, :, :) = single((sig - mu) ./ sd);
end

% Using stimulus period only for MI / comodulogram statistics
stimIdx = timeEpoch >= 0 & timeEpoch <= epochEnd;
lfpStim = lfpBC(:, :, stimIdx);   % [nElec x nGood x nStim]
nStim   = sum(stimIdx);

fprintf('  %d electrodes | %d good trials | %d stim samples\n', ...
        nElec, nGood, nStim);

%% 2. PRE-COMPUTE PHASE & AMPLITUDE FOR TARGET BANDS
%
%  This section computes the specific Aim-2 test -
%    slow gamma phase (20-35 Hz) to fast gamma amplitude (40-65 Hz)
%
%  The comodulogram later sweeps other frequency pairs separately.

fprintf('\nPre-computing target-band phase and amplitude (parfor)...\n');

[bSlow, aSlow] = butter(4, slowGamma / (Fs/2), 'bandpass');
[bFast, aFast] = butter(4, fastGamma / (Fs/2), 'bandpass');

phase_sg = zeros(nElec, nGood, nStim, 'single');
ampl_fg  = zeros(nElec, nGood, nStim, 'single');

parfor e = 1:nElec
    ph_e  = zeros(nGood, nStim, 'single');
    amp_e = zeros(nGood, nStim, 'single');

    for tr = 1:nGood
        sig = double(squeeze(lfpStim(e, tr, :)));

        slowSig = filtfilt(bSlow, aSlow, sig);
        fastSig = filtfilt(bFast, aFast, sig);

        ph_e(tr, :)  = single(angle(hilbert(slowSig)));
        amp_e(tr, :) = single(abs(hilbert(fastSig)));
    end

    phase_sg(e, :, :) = ph_e;
    ampl_fg(e, :, :)  = amp_e;
end

fprintf('  Done.\n');

%% 3. MODULATION INDEX PER ELECTRODE

fprintf('\nComputing per-trial MI (%d electrodes, %d surrogates)...\n', ...
        nElec, nSurr);

MI_obs    = zeros(nElec, 1);
MI_pval   = zeros(nElec, 1);
MI_surr95 = zeros(nElec, 1);

% Shared surrogate permutations
rng(42);
surr_idx_MI = zeros(nSurr, nGood, 'uint16');
for s = 1:nSurr
    surr_idx_MI(s, :) = randperm(nGood);
end

ticMI = tic;

parfor e = 1:nElec
    ph_e  = squeeze(phase_sg(e, :, :));   % [nGood x nStim]
    amp_e = squeeze(ampl_fg(e, :, :));    % [nGood x nStim]

    % Observed MI
    mi_trials = zeros(nGood, 1);
    for tr = 1:nGood
        mi_trials(tr) = computeMI(ph_e(tr, :)', amp_e(tr, :)', nBins_phase);
    end
    mi_obs_e = mean(mi_trials);

    % Surrogates: shuffle amplitude trials
    mi_surr = zeros(nSurr, 1);
    for s = 1:nSurr
        shuf = double(surr_idx_MI(s, :));
        mi_s = zeros(nGood, 1);
        for tr = 1:nGood
            mi_s(tr) = computeMI(ph_e(tr, :)', amp_e(shuf(tr), :)', nBins_phase);
        end
        mi_surr(s) = mean(mi_s);
    end

    MI_obs(e)    = mi_obs_e;
    MI_pval(e)   = (sum(mi_surr >= mi_obs_e) + 1) / (nSurr + 1);
    MI_surr95(e) = prctile(mi_surr, 95);
end

fprintf('MI done in %.1f min.\n', toc(ticMI) / 60);

% FDR correction across electrodes
[MI_sig_FDR, ~] = bhFDR(MI_pval, alpha);

fprintf('\n================ MI Summary ======================\n');
fprintf('  Mean MI                     : %.6f +/- %.6f\n', ...
        mean(MI_obs), std(MI_obs));
fprintf('  Sig electrodes (raw p<%.2f) : %d / %d (%.1f%%)\n', ...
        alpha, sum(MI_pval < alpha), nElec, ...
        100 * sum(MI_pval < alpha) / nElec);
fprintf('  Sig electrodes (FDR)        : %d / %d (%.1f%%)\n', ...
        sum(MI_sig_FDR), nElec, 100 * sum(MI_sig_FDR) / nElec);
fprintf('=================================================\n\n');

%% 4. PHASE-BINNED AMPLITUDE

phaseBins   = linspace(-pi, pi, nBins_phase + 1);
binCentres  = phaseBins(1:end-1) + diff(phaseBins) / 2;
phaseBinAmp = zeros(nElec, nBins_phase);

parfor e = 1:nElec
    ph_e  = squeeze(phase_sg(e, :, :));
    amp_e = squeeze(ampl_fg(e, :, :));

    binAmp = zeros(nBins_phase, 1);
    for b = 1:nBins_phase
        mask = ph_e >= phaseBins(b) & ph_e < phaseBins(b + 1);
        if any(mask(:))
            binAmp(b) = mean(amp_e(mask));
        end
    end
    phaseBinAmp(e, :) = binAmp';
end

% Preferred phase per electrode
[~, prefPhaseIdx] = max(phaseBinAmp, [], 2);
prefPhase_rad     = binCentres(prefPhaseIdx)';

% Electrode set for the phase-binned figure
sigElecs = find(MI_sig_FDR);
sigLabel = 'FDR significant';
if isempty(sigElecs)
    sigElecs = find(MI_pval < alpha);
    sigLabel = 'raw p<0.05';
end
if isempty(sigElecs)
    [~, ord] = sort(MI_obs, 'descend');
    sigElecs = ord(1:min(3, nElec));
    sigLabel = 'top MI electrodes (no significant sites)';
end

meanBinAmp_sig = mean(phaseBinAmp(sigElecs, :), 1);

%% 5. PAC COMODULOGRAM
%
%  This section computes a Tort-style MI comodulogram across
%  many phase/amplitude frequency pairs.

fprintf('\nComputing PAC comodulogram (analytic FFT method)...\n');

phaseFreqs   = comod_phaseRange(1) : comod_phaseStep : comod_phaseRange(2);
ampFreqs     = comod_ampRange(1)   : comod_ampStep   : comod_ampRange(2);
nPF          = numel(phaseFreqs);
nAF          = numel(ampFreqs);
nSurr_comod  = 19;
halfBW       = comod_bandwidth / 2;

% FFT settings
nfft_cm   = 2 ^ nextpow2(nStim);
freqAxis  = (0:nfft_cm/2) * Fs / nfft_cm;
nFreq_cm  = numel(freqAxis);
hannWin   = single(hann(nStim))';

fprintf('  Pre-computing one-sided FFTs (%d elec x %d trials)...\n', nElec, nGood);
ticFFT = tic;

X_fft = complex(zeros(nElec, nGood, nFreq_cm, 'single'));
for e = 1:nElec
    sig = squeeze(lfpStim(e, :, :));      % [nGood x nStim]
    sig = sig - mean(sig, 2);             % remove per-trial DC offset
    F   = fft(sig .* hannWin, nfft_cm, 2);
    X_fft(e, :, :) = single(F(:, 1:nFreq_cm));
end

fprintf('  FFT done in %.1f s\n', toc(ticFFT));

% Shared surrogate permutations for comodulogram
rng(43);
surr_comod = zeros(nSurr_comod, nGood, 'uint16');
for s = 1:nSurr_comod
    surr_comod(s, :) = randperm(nGood);
end

PAC_rows      = zeros(nPF, nAF);
PAC_pval_rows = ones(nPF, nAF);

ticComod = tic;

parfor pi_ = 1:nPF
    fp = phaseFreqs(pi_);

    pac_row  = zeros(1, nAF);
    pval_row = ones(1, nAF);

    phase_cache = cell(nElec, 1);
    for e = 1:nElec
        Xf_e = squeeze(X_fft(e, :, :));   % [nGood x nFreq_cm]
        analytic_phase = analyticBandLimited(Xf_e, fp, halfBW, freqAxis, nfft_cm, nStim);
        phase_cache{e} = single(angle(analytic_phase));
    end

    for ai = 1:nAF
        fa = ampFreqs(ai);

        mi_obs_sum  = 0;
        mi_surr_sum = zeros(1, nSurr_comod);

        for e = 1:nElec
            Xf_e = squeeze(X_fft(e, :, :));   % [nGood x nFreq_cm]
            ph_e = phase_cache{e};            % [nGood x nStim]

            analytic_amp = analyticBandLimited(Xf_e, fa, halfBW, freqAxis, nfft_cm, nStim);
            amp_e        = single(abs(analytic_amp));

            % Observed
            mi_tr = zeros(nGood, 1);
            for tr = 1:nGood
                mi_tr(tr) = computeMI(ph_e(tr, :)', amp_e(tr, :)', nBins_phase);
            end
            mi_obs_sum = mi_obs_sum + mean(mi_tr);

            % Surrogates
            for s = 1:nSurr_comod
                shuf = double(surr_comod(s, :));
                mi_s = zeros(nGood, 1);
                for tr = 1:nGood
                    mi_s(tr) = computeMI(ph_e(tr, :)', amp_e(shuf(tr), :)', nBins_phase);
                end
                mi_surr_sum(s) = mi_surr_sum(s) + mean(mi_s);
            end
        end

        obs_mean  = mi_obs_sum  / nElec;
        surr_mean = mi_surr_sum / nElec;

        pac_row(ai)  = obs_mean;
        pval_row(ai) = (sum(surr_mean >= obs_mean) + 1) / (nSurr_comod + 1);
    end

    PAC_rows(pi_, :)      = pac_row;
    PAC_pval_rows(pi_, :) = pval_row;
end

PAC_comod      = PAC_rows;
PAC_comod_pval = PAC_pval_rows;

fprintf('Comodulogram done in %.1f min.\n', toc(ticComod) / 60);

% Peak coupling frequency pair
[~, pkIdx] = max(PAC_comod(:));
[pkRow, pkCol] = ind2sub(size(PAC_comod), pkIdx);
fprintf('  Peak coupling: phase %.0f Hz x amplitude %.0f Hz (MI=%.6f)\n', ...
        phaseFreqs(pkRow), ampFreqs(pkCol), PAC_comod(pkRow, pkCol));

%% 6. FIGURES 

if ~exist(outDir, 'dir')
    mkdir(outDir);
end

% Fig 1: MI per electrode
fig1 = figure('Color', 'w', 'Position', [50 50 1100 420]);
bar(1:nElec, MI_obs, 'FaceColor', [0.2 0.6 0.8], 'EdgeColor', 'none');
hold on;
if any(MI_sig_FDR)
    bar(find(MI_sig_FDR), MI_obs(MI_sig_FDR), ...
        'FaceColor', [0.9 0.3 0.1], 'EdgeColor', 'none');
end
yline(mean(MI_surr95), '--k', ...
      sprintf('Mean surr 95th pctile = %.6f', mean(MI_surr95)), ...
      'LabelVerticalAlignment', 'bottom', 'LineWidth', 1.2, ...
      'HandleVisibility', 'off');
xlabel('Electrode #');
ylabel('Modulation Index');
title(sprintf('Per-trial MI: slow gamma phase (%d-%d Hz) -> fast gamma amplitude (%d-%d Hz)', ...
      slowGamma(1), slowGamma(2), fastGamma(1), fastGamma(2)));
legend({'Not significant', 'FDR significant'}, 'Location', 'northeast');
grid on; box off;
saveas(fig1, fullfile(outDir, 'MI_per_electrode.png'));

% Fig 2: MI distribution
fig2 = figure('Color', 'w', 'Position', [50 50 700 500]);
histogram(MI_obs, 20, 'FaceColor', [0.2 0.6 0.8], 'EdgeColor', 'w');
hold on;
xline(mean(MI_obs), '--k', sprintf('Mean=%.6f', mean(MI_obs)), ...
      'LineWidth', 1.8, 'LabelVerticalAlignment', 'top', 'HandleVisibility', 'off');
xline(mean(MI_surr95), '--r', sprintf('Surr 95th=%.6f', mean(MI_surr95)), ...
      'LineWidth', 1.8, 'LabelVerticalAlignment', 'top', 'HandleVisibility', 'off');
xlabel('Modulation Index');
ylabel('Number of electrodes');
title('Distribution of MI across electrodes (per-trial, stimulus period)');
grid on; box off;
saveas(fig2, fullfile(outDir, 'MI_distribution.png'));

% Fig 3: MI significance
fig3 = figure('Color', 'w', 'Position', [50 50 700 500]);
ptColors = double(MI_sig_FDR)  * [0.9 0.3 0.1] + ...
           double(~MI_sig_FDR) * [0.5 0.7 0.9];
scatter(MI_obs, -log10(MI_pval), 50, ptColors, ...
        'filled', 'MarkerFaceAlpha', 0.8);
hold on;
xline(mean(MI_surr95), '--r', 'Surr 95th', ...
      'LineWidth', 1.2, 'HandleVisibility', 'off');
yline(-log10(alpha), '--k', sprintf('p=%.2f', alpha), ...
      'LineWidth', 1.2, 'HandleVisibility', 'off');
xlabel('Modulation Index');
ylabel('-log_{10}(p-value)');
title('MI significance: each point = one electrode');
for e = find(MI_sig_FDR)'
    text(MI_obs(e) + 0.0002, -log10(MI_pval(e)) + 0.05, ...
         sprintf('%d', e), 'FontSize', 7, 'Color', [0.7 0.1 0]);
end
grid on; box off;
saveas(fig3, fullfile(outDir, 'MI_significance.png'));

% Fig 4: PAC comodulogram
fig4 = figure('Color', 'w', 'Position', [50 50 950 620]);
imagesc(ampFreqs, phaseFreqs, PAC_comod);
axis xy;
colormap(hot);
cb = colorbar;
cb.Label.String = 'Mean MI (per-trial, all electrodes)';
hold on;
[sr, sc] = find(PAC_comod_pval < alpha);
if ~isempty(sr)
    plot(ampFreqs(sc), phaseFreqs(sr), 'c*', 'MarkerSize', 8);
end
yline(slowGamma(1), 'w--', 'LineWidth', 1.2, 'HandleVisibility', 'off');
yline(slowGamma(2), 'w--', 'LineWidth', 1.2, 'HandleVisibility', 'off');
xline(fastGamma(1), 'w--', 'LineWidth', 1.2, 'HandleVisibility', 'off');
xline(fastGamma(2), 'w--', 'LineWidth', 1.2, 'HandleVisibility', 'off');
xlabel('Amplitude frequency (Hz)');
ylabel('Phase frequency (Hz)');
title(sprintf(['PAC comodulogram - mean MI across %d electrodes x %d trials\n' ...
               '(cyan * = significant at p<%.2f, white dashed = target bands)'], ...
      nElec, nGood, alpha));
saveas(fig4, fullfile(outDir, 'PAC_comodulogram.png'));

% Fig 5: Phase-binned amplitude
fig5 = figure('Color', 'w', 'Position', [50 50 800 500]);
binDeg = rad2deg(binCentres);

for e = sigElecs(:)'
    normAmp = phaseBinAmp(e, :) / max(phaseBinAmp(e, :) + eps);
    plot([binDeg, binDeg(1) + 360], [normAmp, normAmp(1)], ...
         '-', 'Color', [0.7 0.7 0.7], 'LineWidth', 0.8, ...
         'HandleVisibility', 'off');
    hold on;
end

normMean = meanBinAmp_sig / max(meanBinAmp_sig + eps);
plot([binDeg, binDeg(1) + 360], [normMean, normMean(1)], ...
     'b-', 'LineWidth', 2.5, ...
     'DisplayName', sprintf('Mean (n=%d, %s)', numel(sigElecs), sigLabel));

[~, pkBin] = max(normMean);
xline(binDeg(pkBin), '--r', sprintf('Preferred phase: %.0f deg', binDeg(pkBin)), ...
      'LineWidth', 1.5, 'HandleVisibility', 'off');

xlabel('Slow gamma phase (degrees)');
ylabel('Normalised fast gamma amplitude');
title('Fast gamma amplitude as function of slow gamma phase');
xlim([-180 540]);
xticks(-180:60:540);
legend('Location', 'northeast');
grid on; box off;
saveas(fig5, fullfile(outDir, 'phaseBinned_amplitude.png'));

%% 7. SAVE

save(fullfile(outDir, 'aim2a_CFC.mat'), ...
    'MI_obs', 'MI_pval', 'MI_surr95', 'MI_sig_FDR', ...
    'phaseBinAmp', 'binCentres', 'prefPhase_rad', ...
    'PAC_comod', 'PAC_comod_pval', ...
    'phaseFreqs', 'ampFreqs', ...
    'slowGamma', 'fastGamma', ...
    'nElec', 'nGood', 'Fs', 'nSurr', 'nBins_phase', ...
    'sigElecs', 'sigLabel', ...
    '-v7.3');

fprintf('\nCFC results saved to %s/aim2a_CFC.mat\n', outDir);
fprintf('Done.\n');

%% =========================================================
%  LOCAL FUNCTIONS
%% =========================================================

function MI = computeMI(phase, amplitude, nBins)
% COMPUTEMI  Tort modulation index.

phaseBins = linspace(-pi, pi, nBins + 1);
ampPerBin = zeros(nBins, 1);

for b = 1:nBins
    mask = phase >= phaseBins(b) & phase < phaseBins(b + 1);
    if any(mask)
        ampPerBin(b) = mean(amplitude(mask));
    end
end

P  = ampPerBin / (sum(ampPerBin) + eps);
H  = -sum(P .* log(P + eps));
MI = (log(nBins) - H) / log(nBins);
end

function analyticSig = analyticBandLimited(Xf, fc, halfBW, freqAx, nfft, nT)
% ANALYTICBANDLIMITED
% Building a true analytic, band-limited signal from the one-sided FFT.
%
% Inputs:
%   Xf      : [nTrials x (nfft/2+1)] one-sided FFT of a real signal
%   fc      : center frequency (Hz)
%   halfBW  : half-bandwidth (Hz)
%   freqAx  : one-sided frequency axis
%   nfft    : FFT length
%   nT      : number of time samples to keep
%
% Output:
%   analyticSig : [nTrials x nT] complex analytic band-limited signal

nTrials = size(Xf, 1);

posMask = freqAx >= (fc - halfBW) & freqAx <= (fc + halfBW);
Xa = complex(zeros(nTrials, nfft, 'single'));

% Doubling positive-frequency bins to form the analytic signal.
Xa(:, 1:nfft/2+1) = 0;
Xa(:, posMask)    = 2 * Xf(:, posMask);

% Preserving DC / Nyquist scaling if they happen to be selected.
if posMask(1)
    Xa(:, 1) = Xf(:, 1);
end
if mod(nfft, 2) == 0 && posMask(end)
    Xa(:, nfft/2 + 1) = Xf(:, end);
end

analyticSig = ifft(Xa, nfft, 2);
analyticSig = analyticSig(:, 1:nT);
end

function [sigMask, FDR_thresh] = bhFDR(pvals, alpha)
% BHFDR  Benjamini-Hochberg FDR correction.

m = numel(pvals);
[ps, idx] = sort(pvals(:));
thresh    = (1:m)' / m * alpha;
last      = find(ps <= thresh, 1, 'last');

sigMask = false(m, 1);
if ~isempty(last)
    sigMask(idx(1:last)) = true;
    FDR_thresh = thresh(last);
else
    FDR_thresh = NaN;
end
end
