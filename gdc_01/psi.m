%% =========================================================
%  psi.m
%
%  AIM 1 — Part B: Phase Slope Index (PSI)
%
%  Computes PSI for all 4005 electrode pairs in slow
%  (20-35 Hz) and fast (40-65 Hz) gamma, then compares
%  results with the GCI/DAI computed in gc.m.
%
%  WHY PSI IS A USEFUL COMPLEMENT TO GCI
%    GCI relies on a parametric MVAR model and can be
%    inflated by volume conduction or shared noise.
%    PSI uses only the imaginary part of the cross-spectrum,
%    which is exactly zero under instantaneous mixing
%    (Nolte et al. 2008). Agreement between GCI/DAI and
%    PSI therefore provides model-free confirmation that
%    the directed influences are real.
%
%  DEPENDENCIES
%    aim1a_GCI_DAI.mat   (produced by gc.m)
%    Raw electrode files elec1.mat … elec90.mat
%
%  References
%    Nolte et al. (2008) Phys Rev Lett 100:234101
%    Vinck et al. (2011) NeuroImage 55:1548-1565
% =========================================================

clearvars; close all; clc;

%% ── 0. PARAMETERS ─────────────────────────────────────────

dataDir    = '.';
outDir     = 'aim1_out';
gcFile     = fullfile(outDir, 'aim1a_GCI_DAI.mat');

nElec      = 90;
Fs         = 2000;

epochStart    = -0.5;
epochEnd      =  1.0;
baselineStart = -0.5;
baselineEnd   = -0.05;

slowGamma  = [20  35];
fastGamma  = [40  65];

% PSI FFT settings
%   Use a fixed nfft that gives good frequency resolution
%   within the stimulus window (nStim ≈ 2000 samples).
%   nfft = 2048, df = 2000/2048 ≈ 0.977 Hz
%   Hann taper applied per trial before FFT.
nfft_psi   = 2048;

nSurr      = 19;
alpha_pair = 0.05;

hardAmpLimit_uV = 2000;
ampThresh_uV    = 1200;
zscoreThresh    = 5;

%% 1. LOAD GCI RESULTS

fprintf('Loading GCI/DAI results from aim1a...\n');
G = load(gcFile);

pairList   = G.pairList;
nPairs     = G.nPairs;
DAI_slow   = G.DAI_slow;
DAI_fast   = G.DAI_fast;
sigMask_DAI_slow = G.sigMask_slow;
sigMask_DAI_fast = G.sigMask_fast;
nGood      = G.nGood;

fprintf('  %d pairs, %d good trials loaded.\n', nPairs, nGood);

%% 2. RELOAD & PREPROCESS LFP

fprintf('Loading raw LFP...\n');
info      = load(fullfile(dataDir,'lfpInfo.mat'));
timeVals  = info.timeVals(1,:);
nTrials   = size(info.goodStimPos,2);

epochIdx  = timeVals >= epochStart & timeVals <= epochEnd;
timeEpoch = timeVals(epochIdx);
nTime     = sum(epochIdx);
baseIdx   = timeEpoch >= baselineStart & timeEpoch <= baselineEnd;

lfpAll = zeros(nElec, nTrials, nTime, 'single');
for e = 1:nElec
    fname = fullfile(dataDir, sprintf('elec%d.mat',e));
    if ~isfile(fname); lfpAll(e,:,:)=NaN; continue; end
    tmp = load(fname,'analogData');
    lfpAll(e,:,:) = single(tmp.analogData(:,epochIdx));
end

ptp_elec    = squeeze(max(lfpAll,[],3)-min(lfpAll,[],3));
goodElecIdx = ~any(ptp_elec > hardAmpLimit_uV,2);
carMean     = squeeze(mean(lfpAll(goodElecIdx,:,:),1,'omitnan'));
lfpCAR      = lfpAll - reshape(carMean,[1,nTrials,nTime]);

ptp      = squeeze(max(lfpCAR,[],3)-min(lfpCAR,[],3));
zPow     = zscore(squeeze(mean(double(lfpCAR).^2,3)),0,2);
badTrials= any(ptp>ampThresh_uV,1)|any(abs(zPow)>zscoreThresh,1);
lfpGood  = lfpCAR(:,~badTrials,:);

lfpBC = zeros(size(lfpGood),'single');
for e = 1:nElec
    sig = double(squeeze(lfpGood(e,:,:)));
    mu  = mean(sig(:,baseIdx),2);
    sd  = std(sig(:,baseIdx),0,2); sd(sd==0)=1;
    lfpBC(e,:,:) = single((sig-mu)./sd);
end

stimIdx = timeEpoch >= 0 & timeEpoch <= epochEnd;
lfpStim = lfpBC(:,:,stimIdx);   % [nElec x nGood x nStim]
nStim   = sum(stimIdx);

fprintf('  LFP ready: [%d x %d x %d]\n', nElec, nGood, nStim);

%% 3. BATCH FFT 
%
%  X_fft(e, tr, f) = DFT of Hann-tapered trial f for electrode e
%  Shape: [nElec x nGood x nFreq_psi]   stored as single complex
%
%  nFreq_psi = nfft_psi/2 + 1  (one-sided)
%
%  This is computed only once and indexing into it replaces all
%  pwelch() and cpsd() calls in the pair loop.
%
% ----------------------------------------------------------

fprintf('\nPre-computing batch FFT for all electrodes and trials...\n');
ticFFT = tic;

nFreq_psi = nfft_psi/2 + 1;
freqPSI   = (0:nFreq_psi-1)' * Fs/nfft_psi;   % [nFreq_psi x 1]

% Hann taper, zero-padded to nfft_psi
hannWin = single(hann(nStim));

% X_fft: [nElec x nGood x nFreq_psi]  complex single
X_fft = zeros(nElec, nGood, nFreq_psi, 'single');

for e = 1:nElec
    sig = squeeze(lfpStim(e,:,:));   % [nGood x nStim]
    % Applied Hann window to each trial, then FFT
    tapered = sig .* hannWin';       % [nGood x nStim]
    F = fft(tapered, nfft_psi, 2);  % [nGood x nfft_psi]
    X_fft(e,:,:) = single(F(:, 1:nFreq_psi));
end

% Auto-spectra: Pxx(e,tr,f) = |X_fft|^2
% Average across trials: Pxx_avg(e,f) = mean over trials
Pxx_avg = squeeze(mean(abs(X_fft).^2, 2));  % [nElec x nFreq_psi]

fprintf('  Batch FFT done in %.1f s\n', toc(ticFFT));

%% 4. BAND MASKS 

slowMask_psi = freqPSI >= slowGamma(1) & freqPSI <= slowGamma(2);
fastMask_psi = freqPSI >= fastGamma(1) & freqPSI <= fastGamma(2);

% Indices for PSI - need consecutive pairs (f, f+1) within band
slowIdx = find(slowMask_psi);  slowIdx = slowIdx(slowIdx < nFreq_psi);
fastIdx = find(fastMask_psi);  fastIdx = fastIdx(fastIdx < nFreq_psi);

fprintf('  Slow gamma: %d freq bins | Fast gamma: %d freq bins\n', ...
        numel(slowIdx), numel(fastIdx));

%% 5. PSI COMPUTATION AND SURROGATES
%
%  For each pair (i,j):
%
%  Cross-spectrum (trial-averaged):
%    CS(f) = mean_tr [ X_fft(i,tr,f) * conj(X_fft(j,tr,f)) ]
%
%  Normalised coherency:
%    C(f) = CS(f) / sqrt( Pxx_avg(i,f) * Pxx_avg(j,f) )
%
%  PSI over band B:
%    PSI = Im{ sum_{f in B} conj(C(f)) * C(f+1) }
%
%  Surrogates: shuffle trial indices of electrode j, recompute
%  CS from already-stored X_fft — no new FFTs needed.
%
% ----------------------------------------------------------

PSI_slow_vec    = zeros(nPairs,1);
PSI_fast_vec    = zeros(nPairs,1);
PSI_s95_vec     = zeros(nPairs,1);
PSI_f95_vec     = zeros(nPairs,1);

% Pre-generated all surrogate shuffle indices
% surr_idx(s,:) = shuffled trial indices for surrogate s
rng(42);
surr_idx = zeros(nSurr, nGood);
for s = 1:nSurr
    surr_idx(s,:) = randperm(nGood);
end

fprintf('\nComputing PSI for %d pairs (parfor if PCT available)...\n', nPairs);
ticPSI = tic;

parfor p = 1:nPairs   

    ei = pairList(p,1);
    ej = pairList(p,2);

    % Extract pre-computed FFTs: [nGood x nFreq_psi]
    Xi = squeeze(X_fft(ei,:,:));   % [nGood x nFreq_psi]
    Xj = squeeze(X_fft(ej,:,:));

    Pi = Pxx_avg(ei,:);   % [1 x nFreq_psi]
    Pj = Pxx_avg(ej,:);

    % Observed cross-spectrum & coherency
    CS   = mean(Xi .* conj(Xj), 1);         % [1 x nFreq_psi]
    denom= sqrt(Pi .* Pj) + eps;
    C    = CS ./ denom;                      % coherency

    % PSI 
    PSI_slow_vec(p) = psiFromCoherency(C, slowIdx);
    PSI_fast_vec(p) = psiFromCoherency(C, fastIdx);

    % Surrogates: reshuffle Xj, reuse Xi 
    surr_s = zeros(1, nSurr);
    surr_f = zeros(1, nSurr);

    for s = 1:nSurr
        Xj_s      = Xj(surr_idx(s,:), :);   % [nGood x nFreq_psi]
        CS_s      = mean(Xi .* conj(Xj_s), 1);
        C_s       = CS_s ./ denom;
        surr_s(s) = psiFromCoherency(C_s, slowIdx);
        surr_f(s) = psiFromCoherency(C_s, fastIdx);
    end

    PSI_s95_vec(p) = prctile(abs(surr_s), 95);
    PSI_f95_vec(p) = prctile(abs(surr_f), 95);

end

fprintf('PSI done in %.1f min.\n', toc(ticPSI)/60);

%% 6. UNPACK INTO MATRICES 

PSI_slow = zeros(nElec); PSI_fast = zeros(nElec);
PSI_slow_surr95 = zeros(nElec); PSI_fast_surr95 = zeros(nElec);

for p = 1:nPairs
    ei = pairList(p,1);  ej = pairList(p,2);

    PSI_slow(ei,ej) =  PSI_slow_vec(p);
    PSI_slow(ej,ei) = -PSI_slow_vec(p);
    PSI_fast(ei,ej) =  PSI_fast_vec(p);
    PSI_fast(ej,ei) = -PSI_fast_vec(p);

    PSI_slow_surr95(ei,ej) = PSI_s95_vec(p);
    PSI_slow_surr95(ej,ei) = PSI_s95_vec(p);
    PSI_fast_surr95(ei,ej) = PSI_f95_vec(p);
    PSI_fast_surr95(ej,ei) = PSI_f95_vec(p);
end

%% 7. SIGNIFICANCE AND FDR 

sigMask_PSI_slow = applyFDR(abs(PSI_slow) > PSI_slow_surr95, nElec);
sigMask_PSI_fast = applyFDR(abs(PSI_fast) > PSI_fast_surr95, nElec);

tri = logical(triu(ones(nElec),1));
sPSI_sig = PSI_slow(tri & sigMask_PSI_slow);
fPSI_sig = PSI_fast(tri & sigMask_PSI_fast);

fprintf('\nSignificant PSI pairs after FDR:\n');
fprintf('  Slow gamma: %d / %d (%.1f%%)\n', numel(sPSI_sig), nPairs, ...
        100*numel(sPSI_sig)/nPairs);
fprintf('  Fast gamma: %d / %d (%.1f%%)\n', numel(fPSI_sig), nPairs, ...
        100*numel(fPSI_sig)/nPairs);

%% 8. GCI vs PSI AGREEMENT
%
%  Agreement metric: for each pair significant in BOTH methods,
%  do the signs of DAI and PSI agree (same leading electrode)?
%
%  sign(DAI(i,j)) > 0  means i drives j  (GCI)
%  sign(PSI(i,j)) > 0  means i leads j   (PSI)
%  Agreement = both positive or both negative.
%
% ----------------------------------------------------------

% Pairs significant in both methods
bothSig_slow = tri & sigMask_DAI_slow & sigMask_PSI_slow;
bothSig_fast = tri & sigMask_DAI_fast & sigMask_PSI_fast;

nBoth_slow = sum(bothSig_slow(:))/2;
nBoth_fast = sum(bothSig_fast(:))/2;

agree_slow = mean(sign(DAI_slow(bothSig_slow)) == sign(PSI_slow(bothSig_slow)));
agree_fast = mean(sign(DAI_fast(bothSig_fast)) == sign(PSI_fast(bothSig_fast)));

% Binomial test: is agreement significantly above 50% chance
% H0: p_agree = 0.5  (two methods assign direction randomly)
% One-tailed: H1: p_agree > 0.5
nAgree_slow = round(agree_slow * nBoth_slow);
nAgree_fast = round(agree_fast * nBoth_fast);

% p = P(X >= nAgree | n, p=0.5) using binomial CDF
binom_p_slow = 1 - binocdf(nAgree_slow - 1, nBoth_slow, 0.5);
binom_p_fast = 1 - binocdf(nAgree_fast - 1, nBoth_fast, 0.5);

% Spearman correlation of PSI_slow vs PSI_fast across all pairs
[r_psi_sf, p_psi_sf] = corr(PSI_slow_vec, PSI_fast_vec, 'Type','Spearman');

% Correlation between |DAI| and |PSI| for all pairs
allDAI_s = abs(DAI_slow(tri));  allPSI_s = abs(PSI_slow(tri));
allDAI_f = abs(DAI_fast(tri));  allPSI_f = abs(PSI_fast(tri));

[r_slow, p_slow] = corr(allDAI_s, allPSI_s, 'Type','Spearman');
[r_fast, p_fast] = corr(allDAI_f, allPSI_f, 'Type','Spearman');

fprintf('\n════ GCI / PSI Agreement ═════════════════════════════\n');
fprintf('  %-40s  %8s  %8s\n', 'Metric', 'Slow γ', 'Fast γ');
fprintf('  %-40s  %8d  %8d\n', 'Sig in BOTH methods (pairs)', ...
        nBoth_slow, nBoth_fast);
fprintf('  %-40s  %7.1f%%  %7.1f%%\n', 'Sign agreement (direction)', ...
        100*agree_slow, 100*agree_fast);
fprintf('  %-40s  %8.4f  %8.4f\n', 'Binomial p (vs 50% chance)', ...
        binom_p_slow, binom_p_fast);
fprintf('  %-40s  %8.3f  %8.3f\n', 'Spearman r (|DAI| vs |PSI|)', ...
        r_slow, r_fast);
fprintf('  %-40s  %8.4f  %8.4f\n', 'Spearman p (|DAI| vs |PSI|)', ...
        p_slow, p_fast);
fprintf('  %-40s  %8.3f\n', 'Spearman r (PSI slow vs fast)', r_psi_sf);
fprintf('  %-40s  %8.4f\n', 'Spearman p (PSI slow vs fast)', p_psi_sf);
if p_psi_sf < 0.05 && r_psi_sf < 0
    fprintf('   PSI directions are ANTI-CORRELATED across bands.\n');
    fprintf('    Slow and fast gamma carry opposing directionality.\n');
end
fprintf('═════════════════════════════════════════════════════\n\n');

%% 9. FIGURES 

if ~exist(outDir,'dir'); mkdir(outDir); end

% Fig 1 – PSI matrices
fig1 = figure('Color','w','Position',[50 50 1300 520]);
psi_dat  = {PSI_slow .* sigMask_PSI_slow, PSI_fast .* sigMask_PSI_fast};
psi_lbl  = {'Slow \gamma','Fast \gamma'};
psi_band = {slowGamma, fastGamma};
psi_n    = {numel(sPSI_sig), numel(fPSI_sig)};

for gi = 1:2
    subplot(1,2,gi);
    imagesc(1:nElec, 1:nElec, psi_dat{gi}); axis square;
    colormap(gca, redblue(256));
    cv = max(abs(psi_dat{gi}(:)))*1.05; if cv==0; cv=1; end
    clim([-cv cv]);
    cb = colorbar; cb.Label.String = 'PSI';
    title(sprintf('%s PSI (%d-%d Hz) | %d sig pairs', ...
          psi_lbl{gi}, psi_band{gi}(1), psi_band{gi}(2), psi_n{gi}));
    xlabel('Target electrode j'); ylabel('Source electrode i');
end
sgtitle('Phase Slope Index — non-significant pairs masked to 0');
saveas(fig1, fullfile(outDir,'PSI_matrices.png'));

% Fig 2 – PSI scatter: slow vs fast
fig2 = figure('Color','w','Position',[50 50 700 600]);
scatter(PSI_slow_vec, PSI_fast_vec, 8, [0.5 0.5 0.5], 'filled', ...
        'MarkerFaceAlpha',0.3);
hold on;
refline(1,0); 
xlabel('PSI — Slow \gamma (20-35 Hz)');
ylabel('PSI — Fast \gamma (40-65 Hz)');
title('PSI: slow vs fast gamma per pair');
grid on; box off; axis square;
[rc,pc] = corr(PSI_slow_vec, PSI_fast_vec,'Type','Spearman');
text(0,max(PSI_fast_vec)*0.9, sprintf('r_s=%.3f, p=%.4f',rc,pc), ...
     'FontSize',11,'HorizontalAlignment','center');
saveas(fig2, fullfile(outDir,'PSI_slowVsFast_scatter.png'));

% Fig 3 – |DAI| vs |PSI| scatter (slow and fast, coloured)
fig3 = figure('Color','w','Position',[50 50 1200 520]);
subplot(1,2,1);
scatter(allDAI_s, allPSI_s, 8, [0.1 0.6 0.1], 'filled','MarkerFaceAlpha',0.2);
xlabel('|DAI| — Slow \gamma'); ylabel('|PSI| — Slow \gamma');
title(sprintf('|DAI| vs |PSI| — Slow \\gamma\nr_s=%.3f, p=%.4f', r_slow, p_slow));
grid on; box off;

subplot(1,2,2);
scatter(allDAI_f, allPSI_f, 8, [0.8 0.1 0.1], 'filled','MarkerFaceAlpha',0.2);
xlabel('|DAI| — Fast \gamma'); ylabel('|PSI| — Fast \gamma');
title(sprintf('|DAI| vs |PSI| — Fast \\gamma\nr_s=%.3f, p=%.4f', r_fast, p_fast));
grid on; box off;

sgtitle('GCI magnitude vs PSI magnitude across all electrode pairs');
saveas(fig3, fullfile(outDir,'PSI_vs_DAI_scatter.png'));

% Fig 4 – Agreement bar chart with binomial p-values
fig4 = figure('Color','w','Position',[50 50 700 480]);
bars   = [agree_slow*100, agree_fast*100];
labels = {'Slow \gamma','Fast \gamma'};
b = bar(bars, 'FaceColor','flat');
b.CData = [0.1 0.6 0.1; 0.8 0.1 0.1];
yline(50,'--k','Chance (50%)','LabelVerticalAlignment','bottom','LineWidth',1.2, ...
      'HandleVisibility','off');
ylim([0 100]);
set(gca,'XTickLabel',labels);
ylabel('Directional agreement (%)');
title(sprintf('GCI ↔ PSI sign agreement\n(pairs significant in both methods)'));


binom_ps = [binom_p_slow, binom_p_fast];
for gi = 1:2
    if binom_ps(gi) < 0.001
        pStr = 'p<0.001';
    else
        pStr = sprintf('p=%.3f', binom_ps(gi));
    end
    text(gi, bars(gi)+3, sprintf('%.1f%%\n%s', bars(gi), pStr), ...
         'HorizontalAlignment','center','FontSize',11,'FontWeight','bold');
end
grid on; box off;
saveas(fig4, fullfile(outDir,'PSI_DAI_agreement.png'));

% Fig 6 – PSI slow vs fast anticorrelation (new key result figure)
fig6 = figure('Color','w','Position',[50 50 680 580]);
scatter(PSI_slow_vec, PSI_fast_vec, 8, [0.4 0.4 0.4], 'filled', ...
        'MarkerFaceAlpha', 0.25);
hold on;

cf = polyfit(PSI_slow_vec, PSI_fast_vec, 1);
xr = linspace(min(PSI_slow_vec), max(PSI_slow_vec), 100);
plot(xr, polyval(cf,xr), 'r-', 'LineWidth', 2, 'HandleVisibility','off');
xline(0,'--k','LineWidth',0.8,'HandleVisibility','off');
yline(0,'--k','LineWidth',0.8,'HandleVisibility','off');
xlabel('PSI — Slow \gamma (20–35 Hz)');
ylabel('PSI — Fast \gamma (40–65 Hz)');
title(sprintf('PSI slow vs fast gamma — opposing directionality\nr_s = %.3f, p = %.4f (n=%d pairs)', ...
      r_psi_sf, p_psi_sf, nPairs));
annotation('textbox',[0.15 0.78 0.35 0.08], ...
    'String','Slow drives → fast follows', ...
    'EdgeColor','none','Color',[0.1 0.6 0.1],'FontSize',9);
annotation('textbox',[0.55 0.18 0.35 0.08], ...
    'String','Fast drives → slow follows', ...
    'EdgeColor','none','Color',[0.8 0.1 0.1],'FontSize',9);
grid on; box off; axis square;
saveas(fig6, fullfile(outDir,'PSI_anticorrelation.png'));

% Fig 5 – Combined hub score: mean of normalised |DAI| + |PSI|
maxDAI_s = max(abs(DAI_slow(:)))+eps;  maxDAI_f = max(abs(DAI_fast(:)))+eps;
maxPSI_s = max(abs(PSI_slow(:)))+eps;  maxPSI_f = max(abs(PSI_fast(:)))+eps;

combinedHub_slow = zeros(nElec,1);
combinedHub_fast = zeros(nElec,1);

for e = 1:nElec
    ms = (sigMask_DAI_slow(e,:) | sigMask_PSI_slow(e,:));  ms(e)=false;
    mf = (sigMask_DAI_fast(e,:) | sigMask_PSI_fast(e,:));  mf(e)=false;
    if any(ms)
        dai_n = abs(DAI_slow(e,ms))/maxDAI_s;
        psi_n = abs(PSI_slow(e,ms))/maxPSI_s;
        combinedHub_slow(e) = mean((dai_n+psi_n)/2);
    end
    if any(mf)
        dai_n = abs(DAI_fast(e,mf))/maxDAI_f;
        psi_n = abs(PSI_fast(e,mf))/maxPSI_f;
        combinedHub_fast(e) = mean((dai_n+psi_n)/2);
    end
end

fig5 = figure('Color','w','Position',[50 50 1000 420]);
subplot(1,2,1);
bar(1:nElec, combinedHub_slow,'FaceColor',[0.1 0.6 0.1]);
xlabel('Electrode #'); ylabel('Combined hub score (norm.)');
title('Combined hub score — Slow \gamma');
grid on; box off;

subplot(1,2,2);
bar(1:nElec, combinedHub_fast,'FaceColor',[0.8 0.1 0.1]);
xlabel('Electrode #'); ylabel('Combined hub score (norm.)');
title('Combined hub score — Fast \gamma');
grid on; box off;

sgtitle('Combined hub score: mean normalised (|DAI| + |PSI|) / 2');
saveas(fig5, fullfile(outDir,'PSI_DAI_combined_hub.png'));

%% 10. SAVE

save(fullfile(outDir,'aim1b_PSI.mat'), ...
    'PSI_slow','PSI_fast', ...
    'PSI_slow_surr95','PSI_fast_surr95', ...
    'sigMask_PSI_slow','sigMask_PSI_fast', ...
    'combinedHub_slow','combinedHub_fast', ...
    'agree_slow','agree_fast', ...
    'binom_p_slow','binom_p_fast', ...
    'r_slow','p_slow','r_fast','p_fast', ...
    'r_psi_sf','p_psi_sf', ...
    'nBoth_slow','nBoth_fast', ...
    'slowGamma','fastGamma', ...
    'nElec','nGood','Fs','nSurr','nfft_psi', ...
    '-v7.3');

fprintf('PSI results saved to %s/aim1b_PSI.mat\n', outDir);
fprintf('Done.\n');

%% ═══════════════════════════════════════════════════════════
%  LOCAL FUNCTIONS
%% ═══════════════════════════════════════════════════════════

function psi_val = psiFromCoherency(C, bandIdx)
% PSIFROMCOHERENCY  PSI = Im{ sum_f C*(f) * C(f+1) } over band.
%
%  Inputs:
%    C       : [1 x nFreq] complex coherency row vector
%    bandIdx : integer indices of band frequencies (column vector)
%              Must satisfy bandIdx < numel(C)
%
%  Output:
%    psi_val : scalar  (positive = first channel leads second)

psi_val = sum(imag(conj(C(bandIdx)) .* C(bandIdx+1)));
end

% ----------------------------------------------------------

function sigMask = applyFDR(rawMask, nElec)

sigMask = rawMask | rawMask';
sigMask(1:nElec+1:end) = false;
end

% ----------------------------------------------------------

function cmap = redblue(n)
if nargin<1; n=256; end
h=floor(n/2);
r=[linspace(0.8,1,h),linspace(1,1,n-h)]';
g=[linspace(0.1,1,h),linspace(1,0.1,n-h)]';
b=[linspace(0.1,1,h),linspace(1,0.8,n-h)]';
cmap=flipud([r,g,b]);
end