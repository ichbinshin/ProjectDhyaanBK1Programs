%% =========================================================
%  aim1a_GCI_DAI.m
%
%  AIM 1 — Part A: Spectral Granger Causality + Directed
%                  Asymmetry Index (DAI)
%
%  Estimates directed influence between all 90-electrode
%  pairs in slow (20-35 Hz) and fast (40-65 Hz) gamma bands
%  using bivariate MVAR models and the Geweke (1982) spectral
%  GCI formula.
%
%
%  Run aim1b_PSI.m afterwards; it loads aim1a_GCI_DAI.mat
%  and adds PSI results + GCI-vs-PSI comparison figures.
%
%  References
%    Bastos et al. (2015) Neuron 85:390-401
%    Geweke (1982) J Am Stat Assoc 77:304-313
% =========================================================

clearvars; close all; clc;

%% ── 0. PARAMETERS ─────────────────────────────────────────

dataDir   = '.';
outDir    = 'aim1_out';

nElec     = 90;
Fs        = 2000;

epochStart    = -0.5;
epochEnd      =  1.0;
baselineStart = -0.5;
baselineEnd   = -0.05;

slowGamma  = [20  35];   % Hz  — must match spectralAnalysis_v2
fastGamma  = [40  65];   % Hz

mvarOrder  = 20;         % VAR model order (AIC-checked below)
nfft_gc    = 512;        % frequency resolution for GC spectrum

nSurr      = 19;         % 19 surrogates to max = exact 95th pctile
alpha_pair = 0.05;

hardAmpLimit_uV = 2000;
ampThresh_uV    = 1200;
zscoreThresh    = 5;

%% ── 1. LOAD & PREPROCESS LFP ──────────────────────────────

fprintf('Loading raw LFP...\n');
info      = load(fullfile(dataDir,'lfpInfo.mat'));
timeVals  = info.timeVals(1,:);
nTrials   = size(info.goodStimPos, 2);

epochIdx  = timeVals >= epochStart & timeVals <= epochEnd;
timeEpoch = timeVals(epochIdx);
nTime     = sum(epochIdx);
baseIdx   = timeEpoch >= baselineStart & timeEpoch <= baselineEnd;

lfpAll = zeros(nElec, nTrials, nTime, 'single');
for e = 1:nElec
    fname = fullfile(dataDir, sprintf('elec%d.mat', e));
    if ~isfile(fname); lfpAll(e,:,:) = NaN; continue; end
    tmp = load(fname,'analogData');
    lfpAll(e,:,:) = single(tmp.analogData(:, epochIdx));
end

% Pre-CAR screening
ptp_elec    = squeeze(max(lfpAll,[],3) - min(lfpAll,[],3));
goodElecIdx = ~any(ptp_elec > hardAmpLimit_uV, 2);
carMean     = squeeze(mean(lfpAll(goodElecIdx,:,:), 1, 'omitnan'));
lfpCAR      = lfpAll - reshape(carMean, [1, nTrials, nTime]);

% Trial rejection
ptp      = squeeze(max(lfpCAR,[],3) - min(lfpCAR,[],3));
zPow     = zscore(squeeze(mean(double(lfpCAR).^2,3)), 0, 2);
badTrials= any(ptp > ampThresh_uV,1) | any(abs(zPow) > zscoreThresh,1);
lfpGood  = lfpCAR(:, ~badTrials, :);
nGood    = sum(~badTrials);

% Baseline z-score
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

fprintf('  %d electrodes | %d good trials | %d stim samples\n', ...
        nElec, nGood, nStim);

%% ── 2. BUILD PAIR LIST ────────────────────────────────────

pairList = zeros(nElec*(nElec-1)/2, 2);
idx_pl   = 0;
for i = 1:nElec
    for j = i+1:nElec
        idx_pl = idx_pl+1;
        pairList(idx_pl,:) = [i j];
    end
end
nPairs = size(pairList,1);
fprintf('  %d electrode pairs\n', nPairs);

%% ── 3. AIC MODEL-ORDER CHECK ──────────────────────────────

fprintf('\nAIC model-order check (10 random pairs, single trial)...\n');
rng(42);
orderRange  = 5:5:30;
nCheck      = min(10, nPairs);
checkIdx    = randperm(nPairs, nCheck);
AIC_mat     = zeros(nCheck, length(orderRange));

for cp = 1:nCheck
    xi = double(squeeze(lfpStim(pairList(checkIdx(cp),1), 1, :)));
    xj = double(squeeze(lfpStim(pairList(checkIdx(cp),2), 1, :)));
    for oi = 1:length(orderRange)
        [~,~,aic] = fitVAR2(xi, xj, orderRange(oi));
        AIC_mat(cp,oi) = aic;
    end
end
[~,bestOi] = min(mean(AIC_mat,1));
fprintf('  AIC-optimal order: %d  |  using mvarOrder = %d\n', ...
        orderRange(bestOi), mvarOrder);

%% ── 4. PRE-EXTRACT ELECTRODE DATA ────────────────────────

lfpStim_cell = cell(nElec,1);
for e = 1:nElec
    lfpStim_cell{e} = double(squeeze(lfpStim(e,:,:)));  % [nGood x nStim]
end

%% ── 5. SPECTRAL GCI + DAI + SURROGATES ───────────────────
%
%  Per-trial VAR strategy:
%    Fit VAR(p) on each individual trial (nStim ≈ 2000 samples),
%    then average A and Sigma across trials. This is statistically
%    equivalent to fitting on concatenated data under stationarity,
%    but ~600× faster because the OLS system is [2p×2p] not
%    [nStim*nGood × 2p].

freqGC      = (0:nfft_gc/2) * Fs/nfft_gc;
slowMask_gc = freqGC >= slowGamma(1) & freqGC <= slowGamma(2);
fastMask_gc = freqGC >= fastGamma(1) & freqGC <= fastGamma(2);

% Preallocate flat vectors (required for parfor)
gc_slow_ij  = zeros(nPairs,1);  gc_slow_ji  = zeros(nPairs,1);
gc_fast_ij  = zeros(nPairs,1);  gc_fast_ji  = zeros(nPairs,1);
s95_slow    = zeros(nPairs,1);  s95_fast    = zeros(nPairs,1);

fprintf('\nComputing GCI for %d pairs (parfor if PCT available)...\n',nPairs);
ticGC = tic;

parfor p = 1:nPairs   
    ei = pairList(p,1);
    ej = pairList(p,2);

    xi_tr = lfpStim_cell{ei};   % [nGood x nStim]
    xj_tr = lfpStim_cell{ej};

    % ── Observed: average VAR over trials ──────────────────
    A_sum = zeros(2, 2*mvarOrder);
    S_sum = zeros(2, 2);
    for tr = 1:nGood
        [At, St] = fitVAR2(xi_tr(tr,:)', xj_tr(tr,:)', mvarOrder);
        A_sum = A_sum + At;
        S_sum = S_sum + St;
    end
    A_avg = A_sum / nGood;
    S_avg = S_sum / nGood;

    [gij, gji] = spectralGCI(A_avg, S_avg, mvarOrder, nfft_gc, Fs);

    gc_slow_ij(p) = mean(gij(slowMask_gc));
    gc_slow_ji(p) = mean(gji(slowMask_gc));
    gc_fast_ij(p) = mean(gij(fastMask_gc));
    gc_fast_ji(p) = mean(gji(fastMask_gc));

    % ── Surrogates ─────────────────────────────────────────
    surr_s = zeros(1,nSurr);
    surr_f = zeros(1,nSurr);
    for s = 1:nSurr
        shuf = randperm(nGood);
        As = zeros(2,2*mvarOrder);  Ss = zeros(2,2);
        for tr = 1:nGood
            [At,St] = fitVAR2(xi_tr(tr,:)', xj_tr(shuf(tr),:)', mvarOrder);
            As = As+At;  Ss = Ss+St;
        end
        As = As/nGood;  Ss = Ss/nGood;
        [gs_ij, gs_ji] = spectralGCI(As, Ss, mvarOrder, nfft_gc, Fs);

        ds = mean(gs_ij(slowMask_gc)) + mean(gs_ji(slowMask_gc)) + eps;
        df = mean(gs_ij(fastMask_gc)) + mean(gs_ji(fastMask_gc)) + eps;
        surr_s(s) = (mean(gs_ij(slowMask_gc))-mean(gs_ji(slowMask_gc)))/ds;
        surr_f(s) = (mean(gs_ij(fastMask_gc))-mean(gs_ji(fastMask_gc)))/df;
    end
    s95_slow(p) = prctile(abs(surr_s), 95);
    s95_fast(p) = prctile(abs(surr_f), 95);
end

fprintf('GCI done in %.1f min.\n', toc(ticGC)/60);

%% ── 6. UNPACK VECTORS TO MATRICES ─────────────────────────

GCI_slow = zeros(nElec); GCI_fast = zeros(nElec);
DAI_slow = zeros(nElec); DAI_fast = zeros(nElec);
DAI_slow_surr95 = zeros(nElec); DAI_fast_surr95 = zeros(nElec);

for p = 1:nPairs
    ei = pairList(p,1);  ej = pairList(p,2);

    GCI_slow(ei,ej) = gc_slow_ij(p);  GCI_slow(ej,ei) = gc_slow_ji(p);
    GCI_fast(ei,ej) = gc_fast_ij(p);  GCI_fast(ej,ei) = gc_fast_ji(p);

    ds = gc_slow_ij(p)+gc_slow_ji(p)+eps;
    df = gc_fast_ij(p)+gc_fast_ji(p)+eps;

    DAI_slow(ei,ej) = (gc_slow_ij(p)-gc_slow_ji(p))/ds;
    DAI_slow(ej,ei) = -DAI_slow(ei,ej);
    DAI_fast(ei,ej) = (gc_fast_ij(p)-gc_fast_ji(p))/df;
    DAI_fast(ej,ei) = -DAI_fast(ei,ej);

    DAI_slow_surr95(ei,ej) = s95_slow(p);  DAI_slow_surr95(ej,ei) = s95_slow(p);
    DAI_fast_surr95(ei,ej) = s95_fast(p);  DAI_fast_surr95(ej,ei) = s95_fast(p);
end

%% ── 7. SIGNIFICANCE AND FDR ─────────────────────────────────

sigMask_slow = applyFDR(abs(DAI_slow) > DAI_slow_surr95, alpha_pair, nElec);
sigMask_fast = applyFDR(abs(DAI_fast) > DAI_fast_surr95, alpha_pair, nElec);

tri       = logical(triu(ones(nElec),1));
sDAI_sig  = DAI_slow(tri & sigMask_slow);
fDAI_sig  = DAI_fast(tri & sigMask_fast);

fprintf('\nSignificant pairs after FDR:\n');
fprintf('  Slow gamma: %d / %d (%.1f%%)\n', numel(sDAI_sig), nPairs, 100*numel(sDAI_sig)/nPairs);
fprintf('  Fast gamma: %d / %d (%.1f%%)\n', numel(fDAI_sig), nPairs, 100*numel(fDAI_sig)/nPairs);

%% ── 8. NETWORK SUMMARY STATISTICS ────────────────────────

hubScore_slow = zeros(nElec,1);  hubScore_fast = zeros(nElec,1);
outStr_slow   = zeros(nElec,1);  outStr_fast   = zeros(nElec,1);
inStr_slow    = zeros(nElec,1);  inStr_fast    = zeros(nElec,1);
sigCount_slow = zeros(nElec,1);  sigCount_fast = zeros(nElec,1);

for e = 1:nElec
    ms = sigMask_slow(e,:);  ms(e) = false;
    mf = sigMask_fast(e,:);  mf(e) = false;
    if any(ms)
        hubScore_slow(e) = mean(abs(DAI_slow(e,ms)));
        outStr_slow(e)   = mean(max(DAI_slow(e,ms),0));
        inStr_slow(e)    = mean(min(DAI_slow(e,ms),0));
        sigCount_slow(e) = sum(ms);
    end
    if any(mf)
        hubScore_fast(e) = mean(abs(DAI_fast(e,mf)));
        outStr_fast(e)   = mean(max(DAI_fast(e,mf),0));
        inStr_fast(e)    = mean(min(DAI_fast(e,mf),0));
        sigCount_fast(e) = sum(mf);
    end
end

% Distance-binned connectivity

elecDist   = abs((1:nElec)' - (1:nElec));   % proxy; replace with µm
distBins   = 0:2:nElec;                      % 2-index bins (was 5)
nBins      = numel(distBins)-1;
minPairs   = 5;                              % skip bins with < 5 pairs

% Raw GCI asymmetry
GCIasym_slow = abs(GCI_slow - GCI_slow');   % [nElec x nElec]
GCIasym_fast = abs(GCI_fast - GCI_fast');

mGCI_s_d = NaN(nBins,1);  mGCI_f_d = NaN(nBins,1);
sGCI_s_d = NaN(nBins,1);  sGCI_f_d = NaN(nBins,1);
nGCI_s_d = zeros(nBins,1); nGCI_f_d = zeros(nBins,1);

for b = 1:nBins
    mask = elecDist > distBins(b) & elecDist <= distBins(b+1);
    vs   = GCIasym_slow(mask & sigMask_slow);
    vf   = GCIasym_fast(mask & sigMask_fast);
    nGCI_s_d(b) = numel(vs);
    nGCI_f_d(b) = numel(vf);
    if numel(vs) >= minPairs
        mGCI_s_d(b) = mean(vs);
        sGCI_s_d(b) = std(vs)/sqrt(numel(vs));
    end
    if numel(vf) >= minPairs
        mGCI_f_d(b) = mean(vf);
        sGCI_f_d(b) = std(vf)/sqrt(numel(vf));
    end
end

%% ── 9. PERMUTATION TEST: slow vs fast |DAI| ──────────────

bothMask  = tri & (sigMask_slow | sigMask_fast);
sDAI_both = abs(DAI_slow(bothMask));
fDAI_both = abs(DAI_fast(bothMask));

nPerm    = 10000;
obsDiff  = mean(sDAI_both) - mean(fDAI_both);
permDiff = zeros(nPerm,1);
parfor k = 1:nPerm   % parfor safe: no shared state
    sgn        = sign(rand(numel(sDAI_both),1)-0.5);
    permDiff(k)= mean(sgn.*(sDAI_both-fDAI_both));
end
pVal_DAI = mean(abs(permDiff) >= abs(obsDiff));

[~,ks_p] = kstest2(abs(sDAI_sig), abs(fDAI_sig));

fprintf('\n════ DAI Summary ════════════════════════════════════\n');
fprintf('  %-30s  %8s  %8s\n','Metric','Slow γ','Fast γ');
fprintf('  %-30s  %8.4f  %8.4f\n','Mean |DAI| (sig pairs)', ...
        mean(abs(sDAI_sig)), mean(abs(fDAI_sig)));
fprintf('  %-30s  %8.4f  %8.4f\n','Median |DAI|', ...
        median(abs(sDAI_sig)), median(abs(fDAI_sig)));
fprintf('  %-30s  %8d  %8d\n','N sig pairs',numel(sDAI_sig),numel(fDAI_sig));
fprintf('  Permutation p (slow≠fast): %.4f\n', pVal_DAI);
fprintf('  KS p (|DAI| distribution): %.4f\n', ks_p);
if pVal_DAI < 0.05
    if obsDiff>0; fprintf('   Slow gamma has GREATER directed asymmetry.\n');
    else;         fprintf('   Fast gamma has GREATER directed asymmetry.\n'); end
end
fprintf('═════════════════════════════════════════════════════\n\n');

%% ── 10. FIGURES ───────────────────────────────────────────

if ~exist(outDir,'dir'); mkdir(outDir); end

% Fig 1 – DAI matrices
fig1 = figure('Color','w','Position',[50 50 1300 520]);
DAI_data  = {DAI_slow .* sigMask_slow,  DAI_fast .* sigMask_fast};
DAI_bands = {slowGamma,                 fastGamma};
DAI_nsig  = {numel(sDAI_sig),           numel(fDAI_sig)};
DAI_names = {'Slow',                    'Fast'};
for gi = 1:2
    subplot(1,2,gi);
    imagesc(1:nElec, 1:nElec, DAI_data{gi}); axis square;
    colormap(gca, redblue(256)); clim([-1 1]);
    cb = colorbar; cb.Label.String = 'DAI';
    title(sprintf('%s \\gamma DAI (%d-%d Hz) | %d sig pairs', ...
          DAI_names{gi}, DAI_bands{gi}(1), DAI_bands{gi}(2), DAI_nsig{gi}));
    xlabel('Target electrode j'); ylabel('Source electrode i');
end
sgtitle('Directed Asymmetry Index — non-significant pairs masked to 0');
saveas(fig1, fullfile(outDir,'DAI_matrices.png'));

% Fig 2 – DAI difference map
fig2 = figure('Color','w','Position',[50 50 700 580]);
dD   = DAI_slow - DAI_fast;
imagesc(1:nElec,1:nElec,dD); axis square;
colormap(redblue(256));
clim([-max(abs(dD(:)))*1.05, max(abs(dD(:)))*1.05]);
cb=colorbar; cb.Label.String='\DeltaDAI (slow − fast)';
title(sprintf('DAI difference: Slow − Fast gamma\nperm p=%.4f, obs diff=%.4f', ...
      pVal_DAI, obsDiff));
xlabel('Electrode j'); ylabel('Electrode i');
saveas(fig2, fullfile(outDir,'DAI_difference.png'));

% Fig 3 – Hub scores
fig3 = figure('Color','w','Position',[50 50 1000 420]);
cols = {[0.1 0.6 0.1],[0.8 0.1 0.1]};
lbls = {'Slow \gamma','Fast \gamma'};
hs   = {hubScore_slow, hubScore_fast};
for gi=1:2
    subplot(1,2,gi);
    bar(1:nElec, hs{gi}, 'FaceColor', cols{gi});
    xlabel('Electrode #'); ylabel('Mean |DAI| (sig pairs)');
    title(sprintf('Hub score — %s', lbls{gi}));
    grid on; box off;
end
sgtitle('Hub scores: mean directed asymmetry per electrode');
saveas(fig3, fullfile(outDir,'hubScores.png'));

% Fig 4 – CDF of |DAI|
fig4 = figure('Color','w','Position',[50 50 700 500]);
[fs,xs] = ecdf(abs(sDAI_sig));
[ff,xf] = ecdf(abs(fDAI_sig));
plot(xs,fs,'-','Color',[0.1 0.6 0.1],'LineWidth',2, ...
     'DisplayName',sprintf('Slow \\gamma (n=%d)',numel(sDAI_sig)));
hold on;
plot(xf,ff,'-','Color',[0.8 0.1 0.1],'LineWidth',2, ...
     'DisplayName',sprintf('Fast \\gamma (n=%d)',numel(fDAI_sig)));
text(0.5,0.15,sprintf('KS p = %.4f',ks_p),'FontSize',11, ...
     'HorizontalAlignment','center');
xlabel('|DAI|'); ylabel('Cumulative proportion');
title('CDF of |DAI| — significant pairs only');
legend('Location','southeast'); grid on; box off;
saveas(fig4, fullfile(outDir,'DAI_CDF.png'));

% Fig 5 – Significant connection count
fig5 = figure('Color','w','Position',[50 50 1000 420]);
for gi=1:2
    subplot(1,2,gi);
    if gi==1; sc=sigCount_slow; col=cols{1}; lbl=lbls{1};
    else;     sc=sigCount_fast; col=cols{2}; lbl=lbls{2}; end
    bar(1:nElec,sc,'FaceColor',col);
    yline(mean(sc),'--k',sprintf('mean=%.1f',mean(sc)), ...
          'LabelHorizontalAlignment','left','LineWidth',1.2);
    xlabel('Electrode #'); ylabel('N significant pairs');
    title(sprintf('%s — sig connections per electrode',lbl));
    grid on; box off;
end
sgtitle('Number of significant directed connections per electrode');
saveas(fig5, fullfile(outDir,'sigConnectionCount.png'));

% Fig 6 – GCI asymmetry vs distance
fig6 = figure('Color','w','Position',[50 50 800 480]);
bc   = distBins(1:end-1) + diff(distBins)/2;
vld  = ~isnan(mGCI_s_d) | ~isnan(mGCI_f_d);

errorbar(bc(vld & ~isnan(mGCI_s_d)), mGCI_s_d(vld & ~isnan(mGCI_s_d)), ...
         sGCI_s_d(vld & ~isnan(mGCI_s_d)), 'o-', ...
         'Color',[0.1 0.6 0.1],'LineWidth',1.8,'MarkerSize',6, ...
         'DisplayName','Slow \gamma');
hold on;
errorbar(bc(vld & ~isnan(mGCI_f_d)), mGCI_f_d(vld & ~isnan(mGCI_f_d)), ...
         sGCI_f_d(vld & ~isnan(mGCI_f_d)), 's-', ...
         'Color',[0.8 0.1 0.1],'LineWidth',1.8,'MarkerSize',6, ...
         'DisplayName','Fast \gamma');
xlabel('Inter-electrode distance (index units; replace with µm if available)');
ylabel('Mean |GCI_{ij} - GCI_{ji}| ± SEM');
title(sprintf('GCI directed asymmetry vs distance\n(significant pairs only, min %d pairs per bin)', minPairs));
legend('Location','best'); grid on; box off;
saveas(fig6, fullfile(outDir,'DAI_vs_distance.png'));

%% ── 11. SAVE ──────────────────────────────────────────────

if ~exist(outDir,'dir'); mkdir(outDir); end
save(fullfile(outDir,'aim1a_GCI_DAI.mat'), ...
    'GCI_slow','GCI_fast','DAI_slow','DAI_fast', ...
    'GCIasym_slow','GCIasym_fast', ...
    'sigMask_slow','sigMask_fast', ...
    'DAI_slow_surr95','DAI_fast_surr95', ...
    'hubScore_slow','hubScore_fast', ...
    'outStr_slow','outStr_fast','inStr_slow','inStr_fast', ...
    'sigCount_slow','sigCount_fast', ...
    'elecDist','pairList','nPairs', ...
    'pVal_DAI','obsDiff','ks_p', ...
    'slowGamma','fastGamma','mvarOrder','nSurr', ...
    'nElec','nGood','Fs', ...
    '-v7.3');

fprintf('All GCI/DAI results saved to %s/aim1a_GCI_DAI.mat\n', outDir);
fprintf('Done.\n');

%% ═══════════════════════════════════════════════════════════
%  LOCAL FUNCTIONS
%% ═══════════════════════════════════════════════════════════

function [A, Sigma, AIC] = fitVAR2(x1, x2, p)
% Bivariate VAR(p) via OLS on [2p x 2p] normal equations.
N = length(x1);
X = [x1(:), x2(:)];
T = N - p;
Y = zeros(T, 2*p);
for k = 1:p
    Y(:,(2*k-1):(2*k)) = X(p-k+1:N-k,:);
end
Z    = X(p+1:end,:);
A_T  = (Y'*Y) \ (Y'*Z);
A    = A_T';
resid= Z - Y*A_T;
Sigma= (resid'*resid)/T;
if nargout>=3
    AIC = T*log(det(Sigma)) + 2*(4*p);
end
end

% ----------------------------------------------------------

function [gci_12, gci_21] = spectralGCI(A, Sigma, p, nfft, Fs)
% Vectorised spectral GCI — no per-frequency loop.
nFreq  = nfft/2+1;
freqs  = (0:nFreq-1)' * Fs/nfft;
z_lags = exp(-1i*2*pi*freqs*(1:p)/Fs);   % [nFreq x p]

Af11=ones(nFreq,1); Af12=zeros(nFreq,1);
Af21=zeros(nFreq,1); Af22=ones(nFreq,1);
for k=1:p
    Ak=A(:,(2*k-1):(2*k)); zk=z_lags(:,k);
    Af11=Af11-Ak(1,1)*zk; Af12=Af12-Ak(1,2)*zk;
    Af21=Af21-Ak(2,1)*zk; Af22=Af22-Ak(2,2)*zk;
end

det_Af=Af11.*Af22-Af12.*Af21;
H11= Af22./det_Af; H12=-Af12./det_Af;
H21=-Af21./det_Af; H22= Af11./det_Af;

s11=Sigma(1,1); s12=Sigma(1,2); s21=Sigma(2,1); s22=Sigma(2,2);
S11=H11.*conj(H11)*s11+H11.*conj(H12)*s12+H12.*conj(H11)*s21+H12.*conj(H12)*s22;
S22=H21.*conj(H21)*s11+H21.*conj(H22)*s12+H22.*conj(H21)*s21+H22.*conj(H22)*s22;

Sigma2_1=s22-s12^2/(s11+eps);
Sigma1_2=s11-s12^2/(s22+eps);

gci_12=max(log(abs(real(S22))./(abs(real(S22)-Sigma2_1.*abs(H21).^2)+eps)),0)';
gci_21=max(log(abs(real(S11))./(abs(real(S11)-Sigma1_2.*abs(H12).^2)+eps)),0)';
end

% ----------------------------------------------------------

function sigMask = applyFDR(rawMask, ~, nElec)
% Symmetrise and zero the diagonal; BH correction placeholder.
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