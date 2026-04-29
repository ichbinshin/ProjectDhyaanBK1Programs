
function displayECGDataSingleSubject(subjectName,expDate,folderSourceString,channelNumber,medSegmentedFlag)

if ~exist('folderSourceString','var');    folderSourceString=[];        end

if isempty(folderSourceString)
    folderSourceString = 'N:\Projects\ProjectDhyaan\BK1';
end

if medSegmentedFlag
    protocolNameList = [{'EO1'}     {'EC1'}     {'G1'}      {'M1a'}         {'M1b'}         {'M1c'}          {'G2'}      {'EO2'}     {'EC2'}     {'M2a'}         {'M2b'}         {'M2c'}];
    colorNames       = [{[0.9 0 0]} {[0 0.9 0]} {[0 0 0.9]} {[0.9 0.9 0.9]} {[0.8 0.8 0.8]} {[0.7 0.7 0.7]}  {[0 0 0.3]} {[0.3 0 0]} {[0 0.3 0]} {[0.3 0.3 0.3]} {[0.2 0.2 0.2]} {[0.1 0.1 0.1]}];
else
    protocolNameList = [{'EO1'}     {'EC1'}     {'G1'}      {'M1'}          {'G2'}      {'EO2'}     {'EC2'}     {'M2'}];
    colorNames       = [{[0.9 0 0]} {[0 0.9 0]} {[0 0 0.9]} {[0.9 0.9 0.9]} {[0 0 0.3]} {[0.3 0 0]} {[0 0.3 0]} {[0.3 0.3 0.3]}];
end

numProtocols = length(protocolNameList);

hECGPlots = getPlotHandles(numProtocols,1,[0.05 0.05 0.7 0.9],0,0,1);
hRRIntervalPlot = subplot('Position',[0.8 0.55 0.15 0.4]);
hRRHistPlot = subplot('Position',[0.8 0.05 0.15 0.45]);

linkaxes(hECGPlots);

gridType = 'EEG';

for i = 1:numProtocols
    protocolName = protocolNameList{i};

    % Long segment file
    folderName = fullfile(folderSourceString,'data','segmentedDataLong',subjectName,gridType,expDate,protocolName);

    tmpFile = fullfile(folderName,'segmentedData','LFP',['elec' num2str(channelNumber) '.mat']);

    if exist(tmpFile,'file')
        % Get Good Times
        folderExtract = fullfile(folderName,'extractedData');
        tmp = load(fullfile(folderExtract,'goodStimTimes.mat'),'goodStimTimes');
        goodStimTimes = tmp.goodStimTimes;

        % Get Long segment
        tmp = load(tmpFile);
        analogData = tmp.analogData;

        % timeVals
        tmp = load(fullfile(folderName,'segmentedData','LFP','lfpInfo.mat'));
        timeVals = tmp.timeVals;

        plot(hECGPlots(i),timeVals,analogData,'color',colorNames{i}); 
        hold(hECGPlots(i),'on');
        plot(hECGPlots(i),goodStimTimes,zeros(1,length(goodStimTimes)),'ko');
        ylabel(hECGPlots(i),protocolNameList{i},'color',colorNames{i});

        if i<numProtocols
            set(hECGPlots(i),'XTickLabel',[],'YTickLabel',[]);
        else
            xlabel(hECGPlots(i),'Time (s)');
        end

    else
        disp([tmpFile ' not found']);
    end
end

xlim = get(hECGPlots(1),'XLim');
set(hECGPlots(1),'XLim',[0 xlim(2)]);