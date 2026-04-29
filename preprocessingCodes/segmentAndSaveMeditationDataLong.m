% Meditation data is saved as M1a,b and c.

function segmentAndSaveMeditationDataLong(subjectName,expDate,folderSourceString,channelNumbers)

if ~exist('folderSourceString','var');    folderSourceString=[];        end

if isempty(folderSourceString)
    folderSourceString = 'D:\OneDrive - Indian Institute of Science\Supratim\Projects\ProjectDhyaan\BK1';
end

gridType = 'EEG';
protocolNameList = [{'M1'} {'M2'}];

segmentNames = [{'a'} {'b'} {'c'}];
trialStartNums = [0 120 240];

timeStartFromBaseLine = -1.25;

for i=1:length(protocolNameList)
    protocolName = protocolNameList{i};

    folderNameLong = fullfile(folderSourceString,'data','segmentedDataLong',subjectName,gridType,expDate,protocolName);
    folderExtractLong = fullfile(folderNameLong,'extractedData');
    folderSegmentLong = fullfile(folderNameLong,'segmentedData');
     
    % Get goodStimTimes from extractedData
    tmp = load(fullfile(folderExtractLong, 'goodStimTimes.mat'), 'goodStimTimes');
    goodStimTimesFull = tmp.goodStimTimes;

    % timeVals
    tmp = load(fullfile(folderSegmentLong,'LFP','lfpInfo.mat'));
    timeValsFull = tmp.timeVals;

    % Get signal from segmentedData
    for j = 1:length(channelNumbers)
        tmp = load(fullfile(folderSegmentLong,'LFP',['elec' num2str(channelNumbers(j)) '.mat']));
        analogDataFull = tmp.analogData;
        analogInfo = tmp.analogInfo;

        for k = 1:length(trialStartNums)

            baseTime = goodStimTimesFull(trialStartNums(k)+1); % The time series will be normalized w.r.t to this time
            if k < length(trialStartNums)
                timesToSegment = [baseTime+timeStartFromBaseLine goodStimTimesFull(trialStartNums(k+1))-timeStartFromBaseLine];
                goodStimTimes = goodStimTimesFull(trialStartNums(k)+1 : trialStartNums(k+1)) - baseTime;
            else
                timesToSegment = [baseTime+timeStartFromBaseLine goodStimTimesFull(end)-timeStartFromBaseLine];
                goodStimTimes = goodStimTimesFull(trialStartNums(k)+1 : length(goodStimTimesFull)) - baseTime;
            end

            goodTimePos = intersect(find(timeValsFull>=timesToSegment(1)),find(timeValsFull<timesToSegment(2)));
            
            timeVals = timeValsFull(goodTimePos) - baseTime;
            analogData = analogDataFull(goodTimePos);
            
            folderNameTMP = fullfile(folderSourceString,'data','segmentedDataLong',subjectName,gridType,expDate,[protocolName segmentNames{k}]);
            folderExtractTMP = fullfile(folderNameTMP,'extractedData');
            folderSegmentTMP = fullfile(folderNameTMP,'segmentedData','LFP');
            makeDirectory(folderExtractTMP);
            makeDirectory(folderSegmentTMP);
            save(fullfile(folderExtractTMP,'goodStimTimes.mat'),'goodStimTimes','baseTime');
            save(fullfile(folderSegmentTMP,['elec' num2str(channelNumbers(j)) '.mat']),'analogData','analogInfo');
            save(fullfile(folderSegmentTMP,'lfpInfo.mat'),'timeVals');
        end
    end
end