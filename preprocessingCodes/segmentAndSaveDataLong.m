% Unlike segmentAndSaveData, in which a 2.5 second segment is created
% around each marker, here a single long segment spanning all markers is
% saved for selected channels.

function segmentAndSaveDataLong(subjectName,expDate,folderSourceString,channelNumbers)

if ~exist('folderSourceString','var');    folderSourceString=[];        end

if isempty(folderSourceString)
    folderSourceString = 'D:\OneDrive - Indian Institute of Science\Supratim\Projects\ProjectDhyaan\BK1';
end

gridType = 'EEG';
protocolNameList = [{'EO1'} {'EC1'} {'G1'} {'M1'} {'G2'} {'EO2'} {'EC2'} {'M2'}];

timeStartFromBaseLine = -1.25;

for i=1:length(protocolNameList)
    protocolName = protocolNameList{i};

    % Get Digital events from BrainProducts (BP)
    folderName = fullfile(folderSourceString,'data','segmentedData',subjectName,gridType,expDate,protocolName);
    folderExtract = fullfile(folderName,'extractedData');
    
    % Load goodStimTimes folderExtract
    load(fullfile(folderExtract,'goodStimCodeNums.mat'),'goodStimTimes');
    
    % Save long segmented data
    goodStimTime = goodStimTimes(1);
    deltaT = goodStimTimes(end) - goodStimTime + 2*abs(timeStartFromBaseLine);
    getEEGDataBrainProductsLong(subjectName,expDate,protocolName,folderSourceString,gridType,goodStimTime,timeStartFromBaseLine,deltaT,channelNumbers);

    % Save goodStimTimes in extractedData
    folderNameLong = fullfile(folderSourceString,'data','segmentedDataLong',subjectName,gridType,expDate,protocolName);
    folderExtractLong = fullfile(folderNameLong,'extractedData');
    makeDirectory(folderExtractLong);

    goodStimTimes = goodStimTimes - goodStimTime; % All times relative to the first one
    save(fullfile(folderExtractLong, 'goodStimTimes.mat'), 'goodStimTimes');
    
end