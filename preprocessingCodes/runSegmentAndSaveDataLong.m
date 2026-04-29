% runSegmentAndSaveDataLong for the BK1 project - segment physiology
% channels into longer segments spanning all markers

gridType = 'EEG';
capType = 'actiCap64_2019';
displayFlag = 0;
FsEye = 1000;
folderSourceString = 'D:\OneDrive - Indian Institute of Science\Supratim\Projects\ProjectDhyaan\BK1';
[subjectNameList,expDateList] = getDemographicDetails('BK1'); % All subjects

% get all good subjects
goodSubjectList = getGoodSubjectsBK1;

% problamaticSubjects  = [{'095KM'} {'053DR'} {'010PB'}]; % To be extracted separately
problamaticSubjects  = {'053DR'};
goodSubjectList = setdiff(goodSubjectList,problamaticSubjects,'stable');

% Channels
channelNumbers = [66 69];
for i=1:length(goodSubjectList)
    subjectName = goodSubjectList{i};
    disp(['Extracting Subject ' subjectName]);
    %-------------------------------------------------------------------------------------------------------------------------------
    expDate = expDateList{strcmp(subjectName,subjectNameList)};
%    segmentAndSaveDataLong(subjectName,expDate,folderSourceString,channelNumbers); % Segment data
    segmentAndSaveMeditationDataLong(subjectName,expDate,folderSourceString,channelNumbers); % Segment meditation data further
end

%-----------------------------------------------------------------------------------------------------
% Issues during the data segment process:
% problamaticIndices  = [5 12 75];
% goodSegmentIndices  = setdiff(segmentTheseIndices,problamaticIndices);
% ----------------------------------------------------------------------------------------------------
% Marker Issues:-
% i=5;  095KM, M2 protocol, .vmrk file had junk markers; for extraction we have used the markers which matches with the .bhv2 file;
% i=75; 099SP, M2 protocol has less trial start markers compared to .bhv2 file. Saved data for only the trials which have markers matched between
%       .vmrk and .bhv2 file. 2 trials were lost at the beggining of the experiment
% ----------------------------------------------------------------------------------------------------
% File Missing:-
% i=12; 053DR, EO1 protocol could not be extracted as .bhv2 file is missing;
%-----------------------------------------------------------------------------------------------------
% Time issues:- (could be extracted with the updated version of the segmentation code)
% Fixed with additional condition while processing the eye data in the segmentation codes
% i=15; 035SS, EC2
% i=26; 006SR, EO1 protocol
% i=35; 089AB, EO1 and EC1 protocol
% i=50; 064PK, EO1 and EC1 protocol
%-----------------------------------------------------------------------------------------------------