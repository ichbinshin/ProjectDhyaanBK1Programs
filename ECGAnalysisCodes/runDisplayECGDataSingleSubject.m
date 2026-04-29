clear; close all

[allSubjectNames,expDateList] =getDemographicDetails('BK1');
goodSubjectList = getGoodSubjectsBK1;

folderSourceString = 'D:\OneDrive - Indian Institute of Science\Supratim\Projects\ProjectDhyaan\BK1';
saveFolderName = 'ECGResultsSingleSubject';

saveFileFlag = 1;
useTheseIndices = 1:length(goodSubjectList);

channelNumber = 66;
medSegmentedFlag = 0;

for i=1:length(useTheseIndices)
    % fh=figure(1); clf(fh);
    % fh.WindowState = 'maximized';
    clf;
    subjectName = goodSubjectList{useTheseIndices(i)};
    disp(['Analyzing for the subject ' subjectName]);
    expDate = expDateList{strcmp(subjectName,allSubjectNames)};
    displayECGDataSingleSubject(subjectName,expDate,folderSourceString,channelNumber,medSegmentedFlag);
    pause;

    % if saveFileFlag
    %     makeDirectory(saveFolderName);
    %     fileNameTif = fullfile(saveFolderName,[subjectName badTrialNameStr '_badElecChoice' num2str(badElectrodeRejectionFlag) '_raw' num2str(plotRawTFFlag) '_sort' num2str(sortByBadTrialFlag) '.tif']);
    %     print(fh,fileNameTif,'-dtiff','-r300');
    % end
end