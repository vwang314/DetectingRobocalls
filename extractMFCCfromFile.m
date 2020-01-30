%% Code to extract mfcc
myFolder='C:\Users\Vincent\Desktop\College\ORS\DS_10283_3336\LA\ASVspoof2019_LA_eval\flac';
filePattern = fullfile(myFolder, '*.flac'); % Change to whatever pattern you need.
theFiles = dir(filePattern);
for k = 1 : length(theFiles)
  baseFileName = theFiles(k).name;
  fullFileName = fullfile(myFolder, baseFileName);
  fprintf(1, 'Now reading %s\n', baseFileName);
  [audioIn,fs] = audioread(fullFileName);
  [mm,aspc] = melfcc(audioIn*3.3752, fs, 'maxfreq', 8000, 'numcep', 13, 'nbands', 40, 'fbtype', 'fcmel', 'dcttype', 1, 'usecmp', 1, 'wintime', 0.025, 'hoptime', 0.01, 'preemph', 0, 'dither', 1);
  saveName = strcat('C:\Users\Vincent\Desktop\College\ORS\DS_10283_3336\LA\ASVspoof2019_LA_eval\MFCC\',baseFileName(1:end-5));
  save(saveName, 'mm');
end