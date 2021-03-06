%% Variables
nDims = 13;             % dimensionality of feature vectors
nMixtures = 32;         % How many mixtures used to generate data
nChannels = 99999999;   % Number of channels (sessions) per speaker,need to be >10, will be changed later to accurate amount
nWorkers = 1;           % Number of parfor workers, if available

%% Load mfcc
fileID = fopen('C:\Users\Vincent\Desktop\College\ORS\DS_10283_3336\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.train.trn.txt');
x = textscan(fileID, 'LA_00%u %s - - bonafide'); %format to read textfile for only bonafide audio no spoof
fclose(fileID);                                  %spoof format:  'LA_00%u %s - %s %s'
speakerLA_ID = x{1,1};                           %all speaker id in original order extract from txt file (78, 79, 80, ... 98)
z = unique(speakerLA_ID);                        %array of unique speaker id sorted (1,2,3, .. 20)
nSpeakers = length(z);                           %num speakers = num of distinct speaker id
filename = x{1,2};                               %all file name extract from txt file
count = ones(1,length(z));                       %array used as counter
for i=1:length(z)                                %loop to find the speaker with least audio files
    id = z(i);
    k = length(find(speakerLA_ID==id));
    a = [nChannels k];
    nChannels = min(a);
end
trainSpeakerData = cell(nSpeakers,nChannels);   %initialize trainSpeakerData
for a=1:length(filename)                        %loop to import mfcc for each file name in txt file
    c = find(z==speakerLA_ID(a));               %get index corresponding to LA_00xx id (convert LA_00xx id to 1,2,3,.. id)
    f = matfile(char(filename(a)));             %load mfcc .mat file
    if (count(c)<nChannels+1)                   %make sure to import the same amount of files as speaker with least files
        trainSpeakerData{c, count(c)} = f.mm;   %load mfcc into row corresponding to speaker
        count(c) = count(c)+1;
    end
end

speakerID=(1:nSpeakers)'*ones(1,nChannels);     %matrix of same dimension as trainSpeakerData with each cell containing speakerID of the corresponding cell's mfcc
testSpeakerData = trainSpeakerData;

%% Step1: Create the universal background model from all the training speaker data
nmix = nMixtures;           % In this case, we know the # of mixtures needed 32, then 512
final_niter = 10;
ds_factor = 1;
nWorkers = 1;
ubm = gmm_em(trainSpeakerData(:), nmix, final_niter, ds_factor, nWorkers);

%%
% Step2.1: Calculate the statistics needed for the iVector model.
stats = cell(nSpeakers, nChannels);
for s=1:nSpeakers
    for c=1:nChannels
        [N,F] = compute_bw_stats(trainSpeakerData{s,c}, ubm);
        stats{s,c} = [N; F];
    end
end

% Step2.2: Learn the total variability subspace from all the speaker data.
tvDim = 100;
niter = 5;
T = train_tv_space(stats(:), ubm, tvDim, niter, nWorkers);
%
% Now compute the ivectors for each speaker and channel.  The result is size
%   tvDim x nSpeakers x nChannels
devIVs = zeros(tvDim, nSpeakers, nChannels);
for s=1:nSpeakers
    for c=1:nChannels
        devIVs(:, s, c) = extract_ivector(stats{s, c}, ubm, T);
    end
end

%%
% Step3.1: Now do LDA on the iVectors to find the dimensions that matter.
ldaDim = min(100, nSpeakers-1);
devIVbySpeaker = reshape(devIVs, tvDim, nSpeakers*nChannels);
[V,D] = lda(devIVbySpeaker, speakerID(:));
finalDevIVs = V(:, 1:ldaDim)' * devIVbySpeaker;

% Step3.2: Now train a Gaussian PLDA model with development i-vectors
nphi = ldaDim;                  % should be <= ldaDim
niter = 10;
pLDA = gplda_em(finalDevIVs, speakerID(:), nphi, niter);

%%
% Step4.1: OK now we have the channel and LDA models. Let's build actual speaker
% models. Normally we do that with new enrollment data, but now we'll just
% reuse the development set.
averageIVs = mean(devIVs, 3);           % Average IVs across channels.
modelIVs = V(:, 1:ldaDim)' * averageIVs;


% Step4.2: Now compute the ivectors for the test set 
% and score the utterances against the models
testIVs = zeros(tvDim, nSpeakers, nChannels); 
for s=1:nSpeakers
    for c=1:nChannels
        [N, F] = compute_bw_stats(testSpeakerData{s, c}, ubm);
        testIVs(:, s, c) = extract_ivector([N; F], ubm, T);
    end
end
testIVbySpeaker = reshape(permute(testIVs, [1 3 2]), ...
                            tvDim, nSpeakers*nChannels);
finalTestIVs = V(:, 1:ldaDim)' * testIVbySpeaker;

%%
% Step5: Now score the models with all the test data.
ivScores = score_gplda_trials(pLDA, modelIVs, finalTestIVs);
imagesc(ivScores)
title('Speaker Verification Likelihood (iVector Model)');
xlabel('Test # (Channel x Speaker)'); ylabel('Model #');
colorbar; axis xy; drawnow;

answers = zeros(nSpeakers*nChannels*nSpeakers, 1);
for ix = 1 : nSpeakers,
    b = (ix-1)*nSpeakers*nChannels + 1;
    answers((ix-1)*nChannels+b : (ix-1)*nChannels+b+nChannels-1) = 1;
end

ivScores = reshape(ivScores', nSpeakers*nChannels* nSpeakers, 1);
figure;
eer = compute_eer(ivScores, answers, true);