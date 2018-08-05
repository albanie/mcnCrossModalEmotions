function [miniImdb,expDirs,valIdxSets] = run_cross_val(varargin)
%RUN_CROSS_VAL evaluate model using cross validation
%   [MINIIMDB, EXPDIRS, VALIDX] = RUN_CROSS_VAL(VARARGIN) runs
%   K-fold evaluation of a model with the options specified in
%   VARARGIN. It returns MINIIMDB, a structure containing the
%   predictions of the model and the corresponding ground truth
%   labels and EXPDIRS, a cell array of paths to directories where
%   the parameters fitted by cross validation are stored. The final
%   return value, VALIDXSETS, is a cell array of the indicies that
%   correspond to the validation set for each fold.
%
%   RUN_CROSS_VAL(..'name', value) accepts the following
%   options:
%
%   `numFolds` :: 10
%    The number of folds to use in the cross validation procedure.
%
%   `aggregator` :: 'max'
%    For visual models (that compute predictions on each frame separately,
%    rather than on a full track, this option specifies how the predictions
%    will be aggregated to make a single track prediction.
%
%   `targetDataset` :: 'rml'
%    The name of the dataset used for the evaluation.
%
%   `numTargetEmotions` :: 6
%    The number of emotions to be predicted for the target dataset.
%
%   `numSrcEmotions` :: 8
%    The number of emotions predicted by the model (this may differ from
%    the number of emotions that must be predicted for the target dataset).
%
%   `useExstingVal` :: false
%    If true, this option runs a single training run on an existing
%    train/validation split, rather than performing cross validation. If
%    used, then the `numFolds` option must also be set to 1.
%
%   `modelName` :: 'emovoxceleb-student'
%    The name of the emotion recognition model to be evaluated.
%
% Copyright (C) 2018 Samuel Albanie, Arsha Nagrani
% Licensed under The MIT License [see LICENSE.md for details]

  opts.numFolds = 10 ;
  opts.aggregator = 'max' ; % combination method for preds in track
  opts.targetDataset = 'rml' ;
  opts.numTargetEmotions = 6 ;
  opts.numSrcEmotions = 8 ;
  opts.labelType = 'labels' ;
  opts.useExstingVal = false ;
  opts.modality = 'visual' ;
  opts.modelName = 'emovoxceleb-student' ;
  opts = vl_argparse(opts, varargin) ;

  rng(0) ; % ensure repeatability

  switch opts.targetDataset
    case 'rml'
      opts.dataDir = fullfile(vl_rootnn, 'data/datasets/rml') ;
    case {'afew', 'afew-6'}
      opts.dataDir = fullfile(vl_rootnn, 'data/datasets/emotiw2016') ;
    case 'enterface'
      opts.dataDir = fullfile(vl_rootnn, 'data/datasets/enterface') ;
    otherwise, error('unknown dataset %s\n', opts.targetDataset) ;
  end

  % to fit a classifier layer, we first precompute features on the target
  % dataset with the main network.
  switch opts.modality
    case 'visual'
      featConstructor = @compute_visual_feats ;
    case 'audio'
      featConstructor = @compute_audio_feats ;
    otherwise, error('unknown modality %s\n', opts.modality) ;
  end
  imdbDir = fullfile(vl_rootnn, 'data/mcnCrossModalEmotions', ...
                              sprintf('cachedFeats-%s', opts.modality)) ;
  expRoot = fullfile(vl_rootnn, ...
             sprintf('data/%s-%s', opts.targetDataset, opts.modality)) ;
  if ~exist(imdbDir, 'dir'), mkdir(imdbDir) ; end
  imdbPath = fullfile(imdbDir, ...
       sprintf('%s-%s-feats.mat', opts.modelName, opts.targetDataset)) ;
  if ~exist(imdbPath, 'file')
    featConstructor(imdbPath, 'modelName', opts.modelName, ...
                              'targetDataset', opts.targetDataset) ;
  end

  fprintf('loading imdb from memory...') ; tic ;
  imdb = load(imdbPath) ;
  fprintf('done in %g s\n', toc) ;

  if opts.useExstingVal
    msg = 'when using an existing val set, only one fold should be specified' ;
    assert(opts.numFolds == 1, msg) ;
    trainIdxSets{1} = find(imdb.tracks.set == 1) ;
    valIdxSets{1} = find(imdb.tracks.set == 2) ;
  else
    numSamples = numel(imdb.tracks.set) ;
    sampleOrder = randperm(numSamples) ;
    splits = round(linspace(0, numSamples, opts.numFolds + 1)) ;
    trainIdxSets = cell(1, opts.numFolds) ;
    valIdxSets = cell(1, opts.numFolds) ;
    for ii = 1:opts.numFolds
      valIdx = sampleOrder(splits(ii)+1:splits(ii+1)) ;
      trainIdx = sampleOrder(~ismember(sampleOrder, valIdx)) ;
      valIdxSets{ii} = valIdx ;
      trainIdxSets{ii} = trainIdx ;
    end
  end

  expDirs = cell(1, opts.numFolds) ;
  for foldNum = 1:opts.numFolds
    fprintf('finetuning with fold %d/%d\n', foldNum, opts.numFolds) ;
    trainIdx = trainIdxSets{foldNum} ;
    faceLogits = imdb.faceLogits ;

    % train the tiny classifier
    expName = sprintf('%s-%s-foldNum-%d', opts.modelName, ...
                                      opts.aggregator, foldNum) ;
    expDir = fullfile(expRoot, expName) ;
    if ~exist(expDir, 'dir'), mkdir(expDir) ; end
    expDirs{foldNum} = expDir ;

    switch opts.aggregator
      case 'mean1', aggregator = @(x) mean(x, 1) ;
      case 'max', aggregator = @(x) max(x, [], 1) ;
      case 'peak', aggregator = @(x) selectPeakLogit(x) ;
      otherwise, error('aggregator: %s unrecognised\n', opts.aggregator) ;
    end

    fusedLogits = cellfun(aggregator, faceLogits, 'uni', 0) ;
    fusedLogits = vertcat(fusedLogits{:}) ; % form matrix
    labels = imdb.tracks.(opts.labelType) ;

    % generate mini imdb
    miniImdb.labels = labels ;
    miniImdb.fusedLogits = fusedLogits ;
    miniImdb.images.set = imdb.tracks.set ;

    trainLogits = double(fusedLogits(trainIdx(:),:)) ;
    trainLabels = double(labels(trainIdx(:)))' ;
    coefficients = mnrfit(trainLogits, trainLabels) ; %#ok
    paramPath = fullfile(expDir, 'mnr-params.mat') ;
    save(paramPath, 'coefficients') ;
  end
end

% -------------------------------------------------------------
function logit = selectPeakLogit(logits)
% -------------------------------------------------------------
% select logit by strongest spike
	[~, spikeLoc] = max(logits(:)) ;
	[r,~] = ind2sub(size(logits), spikeLoc) ;
  logit = logits(r,:) ;
end
