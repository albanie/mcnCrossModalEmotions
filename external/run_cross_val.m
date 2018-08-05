function [miniImdb,expDirs,valIdxSets] = run_cross_val(varargin)

  opts.numFolds = 10 ; % use 10 folds as standard
  opts.affBias = true ;
  opts.numEpochs = 100 ;
  opts.aggregator = 'max' ; % combination method for preds in track
  opts.numSrcEmotions = 8 ;
  opts.targetDataset = 'rml' ;
  opts.labelType = 'labels' ;
  opts.numTargetEmotions = 6 ;
  opts.modality = 'visual' ;
  opts.refreshCkpts = false ;
  opts.mnrfit = false ;
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
  % dataset with the main network (this speeds things up a bit). They are
  % appended to a copy of the imdb used for the target dataset.
  switch opts.modality
    case 'visual'
      featDir = fullfile(vl_rootnn, ...
          sprintf('data/xEmo18/%s_storedFeats', opts.targetDataset)) ;
      imdbPath = fullfile(featDir, ...
                     sprintf('%s-logits-.mat', opts.modelName)) ;
			expRoot = fullfile(vl_rootnn, ...
									 sprintf('data/%s-visual', opts.targetDataset)) ;
    case 'audio'
      imdbDir = fullfile(fullfile(vl_rootnn, 'data/mcnCrossModalEmotions', ...
                                  'cachedAudioFeats')) ;
      if ~exist(imdbDir, 'dir'), mkdir(imdbDir) ; end
      imdbPath = fullfile(imdbDir, ...
            sprintf('%s-%s-feats.mat', opts.modelName, opts.targetDataset)) ;
      if ~exist(imdbPath, 'file')
        compute_audio_feats(imdbPath, 'modelName', opts.modelName, ...
                            'targetDataset', opts.targetDataset) ;
      end
			expRoot = fullfile(vl_rootnn, ...
   		                   sprintf('data/%s-audio', opts.targetDataset)) ;
    otherwise, error('unknown modality %s\n', opts.modality) ;
  end

  fprintf('loading imdb from memory...') ; tic ;
  imdb = load(imdbPath) ;
  fprintf('done in %g s\n', toc) ;

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

  expDirs = cell(1, opts.numFolds) ;
  for foldNum = 1:opts.numFolds
    fprintf('finetuning with fold %d/%d\n', foldNum, opts.numFolds) ;

    trainIdx = trainIdxSets{foldNum} ;
    valIdx = valIdxSets{foldNum} ;
    faceLogits = imdb.faceLogits ;

    % train the tiny classifier
    expName = sprintf('%s-%s-foldNum-%d', opts.modelName, ...
                                      opts.aggregator, foldNum) ;
    expDir = fullfile(expRoot, expName) ;
    expDirs{foldNum} = expDir ;

    switch opts.aggregator
      case 'mean1', aggregator = @(x) mean(x, 1) ;
      case 'max', aggregator = @(x) max(x, [], 1) ;
      case 'peak', aggregator = @(x) selectPeakLogit(x) ;
      otherwise, error('aggregator: %s unrecognised\n', opts.aggregator) ;
    end

    fusedLogits = cellfun(aggregator, faceLogits, 'uni', 0) ;
    fusedLogits = vertcat(fusedLogits{:}) ; % form matrix
    fusedLogits = fusedLogits(:, 1:opts.numSrcEmotions) ;
    labels = imdb.tracks.(opts.labelType) ;

    % generate mini imdb
    miniImdb.labels = labels ;
    miniImdb.fusedLogits = fusedLogits ;
    miniImdb.images.set = imdb.tracks.set ;

    if opts.mnrfit
      trainLogits = double(fusedLogits(trainIdx(:),:)) ;
      trainLabels = double(labels(trainIdx(:)))' ;
      coefficients = mnrfit(trainLogits, trainLabels) ; %#ok
      paramPath = fullfile(expDir, 'mnr-params.mat') ;
      save(paramPath, 'coefficients') ;
      continue ; % use as an alternative to SGD
    end

    % define inputs
    x = Input() ; y = Input() ;
    prediction = vl_nnconv(x, ...
             'size', [1, 1, opts.numSrcEmotions, opts.numTargetEmotions], ...
             'hasBias', opts.affBias) ;

    % define loss, and classification error
    loss = vl_nnloss(prediction, y) ;
    errorVar = vl_nnloss(prediction, y, 'loss','classerror') ;
    Layer.workspaceNames() ;
    errorVar.name = 'error' ; % for consistency
    net = Net(loss, errorVar) ;

    opts.train = struct() ;
    opts.train.gpus = [] ;
    opts.train.continue = ~opts.refreshCkpts ;
    opts.train.batchSize = 10 ;
    opts.train.numEpochs = opts.numEpochs ;
    opts.train.learningRate = 0.001 ;
    opts.batchOpts = struct() ;

    [~,info] = cnn_train_autonn(net, miniImdb, ...
                      @(i,b) get_batch(i, b), ...
                      opts.train, 'expDir', expDir, ...
                      'train', trainIdx, 'val', valIdx) ;
    valAcc = 100 * (1 - min([info.val.error])) ;
    fprintf('%s: best val acc %.2f\n', expName, valAcc) ;
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

% -------------------------------------------------------------
function batchData = get_batch(imdb, batch)
% -------------------------------------------------------------
  feats = reshape(imdb.fusedLogits(batch,:)', 1, 1, [], numel(batch)) ;
  labels = imdb.labels(batch) ;
  batchData = {'x', feats, 'y', labels} ;
end
