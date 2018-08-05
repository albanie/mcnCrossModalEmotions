function [miniImdb,expDirs,valIdxSets] = run_cross_val(varargin)

  opts.blur = false ;
  opts.clobberImdb = true ;
  opts.numEpochs = 100 ;
  opts.manualEpoch = 0 ;
  opts.vis = false ;
  %opts.modelName = 'senet50_ft-dag-distributions-CNTK-dropout-0.5-aug' ;
  %opts.labelType = 'labelsFerPlus' ;
  %opts.modelName = 'senet50_ft-dag-distributions-CNTK-dropout-0.5-aug' ;
  opts.refreshCkpts = false ;
  opts.modality = 'visual' ;
  opts.modelName = 'senet50_ft-dag-distributions-dropout-0.5-aug' ;
  opts.heuristic = false ;
  opts.dropContempt = false ;
  opts.official = false ;
  opts.testMode = 0 ;
  opts.affBias = true ;
  opts.evaluationMethod = 'train-val' ; % used in submission (use x-val for final)
	%opts.aggregator = 'mean1' ; % combination method for preds in track
  opts.aggregator = 'max' ; % combination method for preds in track
	%opts.aggregator = 'peak' ; % combination method for preds in track
  opts.scratch = false ;
  % best so far with a linear weighting (approx 0.79 error)
  %opts.preproc = 'shiftAndScale' ;
  opts.preprocessing = 'none' ;
  opts.audioModelDir = '' ;
  opts.softmax = false ; % not helpful
  %opts.preprocessing = 'shiftAndScale' ; % cov matrix is fairly reasonable already
  opts.targetDataset = 'rml' ;
  opts.numSrcEmotions = 8 ;
  opts = vl_argparse(opts, varargin) ;

  switch opts.targetDataset
    case 'rml'
      numTargetEmotions = 7 ; % keep one extra (useless) dimension
      opts.dataDir = fullfile(vl_rootnn, 'data/datasets/rml') ;
    case {'afew', 'afew-6'}
      numTargetEmotions = 7 ; % no contempt (which would appear last)
      opts.dataDir = fullfile(vl_rootnn, 'data/datasets/emotiw2016') ;
    case 'enterface'
      numTargetEmotions = 7 ; % keep one extra (useless) dimension
      opts.dataDir = fullfile(vl_rootnn, 'data/datasets/enterface') ;
    otherwise, error('unknown dataset %s\n', opts.targetDataset) ;
  end

  switch opts.modality
    case 'visual'
      switch opts.modelName
        case 'vgg-vd-face-fer'
          opts.labelType = 'labels' ;
        case 'senet50_ft-dag-distributions-dropout-0.5-aug'
          opts.labelType = 'labelsFerPlus' ;
        case 'senet50_ft-dag-distributions-CNTK-dropout-0.5-aug'
          opts.labelType = 'labelsFerPlus' ;
        otherwise, error('unknown model %s\n', opts.modelName) ;
      end
      featDir = fullfile(vl_rootnn, ...
          sprintf('data/xEmo18/%s_storedFeats', opts.targetDataset)) ;
      imdbPath = fullfile(featDir, ...
         sprintf('%s-logits-blur-%d.mat', opts.modelName, opts.blur)) ;
			expRoot = fullfile(vl_rootnn, ...
									 sprintf('data/%s-strategies', opts.targetDataset)) ;
    case 'audio'
      opts.labelType = 'labelsFerPlus' ;
      imdbPath = getLocalImdbPath(opts.modelName, opts.audioModelDir, ...
                                  opts.targetDataset, opts) ;
			expRoot = fullfile(vl_rootnn, ...
   		     sprintf('data/%s-audio-strategies', opts.targetDataset)) ;

      waitingFor = 0 ;
      while ~exist(imdbPath, 'file')
        fprintf('waiting for %s...(%d)\n', imdbPath, waitingFor) ;
        pause(5) ; waitingFor = waitingFor + 1 ;
        imdbPath = getLocalImdbPath(opts.modelName, opts.audioModelDir, ...
                                    opts.targetDataset, opts) ;
      end

    otherwise, error('unknown modality %s\n', opts.modality) ;
  end

  if exist(imdbPath, 'file') || opts.clobberImdb
    fprintf('loading imdb from memory...') ; tic ;
    try
			imdb = load(imdbPath) ;
    catch
      msg = 'checkopint at %s was malformed, trying agin in 10 secs....\n' ;
      warning(msg, imdbPath) ;
      pause(10) ;
      imdb = load(imdbPath) ;
    end
    fprintf('done in %g s\n', toc) ;

    % fix broken imdb if needed
    if numel(imdb.tracks.set) == 1156 % old ver
      drop = [500, 554, 579, 776, 943] ;
      imdb.tracks.set(drop) = [] ;
      imdb.tracks.id(drop) = [] ;
    end
  end

  if opts.dropContempt || strcmp(opts.modelName, 'vgg-vd-face-fer')
    opts.numSrcEmotions = 7 ;
  end

  % look at some examples
  ferPlusEmotions = {'neutral', 'happiness', 'surprise', 'sadness', ...
                    'anger', 'disgust', 'fear', 'contempt'} ;
  %ii = 1 ;
  if opts.vis
    for ii = 1:10
      args = {'Pack', 'Resize', [100 100]} ;
      ims = vl_imreadjpeg(fullfile(opts.dataDir, imdb.tracks.paths{ii}), args{:}) ;
      label = imdb.tracks.(opts.labelType)(ii) ;
      subplot(1,2,1) ;
      vl_imarraysc(ims{1}) ;
      emoLabel = ferPlusEmotions{label} ;
      title(emoLabel) ;
      subplot(1,2,2) ;
      imagesc(imdb.faceLogits{ii}) ;
      set(gca,'xtick',1:8,'xticklabel', ferPlusEmotions) ;
      xtickangle(45) ; zs_dispFig ;
    end
  end

  switch opts.evaluationMethod
    case 'train-val'
      numFolds = 1 ;
      trainIdx = find(imdb.tracks.set == 1) ;
      valIdx = find(imdb.tracks.set == 2) ;
      valIdxSets{1} = valIdx ;
      trainIdxSets{1} = trainIdx ;
      adjFactor = numel(valIdx) / (numel(valIdx) + 2) ; % only relevant for main split
    case 'cross-val'
      assert(~opts.testMode, 'test mode should not be used during x-validation') ;
      rng(0) ; % for repeatability
      numFolds = 10 ; % use 10 folds as standard
      numSamples = numel(imdb.tracks.set) ;
      sampleOrder = randperm(numSamples) ;

      splits = round(linspace(0, numSamples, numFolds + 1)) ;
      %splitSize = round(numSamples/numFolds) ;
      %splits = 1:splitSize:numSamples ;

      %lastSplitSize = numSamples - splits(end) ;
      %if lastSplitSize < (splitSize / 2)
        %splits(end) = numSamples ; % ensure that last split contains remaining samples
      %else
        %splits(end+1) = numSamples ; % add into additional split
      %end
      trainIdxSets = cell(1, numFolds) ;
      valIdxSets = cell(1, numFolds) ;
      for ii = 1:numFolds
        valIdx = sampleOrder(splits(ii)+1:splits(ii+1)) ;
        trainIdx = sampleOrder(~ismember(sampleOrder, valIdx)) ;
        valIdxSets{ii} = valIdx ;
        trainIdxSets{ii} = trainIdx ;
      end
      adjFactor = 1 ; % not used for x-val
    otherwise, error('%s unrecognised\n', opts.evaluationMethod) ;
  end

  if opts.heuristic
		valLogits = imdb.faceLogits(valIdx) ;
		valLabels = imdb.tracks.(opts.labelType)(valIdx) ;

		% try simple stuff first
		avgLogits = cellfun(@(x) {mean(x, 1)}, valLogits) ;
    keyboard
		avgPreds = cellfun(@pickMax, avgLogits) ;

		maxLogits = cellfun(@(x) {max(x, [], 1)}, valLogits) ;
		maxPreds = cellfun(@pickMax, maxLogits) ;

		accs = evalPreds(avgPreds, valLabels, opts.numSrcEmotions, 'avg') ;
		accs = evalPreds(maxPreds, valLabels, opts.numSrcEmotions, 'max') ;
  end

  expDirs = cell(1, numFolds) ;
  for foldNum = 1:numFolds
    fprintf('finetuning with fold %d/%d\n', foldNum, numFolds) ;

    trainIdx = trainIdxSets{foldNum} ;
    valIdx = valIdxSets{foldNum} ;

    faceLogits = preprocess(imdb.faceLogits, trainIdx, opts) ;

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

    % define inputs
    x = Input() ; y = Input() ;
    prediction = vl_nnconv(x, ...
                 'size', [1, 1, opts.numSrcEmotions, numTargetEmotions], ...
                 'hasBias', opts.affBias) ;

    % define loss, and classification error
    loss = vl_nnloss(prediction, y) ;
    errorVar = vl_nnloss(prediction, y, 'loss','classerror') ;
    Layer.workspaceNames() ;
    errorVar.name = 'error' ; % for consistency
    net = Net(loss, errorVar) ;

    % generate mini imdb
    miniImdb.fusedLogits = fusedLogits ;
    miniImdb.labels = labels ;
    miniImdb.images.set = imdb.tracks.set ;
    opts.batchOpts = struct() ;
    opts.train = struct() ;
    opts.train.gpus = [] ;
    opts.train.continue = ~opts.refreshCkpts ;
    opts.train.batchSize = 10 ;
    opts.train.numEpochs = opts.numEpochs ;
    opts.train.learningRate = 0.001 ;


    % train the tiny network
    expName = sprintf('%s-%s', opts.modelName, opts.aggregator) ;
    if strcmp(opts.evaluationMethod, 'cross-val')
      expName = sprintf('%s-foldNum-%d', expName, foldNum) ;
    end
    if ~strcmp(opts.preprocessing, 'none')
      expName = [expName '-' opts.preprocessing] ;
    end
    if opts.softmax
      expName = [expName '-softmax'] ;
    end
    if ~opts.affBias
      expName = [expName '-no-bias'] ;
    end

    if opts.official
      miniImdb.images.set(miniImdb.images.set == 2) = 3 ;
      ratio = 0.8 ;
      trainIdx = find(miniImdb.images.set == 1) ;
      keep = randsample(numel(trainIdx), round(numel(trainIdx)*ratio)) ;
      miniImdb.images.set(miniImdb.images.set == 1) = 2 ;
      miniImdb.images.set(keep) = 1 ;
      expName = [expName '-official'] ;
    end
    if opts.manualEpoch
      expName = [expName sprintf('-sel-epoch%d', opts.manualEpoch)] ;
    end

    expDir = fullfile(expRoot, expName) ;
    [net,info] = cnn_train_autonn(net, miniImdb, ...
                      @(i,b) get_batch(i, b, opts), ...
                      opts.train, 'expDir', expDir, ...
                      'train', trainIdx, 'val', valIdx) ;
    valAcc = 100 * (1 - min([info.val.error])) ;
    % adjust to compensate for missing faces (only used in original train-val split)
    valAcc = valAcc * adjFactor ;
    fprintf('%s: best val acc %.2f\n', expName, valAcc) ;

    if opts.testMode % run on test set
      [~,bestEpoch] = min([info.val.error]) ;

      remove = find(miniImdb.images.set == 2 | miniImdb.images.set == 1) ;
      miniImdb.images.set(remove) = 4 ;
      miniImdb.images.set(miniImdb.images.set == 3) = 2 ;
      miniImdb.images.set(remove(1)) = 1 ; % ensure weights are saved
      opts.train.learningRate = 0 ;
      opts.train.continue = 0 ;
      opts.train.numEpochs = 1 ;
      ckptPath = buildCkptPath(expDir, bestEpoch) ;
      tmp = load(ckptPath) ;
      net = Net(tmp.net) ;
      expDir = [expDir 'test'] ;

      [net,info] = cnn_train_autonn(net, miniImdb, ...
                        @(i,b) get_batch(i, b, opts), ...
                        opts.train, 'expDir', expDir, ...
                        'train', trainIdx, 'val', valIdx) ;
    end
    expDirs{foldNum} = expDir ;
  end
end

% -------------------------------------------------------------
function logit = selectPeakLogit(logits)
% -------------------------------------------------------------
% select logit by strongest spike
	[spike, spikeLoc] = max(logits(:)) ;
	[r,c] = ind2sub(size(logits), spikeLoc) ;
  logit = logits(r,:) ;
end

% -------------------------------------------------------------
function batchData = get_batch(imdb, batch, opts)
% -------------------------------------------------------------
  feats = reshape(imdb.fusedLogits(batch,:)', 1, 1, [], numel(batch)) ;
  labels = imdb.labels(batch) ;
  batchData = {'x', feats, 'y', labels} ;
end

% -------------------------------------------------------------
function faceLogits = preprocess(faceLogits, trainIdx, opts)
% -------------------------------------------------------------

  switch opts.preprocessing
    case 'none' % do nothing
    case 'shiftAndScale'
			trainLogits = faceLogits(trainIdx) ;
			logitMatrix = vertcat(trainLogits{:}) ;
			mu = mean(logitMatrix) ;
			sigma = std(logitMatrix) ;
			faceLogits = cellfun(@(x) {(x - mu) ./ sigma}, faceLogits) ;
    case 'whiten'
      transform = whiten(faceLogits, trainIdx, 1E-12) ;
      faceLogits = cellfun(@(x) {x * transform}, faceLogits) ;
			%whitenedLogits = allLogits*V*diag(1./(diag(D)+eps).^(1/2))*V';
    otherwise, error('unrecognised preprocessing %s\n', opts.preprocessing) ;
  end

  if opts.softmax
    faceLogits = cellfun(@(x) {vl_nnsoftmaxt(x, 'dim', 2)}, faceLogits) ;
  end
end
% ---------------------------------------------------------------------
function imdbPath = getLocalImdbPath(modelName, audioModelDir, dataset, opts)
% ---------------------------------------------------------------------
	if ~strcmp(modelName, 'random')
		[~,epoch] = audio_zoo(audioModelDir, 0, ...
                          'manualEpoch', opts.manualEpoch) ;
	else
		epoch = 0 ;
	end
	imdbPath = getAudioFeaturePath(dataset, modelName, epoch) ;
end

% -------------------------------------------------------------------------
function transform = whiten(faceLogits, trainIdx, eps)
% -------------------------------------------------------------------------
	trainLogits = faceLogits(trainIdx) ;
	X = vertcat(trainLogits{:}) ;
	X = bsxfun(@minus, X, mean(X)) ; % centering
	A = X'*X ; [V,D] = eig(A) ;
  transform = V*diag(1./(diag(D)+eps).^(1/2))*V';
end


% -------------------------------------------------------------------------
function [acc, accs, macc] = evalPreds(preds, labels, numClasses, method)
% -------------------------------------------------------------------------
	% Compute the confusion matrix
	idx = sub2ind([numClasses, numClasses], labels, preds) ;
	confus = zeros(numClasses, numClasses) ;
	confus = vl_binsum(confus, ones(size(idx)), idx) ;

	% Plots
	clf ;
	imagesc(confus) ;
  acc = 100 * mean(diag(confus)/numel(labels)) ;
	accs = diag(confus) ./ sum(confus, 2) ;
  if numClasses == 8
		accs(numClasses) = 0 ; % set nans to zero for 8th class
  end
  macc = mean(accs) ;
	title(sprintf('%s confusion matrix (%.2f %% accuracy)', method, acc)) ;
  zs_dispFig ;
end

% -------------------------------------------------------------------------
function y = pickMax(x)
% -------------------------------------------------------------------------
  [~,y] = max(x) ;
end

% ------------------------------------------------------------------
function ckptPath = buildCkptPath(expDir, best)
% ------------------------------------------------------------------
	ckptPath = fullfile(expDir, sprintf('net-epoch-%d.mat', best)) ;
end
