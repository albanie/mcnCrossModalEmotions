function [net, info] = run_distillation(varargin)

  opts.gpus = 2 ;
  opts.cont = true ;
  opts.miniVal = 0.2 ;
  opts.evaluate = 0 ;
  opts.temperature = 2 ;
  opts.numSeconds = 4 ;
  opts.batchSize = 64 ;
  opts.numEpochs = 300 ; % technically, this the number of "mini" epochs
  opts.numPredEmotions = 8 ;
  opts.fromScratch = true ;
  opts.fixedSegments = false ; % essentially disables time-based augmentation
  opts.logitAggregator = 'max' ;
	opts.datasetName = 'voxceleb' ;
  opts.parameterServer = 'tmove' ;
  %opts.numSampledEmotions = 4 ; % only used for sampling
  %opts.samplerTags = 'maxEmoTags' ;
  opts.teacher = 'senet50-ferplus' ;
  opts.student = 'emovoxceleb-student' ;
  %opts.subsetField = 'intersectSet' ;
  opts.lossType = 'hot-cross-ent' ;
	opts.learningRate = logspace(-4, -5, opts.numEpochs) ;
  opts.wavDir = fullfile(vl_rootnn, 'data/ramdisk/voxceleb_all') ;
  opts = vl_argparse(opts, varargin) ;

  opts.miniEpochRatio = 0.05 * numel(opts.gpus) ; % run full heat

	opts.bn = true ;
	opts.numAugments = 1 ;
	opts.finetune = false ;
	opts.meanType = 'image' ; % 'pixel' | 'image'
	opts.transformation = 'I' ;
	opts.batchNormalization = true ;
	opts.imageSize = [512, opts.numSeconds * 100] ;
	opts.dataDir = fullfile('data', opts.datasetName) ;

  % hack to avoid clearing persistent memory issue
  imdb = fetch_emovoxceleb_imdb(opts.teacher) ;

	student = sprintf('%s-%s', opts.student, opts.lossType) ;
	if opts.fromScratch, student = [student '-scratch'] ; end
  expName = sprintf('%s-%s-%s-%dsec-%demo-agg-%s', ...
                    opts.datasetName, opts.teacher, ...
                    student, opts.numSeconds, ...
                    opts.numPredEmotions, opts.logitAggregator) ;

	opts.expDir = fullfile(vl_rootnn, 'data/xEmo18',  expName) ;
  if strcmp(opts.lossType, 'hot-cross-ent')
    opts.expDir = [opts.expDir sprintf('-temp%d', opts.temperature)] ;
  end
	if ~exist(opts.expDir, 'dir'), mkdir(opts.expDir) ; end


	% Audio settings
	opts.audio.window = [0 1] ;
	opts.audio.fs = 16000 ;
	opts.audio.Tw = 25 ;
	opts.audio.Ts = 10 ; % analysis frame shift (ms)
	opts.audio.alpha = 0.97 ; % preemphasis coefficient
	opts.audio.R = [300 3700] ; % frequency range to consider
	opts.audio.M = 40 ; % number of filterbank channels
	opts.audio.C = 13 ; % number of cepstral coefficients
	opts.audio.L = 22 ; % cepstral sine lifter parameter
	opts.meta.audio = opts.audio ;
	opts.numFetchThreads = 12 ;

% -------------------------------------------------------------------------
%                                                             Prepare model
% -------------------------------------------------------------------------

	net = emoVoxZoo(opts.student, ...
                 'scratch', opts.fromScratch, ...
                 'lossType', opts.lossType, ...
                 'numSeconds', opts.numSeconds, ...
                 'numOutputs', opts.numPredEmotions) ;

	net.meta.augmentation.transformation = 'I' ;
	net.meta.audio = opts.audio ;

% -------------------------------------------------------------------------
%                                                              Prepare data
% -------------------------------------------------------------------------
	imdb.wavDir = opts.wavDir ;
  numEpochs = opts.numEpochs ;
  %imdb.images.(opts.subsetField)(imdb.images.(opts.subsetField) == 4) = 1 ;
  trainSamples = find(imdb.images.set == 1) ;
  %trainSamples  = subsampler(imdb, opts.numSampledEmotions, 1, opts) ;
  valSamples = find(imdb.images.set == 2) ;

  % for efficiency, it can be helpful to only use a portion of the
  % validation set
  if opts.miniVal < 1
    rng(0) ;
    pick = randsample(numel(valSamples), ...
                     round(numel(valSamples) * opts.miniVal)) ;
    valSamples = valSamples(pick) ;
  end

  % print summary
  %fprintf('training/validating using the %s partition\n', opts.subsetField) ;
  %fprintf('experiment stats: (%d emotions, %d secs) \n', ...
                                 %opts.numSampledEmotions, opts.numSeconds) ;
  fprintf('training network with balanced train/val subsets...\n') ;
  fprintf('training samples: %d \n', numel(trainSamples)) ;
  fprintf('val samples: %d \n', numel(valSamples)) ;
  fprintf('GPU: %d\n', opts.gpus) ;

  opts.epochSize = numel(trainSamples) * opts.miniEpochRatio ;
  fprintf('------------------------------------------------------\n') ;
  fprintf('training with epoch subsample size: (%d/%d)\n', ...
                              opts.epochSize, numel(trainSamples)) ;
  fprintf('------------------------------------------------------\n') ;

  % update aug
  meta = net.meta ;
  storeMetaInfo(opts) ;

  trainfn = @cnn_train_dag_check2 ; % handles broken checkpoint issue
	[net, info] = trainfn(net, imdb, getBatchFn(opts, meta), ...
		'learningRate', opts.learningRate, ...
		'batchSize', opts.batchSize,...
		'numEpochs', numEpochs, ...
    'train', trainSamples, ...
    'val', valSamples, ...
    'continue', opts.cont, ...
		'expDir', opts.expDir, ...
		'gpus', opts.gpus, ...
    'epochSize', opts.epochSize, ...
    'parameterServer', struct('method', opts.parameterServer), ...
    'extractStatsFn', @extractStats) ;
end

% ----------------------------------------------------------------------------
%function sampledIdx = subsampler(imdb, numEmotions, subsetIdx, opts)
% ----------------------------------------------------------------------------
%  tags = imdb.(opts.samplerTags)(imdb.images.(opts.subsetField) == subsetIdx) ;
%  counts = histcounts(tags, 0.5:numEmotions+0.5) ;
%  samplesPerClass = round((max(counts) + min(counts)) / 4) ;
%  idx = 1 ;
%  fprintf('------------------------------------------------------\n') ;
%  fprintf('QUARTER: Subsampling to highest emo bin: %d tracks (%s)\n', ...
%          samplesPerClass, imdb.meta.emotions{idx}) ;
%  fprintf('------------------------------------------------------\n') ;
%  sampled = zeros(samplesPerClass, numEmotions) ;
%
%  for ii = 1:numEmotions
%    candidates = find(imdb.(opts.samplerTags) == ii & ...
%                         imdb.images.(opts.subsetField) == subsetIdx) ;
%    fprintf('sampling %d from %d (%s)\n', samplesPerClass, ...
%            numel(candidates), imdb.meta.emotions{ii}) ;
%    if samplesPerClass > numel(candidates)
%      % sample with replacement
%      picks = randsample(numel(candidates), samplesPerClass, true) ;
%    else
%      picks = randsample(numel(candidates), samplesPerClass) ;
%    end
%    sampled(:,ii) = candidates(picks) ;
%  end
%  sampledIdx = sampled(:)' ;
%end

% -------------------------------------------------------------------------
function stats = extractStats(stats, net)
% -------------------------------------------------------------------------
	sel = find(cellfun(@(x) isa(x,'dagnn.Loss'), {net.layers.block})) ;
	for ii = 1:numel(sel)
    if isa(net.layers(sel(ii)).block, 'dagnn.ErrorStats')
      metrics = net.layers(sel(ii)).block.average ;
      classDist = net.layers(sel(ii)).block.classDist ;
      population = classDist / sum(classDist) ;
      stats.meanAcc = mean(metrics) ;
      for jj = 1:numel(metrics)
        stats.(net.meta.classes.name{jj}) = metrics(jj) ;
      end
      for jj = 1:numel(population)
        stats.(sprintf('%sPop', net.meta.classes.name{jj})) = population(jj) ;
      end
    else
      if net.layers(sel(ii)).block.ignoreAverage, continue ; end
      outName = net.layers(sel(ii)).outputs{1} ;
      stats.(outName) = net.layers(sel(ii)).block.average ;
    end
	end
end

% -------------------------------------------------------------------------
function fn = getBatchFn(opts, meta)
% -------------------------------------------------------------------------
	useGpu = numel(opts.gpus) > 0 ;
	bopts.numThreads = opts.numFetchThreads ;
	bopts.imageSize = opts.imageSize ;
	bopts.averageImage = [] ; % will be overwritten with zero if inpunorm is used
  bopts.fixedSegments = opts.fixedSegments ;
	bopts.transformation = meta.augmentation.transformation ;
	bopts.numAugments = opts.numAugments ;
  bopts.numPredEmotions = opts.numPredEmotions ;
  %bopts.subsetField = opts.subsetField ;
  bopts.logitAggregator = opts.logitAggregator ;
  bopts.lossType = opts.lossType ;
  %bopts.rawLogits = opts.rawLogits ;
	bopts.meta = meta ;
  %fn = @(x,y) getDagNNBatchAudio(bopts,useGpu,x,y) ;
  fn = @(x,y) getBatchEmoVoxCeleb(bopts,useGpu,x,y) ;
end

% -------------------------------------------------------------------------
function storeMetaInfo(opts)
% -------------------------------------------------------------------------
  nowStr = strrep(strrep(datestr(datetime('now')), ' ', '_'), ':', '-') ;
  metaPathTxt = fullfile(opts.expDir, sprintf('meta-%s.txt', nowStr)) ;
  metaPathMat = fullfile(opts.expDir, sprintf('meta-%s.mat', nowStr)) ;
  fprintf('storing meta information to %s\n', metaPathTxt) ;
  txt = struct2str(opts) ;
  [~,ret] = system('hostname') ;
  txt = [sprintf('server: %s\n', strtrim(ret)) txt] ;
  fid = fopen(metaPathTxt,'w') ;
  fprintf(fid, txt) ;
  fclose(fid) ;
  save(metaPathMat, '-struct', 'opts') ;
end
