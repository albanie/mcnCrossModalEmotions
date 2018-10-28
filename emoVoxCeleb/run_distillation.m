function [net, info] = run_distillation(varargin)
%RUN_DISTILLATION - run distillation process to train student
%   [NET, INFO] = RUN_DISTILLATION - trains a student to match the
%   predictions of a teacher network, as described in:
%
%     S. Albanie, A. Nagrani, A. Vedaldi, A. Zisserman,
%     Emotion Recognition in Speech using Cross-Modal Transfer
%     in the Wild, ACM Multimedia 2018
%
%   RUN_DISTILLATION(..'name', value) accepts the following options:
%
%   `gpus` :: 2
%    The gpu device to use for processing.
%
%   `cont` :: true
%    Whether to restart training from a previous checkpoint when
%    one is available.
%
%   `miniVal` :: 0.2
%    Can be used to only perform validation on a subset of the full
%    validation set to reduce computational cost.
%
%   `numSeconds` :: 4
%    The duration of the audio segment to be processed by the student.
%
%   `batchSize` :: 64
%    The batch size used by the student during training.
%
%   `numEpochs` :: 250
%    While an "epoch" traditionally corresponds to a full pass over the
%    training set, this option can be combined with the `miniEpochRatio`
%    to specify the number of "mini-epochs", which are subsamples of the
%    full training data.
%
%   `miniEpochRatio` :: 0.2
%    The proportion of the data to use in each "mini-epoch".
%
%   `numPredEmotions` :: 8
%    The number of emotions predicted by the student network - can be up
%    to 8 (the number predicted by the teacher).
%
%   `fromScratch` :: true
%    Randomly initialise the weights of the student.
%
%   `logitAggregator` :: 'max'
%    The mechanism used to combine frame-level predictions from the teacher
%    to provide a target for the student.
%
%   `teacher` :: 'senet50-ferplus'
%    The name of the teacher model.
%
%   `student` :: 'emovoxceleb-student'
%    The name of the student model.
%
%   `lossType` :: 'hot-cross-ent'
%    The type of loss to be used for distillation (can also be Euclidean
%    or softmaxlog).
%
%   `temperature` :: 2
%    The temperature used by the softmax in the "hot" cross entropy loss.
%
%   `learningRate` :: logspace(-4, -5, opts.numEpochs)
%    The learning rate used by the student.
%
%   `wavDir` :: fullfile(vl_rootnn, 'data/ramdisk/voxceleb_all')
%    Directory containing the wavfiles of VoxCeleb.
%
% Copyright (C) 2018 Samuel Albanie, Arsha Nagrani
% Licensed under The MIT License [see LICENSE.md for details]

  opts.gpus = 2 ;
  opts.cont = true ;
  opts.miniVal = 0.2 ;
  opts.numSeconds = 4 ;
  opts.batchSize = 64 ;
  opts.numEpochs = 300 ;
  opts.miniEpochRatio = 0.05 * numel(opts.gpus) ;
  opts.numPredEmotions = 8 ;
  opts.fromScratch = true ;
  opts.logitAggregator = 'max' ;
	opts.datasetName = 'voxceleb' ;
  opts.teacher = 'senet50-ferplus' ;
  opts.student = 'emovoxceleb-student' ;
  opts.lossType = 'hot-cross-ent' ;
  opts.temperature = 2 ;
  opts.fixedSegments = false ;
	opts.learningRate = logspace(-4, -5, opts.numEpochs) ;
  opts.parameterServer = 'tmove' ;
  opts.wavDir = fullfile(vl_rootnn, 'data/ramdisk/voxceleb_all') ;
  opts = vl_argparse(opts, varargin) ;

  % hack to avoid clearing persistent memory issue
  imdb = fetch_emovoxceleb_imdb(opts.teacher) ;

	student = sprintf('%s-%s', opts.student, opts.lossType) ;
	if opts.fromScratch, student = [student '-scratch'] ; end
  expName = sprintf('voxceleb-%s-%s-%dsec-%demo-agg-%s', ...
                    opts.teacher, student, opts.numSeconds, ...
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
  trainSamples = find(imdb.images.set == 1) ;
  valSamples = find(imdb.images.set == 2) ;

  % for efficiency, it can be helpful to only use a portion of the val set
  if opts.miniVal < 1
    rng(0) ;
    pick = randsample(numel(valSamples), ...
                     round(numel(valSamples) * opts.miniVal)) ;
    valSamples = valSamples(pick) ;
  end

  % print summary
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

  % sanity check for audio files
  sampleAudioDir = fullfile(opts.wavDir, 'Aamir_Khan') ;
  msg = 'could not find expected audio file at %s, did you download VoxCeleb?' ;
  assert(logical(exist(sampleAudioDir, 'dir')), sprintf(msg, sampleAudioDir)) ;

  %trainfn = @cnn_train_dag_check2 ; % handles broken checkpoint issue
  trainfn = @cnn_train_dag ;
	[net, info] = trainfn(net, imdb, getBatchFn(opts, meta), ...
		'learningRate', opts.learningRate, ...
		'batchSize', opts.batchSize,...
		'numEpochs', opts.numEpochs, ...
    'train', trainSamples, ...
    'val', valSamples, ...
    'continue', opts.cont, ...
		'expDir', opts.expDir, ...
		'gpus', opts.gpus, ...
    'epochSize', opts.epochSize, ...
    'parameterServer', struct('method', opts.parameterServer), ...
    'extractStatsFn', @extractStats) ;
end

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
	bopts.numAugments = 1 ;
	bopts.numThreads = opts.numFetchThreads ;
	bopts.imageSize = [512, opts.numSeconds * 100] ;
	bopts.averageImage = [] ; % overwritten with zero when inputnorm is used
	bopts.transformation = meta.augmentation.transformation ;
  bopts.numPredEmotions = opts.numPredEmotions ;
  bopts.logitAggregator = opts.logitAggregator ;
  bopts.fixedSegments = opts.fixedSegments ;
  bopts.lossType = opts.lossType ;
	bopts.meta = meta ;
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
