function benchmark_ferplus_models(varargin)
%BENCHMARK_FERPLUS_MODELS - evaluate trained models on FERPLUS
%   BENCHMARK_FERPLUS_MODELS - evaluates the performance of trained emotion
%   recognition CNNs on the FER+ validation and test sets.
%
%   BENCHMARK_FERPLUS_MODELS(..'name', value) accepts the following
%   options:
%
%   `refresh` :: false
%    If true, refreshes any cached results from previous evaluations.
%
%   `benchmarkCacheDir` :: fullfile(vl_rootnn, 'data/ferPlus/benchCache')
%    Directory where evaluation results for each model will be cached.
%
%   `dataDir` :: fullfile(vl_rootnn, 'data/datasets/fer2013+')
%    Directory containing the FER2013 and FER2013+ datasets files (in csv
%    format).
%
% Copyright (C) 2018 Samuel Albanie
% Licensed under The MIT License [see LICENSE.md for details]

  opts.refresh = false ;
  opts.dataDir = fullfile(vl_rootnn, 'data/datasets/fer2013+') ;
  opts.benchmarkCacheDir = fullfile(vl_rootnn, 'data/ferPlus/benchCache') ;
  opts = vl_argparse(opts, varargin) ;

  if ~exist(opts.benchmarkCacheDir, 'dir')
    mkdir(opts.benchmarkCacheDir) ;
  end

  pretrained = {
    {'resnet50-ferplus', 'softmaxlog'},...
    {'senet50-ferplus', 'distributions'},...
	} ;

  for ii = 1:numel(pretrained)
    fprintf('-----------------------------------------------------------\n') ;
    modelName = pretrained{ii}{1} ;
    lossType = pretrained{ii}{2} ;
    cachePath = fullfile(opts.benchmarkCacheDir, ...
                         sprintf('%s.mat', modelName)) ;
    if exist(cachePath, 'file') && ~opts.refresh
      fprintf('loading cached results for %s on FER...\n', modelName) ;
      stats = load(cachePath) ;
    else
      commonArgs = {'modelName', modelName, ...
                    'train.batchSize', 32, ...
                    'lossType', lossType, ...
                    'dataDir', opts.dataDir} ;
      fprintf('evaluating %s on FER+...\n', modelName) ;
      [~,valInfo] = ferplus_baselines(commonArgs{:}, ...
                                     'evaluateOnly.subset', 'val') ;
      [~,testInfo] = ferplus_baselines(commonArgs{:}, ...
                                     'evaluateOnly.subset', 'test') ;
      valAcc = 1 - valInfo.val.classerror ;
      testAcc = 1 - testInfo.val.classerror ;
      stats.valAcc = valAcc ; stats.testAcc = testAcc ;
      fprintf('caching results for %s to %s ...', modelName, cachePath) ; tic ;
      save(cachePath, '-struct', 'stats') ;
      fprintf('done in %g s \n', toc) ;
    end
    fprintf('%s: val (%.3f), test: (%.3f)\n', ...
                         modelName, stats.valAcc, stats.testAcc) ;
	end
