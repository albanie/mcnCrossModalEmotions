function audio_benchmarks(varargin)
%AUDIO_BENCHMARKS evalaute speech features
%   AUDIO_BENCHMARKS(VARARGIN) evaluates the performance of the student
%   model on external benchmarks.  For each dataset, it learns a
%   classifier (consisting of a matrix and a bias followed by a softmax)
%   which reweights the emotions of the student.  The size of this learned
%   matrix is S x T, where S is the number of emotions predicted by the student
%   and T is the number of emotions in the target dataset.  Since S
%   is 8 for models trained on EmoVoxCeleb, it essentially treats the
%   predictions of the student as 8-dimensional "embeddings" and uses them
%   to perform a constrained multinomial logistic regression.
%
%   AUDIO_BENCHMARKS(..'name', value) accepts the following
%   options:
%
%   `clobber` :: false
%    If true, refreshes any cached results from previous runs.
%
%   `datasets` :: {'rml', 'enterface'}
%    The names of the datasets to evaluate, as a cell array.
%
%   `modelName` :: 'emovoxceleb-student'
%    The name of the audio model to be evaluated.
%
%   `numEpochs' :: 50
%    Out of terrible laziness, the classifier is fitted with SGD, but
%    should be updated to IRLS or similar.
%
%   `figDir` :: fullfile(vl_rootnn, 'data/affine-figs-audio-splits')
%    The directory where the final figures should be stored.
%
% Copyright (C) 2018 Samuel Albanie, Arsha Nagrani
% Licensed under The MIT License [see LICENSE.md for details]

  opts.clobber = false ;
  opts.numEpochs = 50 ; % out of laziness
  opts.mnrfit = false ;
  opts.datasets = {'rml', 'enterface'} ;
  opts.modelName = 'emovoxceleb-student' ;
  opts.figDir = fullfile(vl_rootnn, 'data/affine-figs-audio-splits') ;
  opts = vl_argparse(opts, varargin) ;

  if ~exist(opts.figDir, 'dir'), mkdir(opts.figDir) ; end

  % The labels learned by the student on EmoVoxCeleb
	modelEmoLabels = {'neutral', 'happiness', 'surprise', ...
										'sadness', 'anger', 'disgust', ...
										'fear', 'contempt'} ;
  numSrcEmotions = numel(modelEmoLabels) ;

  % Labels used by externel benchmarks
  datasetLabels = {'Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise'} ;
  numTargetEmotions = numel(datasetLabels) ;

  for ii = 1:numel(opts.datasets)

    fprintf('learning classifier weights...') ;
    dataset = opts.datasets{ii} ;
    [miniImdb, expDirs, valIdxSets] = run_cross_val(...
               'modelName', opts.modelName, ...
               'targetDataset', dataset, ...
               'refreshCkpts', opts.clobber, ...
               'numSrcEmotions', numSrcEmotions, ...
               'numEpochs', opts.numEpochs, ...
               'mnrfit', opts.mnrfit, ...
               'numTargetEmotions', numTargetEmotions, ...
               'modality', 'audio') ;

    foldAccs = zeros(1, numel(expDirs)) ;
    confSum = zeros(numTargetEmotions, numTargetEmotions) ;

    for ee = 1:numel(expDirs) % compute accuracy metrics

      expDir = expDirs{ee} ;
			% compute desired metrics
      valIdx = valIdxSets{ee} ;
      X = miniImdb.fusedLogits(valIdx,:) ;

      if opts.mnrfit
        paramPath = fullfile(expDir, 'mnr-params.mat') ;
        tmp = load(paramPath) ;
        preds = mnrval(tmp.coefficients, double(X)) ;
      else % SGD
        bestEpoch = findBestEpoch(expDir, 'prune', false) ;
        bestNet = fullfile(expDir, sprintf('net-epoch-%d.mat', bestEpoch)) ;
        tmp = load(bestNet) ; net = Net(tmp.net) ;
        w = net.getValue('prediction_filters') ;
        w = reshape(w, numSrcEmotions, numTargetEmotions) ;
        preds = X * w ;
        b = net.getValue('prediction_biases') ;
        preds = bsxfun(@plus, preds, b') ; % include biases
      end

      labels = miniImdb.labels(valIdx) ;
      [~,cls] = max(preds, [], 2) ;
      matches = cls == labels' ;
      acc = sum(matches) / numel(matches) ;
      confmat = confusionmat(labels', cls, 'Order', 1:numTargetEmotions) ;
      confSum = confSum + confmat ;
      fprintf('(fold %d/%d) recomputed accuracy: %.1f\n', ...
            ee, numel(expDirs), 100 * acc) ;
      acc = 100 * (1 - tmp.stats.val(end).error) ;
      fprintf('(fold %d/%d) accuracy for %s: %.1f\n', ee, numel(expDirs), ...
                                               opts.modelName, acc) ;
      foldAccs(ee) = acc ;
    end

    fprintf('-----------------------------\n') ;
    fprintf('DATASET: %s\n', dataset) ;
    fprintf('MODEL: %s\n', opts.modelName) ;
    fprintf('cross-validation score: %g, std %g \n', ...
                           mean(foldAccs), std(foldAccs)) ;
    fprintf('-----------------------------\n') ;
    fprintf('confusion matrix:\n') ;
    disp(confSum) ;
    fprintf('-----------------------------\n') ;
    fprintf('normalized confusion matrix:\n') ;
    normed = bsxfun(@rdivide, confSum, sum(confSum, 2)) ;
    disp(normed) ;

    generate_confmatrix_fig(normed, datasetLabels, dataset, ...
                           opts.figDir, opts.modelName) ;
  end
end

% ---------------------------------------------------------------------------
function generate_confmatrix_fig(normed, labels, dataset, figDir, modelName)
% ---------------------------------------------------------------------------
  clf ;  h = figure(1) ; confmat_colors(normed, labels) ;
  figName = sprintf('%s-%s.pdf', dataset, modelName) ;
  figPath = fullfile(figDir, 'confmat', figName) ;
  if ~exist(fileparts(figPath), 'dir'), mkdir(fileparts(figPath)) ; end

  set(h,'Units','Inches') ; pos = get(h,'Position') ;
  set(h,'PaperPositionMode','Auto', ...
        'PaperUnits','Inches', ...
        'PaperSize', [pos(3), pos(4)]) ;

  print(figPath, '-dpdf','-r0') ;
  fprintf('saving figure at %s\n', figPath) ;
  if exist('zs_dispFig', 'file'), zs_dispFig ; end
end
