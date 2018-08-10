function emo_benchmarks(varargin)
%EMO_BENCHMARKS evalaute an emotion recogntion model on benchmarks
%   EMO_BENCHMARKS(VARARGIN) evaluates the performance of an emotion
%   recognition model on external benchmarks.  For each benchmark, it learns
%   a classifier (consisting of a matrix and a bias followed by a softmax)
%   which reweights the emotions of the student (essentially it does
%   multinomial logistic regression to map the emotion predictions of the
%   model to the set of target labelled emotions on each benchmark (which
%   typically differ across datasets). The size of the learned parameter
%   matrix is S x T, where S is the number of emotions predicted by the
%   model under evaluation and T is the number of emotions in the target
%   dataset.
%
%   EMO_BENCHMARKS(..'name', value) accepts the following
%   options:
%
%   `datasets` :: {'rml', 'enterface'}
%    The names of the datasets to evaluate, as a cell array.
%
%   `modelName` :: 'emovoxceleb-student'
%    The name of the classification model to be evaluated. Running with
%    'random' provides a useful sanity check (it should achieve an accuracy
%    of between 0.15 and 0.2 (there are six target classes for the dataset
%    above, so expected accuracy is 0.167).
%
%   `modality` :: 'audio'
%    Can be 'audio' or 'visual' (must correspond to the input modality
%    of the model specified by 'modelName').
%
%   `figDir` :: fullfile(vl_rootnn, 'data/affine-figs-audio-splits')
%    The directory where the final figures should be stored.
%
% Copyright (C) 2018 Samuel Albanie, Arsha Nagrani
% Licensed under The MIT License [see LICENSE.md for details]

  opts.modality = 'audio' ;
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

  for ii = 1:numel(opts.datasets)

    fprintf('learning classifier weights...') ;
    dataset = opts.datasets{ii} ;

    % Labels used by externel benchmarks
    switch dataset
      case {'rml', 'enterface'}
        datasetLabels = {'Angry', 'Disgust', 'Fear', 'Happy', ...
                         'Sad', 'Surprise'} ;
        numFolds = 10 ; useExstingVal = false ;
        adjustmentFactor = 1 ; % no adjustment required (see explanation below)
      case 'afew'
        datasetLabels = {'Angry', 'Disgust', 'Fear', 'Happy', ...
                        'Sad', 'Surprise', 'Neutral'} ;
        % Note that AFEW already has a predefined validation set
        % so this is used instead of x-validation.
        numFolds = 1 ; useExstingVal = true ;
        % Afew accuracy assessment requires a v. small adjustment factor to
        % account for the fact that two of the validation tracks are dropped
        % during evaluation (because they contain no face detections), so
        % we must adjust the final accuracy accordingly.
        adjustmentFactor = (381 / 383) ;
    end
    numTargetEmotions = numel(datasetLabels) ;

    [miniImdb, expDirs, valIdxSets] = run_cross_val(...
               'modelName', opts.modelName, ...
               'targetDataset', dataset, ...
               'numSrcEmotions', numSrcEmotions, ...
               'useExstingVal', useExstingVal, ...
               'numFolds', numFolds, ...
               'numTargetEmotions', numTargetEmotions, ...
               'modality', opts.modality) ;

    foldAccs = zeros(1, numel(expDirs)) ;
    confSum = zeros(numTargetEmotions, numTargetEmotions) ;

    for ee = 1:numel(expDirs) % compute accuracy metrics
      expDir = expDirs{ee} ;
      valIdx = valIdxSets{ee} ;
      X = miniImdb.fusedLogits(valIdx,:) ;
      paramPath = fullfile(expDir, 'mnr-params.mat') ;
      tmp = load(paramPath) ;
      preds = mnrval(tmp.coefficients, double(X)) ;
      labels = miniImdb.labels(valIdx) ;
      [~,cls] = max(preds, [], 2) ;
      matches = cls == labels' ;
      acc = sum(matches) / numel(matches) * adjustmentFactor ;
      confmat = confusionmat(labels', cls, 'Order', 1:numTargetEmotions) ;
      confSum = confSum + confmat ;
      fprintf('(fold %d/%d) recomputed accuracy: %.1f\n', ...
            ee, numel(expDirs), 100 * acc) ;
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

    datasetLabels = canonicalLabels(datasetLabels) ;

    generate_confmatrix_fig(normed, datasetLabels, dataset, ...
                           opts.figDir, opts.modelName) ;
  end
end

% ---------------------------------------------------------------------------
function labels = canonicalLabels(labels)
% ---------------------------------------------------------------------------
%CANONICALLABELS - map the label strings to a canonical form
%  LABELS = CANONICALLABELS(LABELS) modifies the names of the dataset labels
%  so that they are consistent with labels used in the Fer2013+ dataset.

  strMap = containers.Map() ;
  strMap('Fear') = 'Fear' ;
  strMap('Sad') = 'Sadness' ;
  strMap('Angry') = 'Anger' ;
  strMap('Neutral') = 'Neutral' ;
  strMap('Happy') = 'Happiness' ;
  strMap('Disgust') = 'Disgust' ;
  strMap('Surprise') = 'Surprise' ;
  labels = cellfun(@(x) {strMap(x)}, labels) ;
end

% ---------------------------------------------------------------------------
function generate_confmatrix_fig(mat, labels, dataset, figDir, modelName)
% ---------------------------------------------------------------------------
%GENERATE_CONFMATRIX_FIG produce confusion matrix figure
%   GENERATE_CONFMATRIX_FIG(MAT, LABELS, DATASET, FIGDIR, MODELNAME) generates
%   a figure displaying the normalized confusion matrix as colors, together
%   with the values of each entry.
%
%  This is based on the 2017 function by Betthauser, which can be found here:
%  https://it.mathworks.com/matlabcentral/mlc-downloads/downloads/...
%  submissions/67821/versions/2/previews/emg_functions/...
%  performance_metrics/plotConfuse.m/index.html?access_key=

  clf ;  h = figure(1) ;
  set(0,'defaulttextinterpreter','latex') ;
  numClasses = numel(labels) ;
	imagesc(mat) ; % Create a colored plot of the matrix values
	colormap(flipud(gray)) ;  % Change colormap to gray (lower values are light)
	textStrings = num2str(mat(:), '%0.2f') ; % Create strings from matrix values
	textStrings = strtrim(cellstr(textStrings)) ; % Remove any space padding
	[x, y] = meshgrid(1:numClasses) ;  % Create x and y coords for the strings
	hStrings = text(x(:), y(:), textStrings(:), ...  % Plot strings
					'HorizontalAlignment', 'center', 'fontsize', 18) ;
	midValue = mean(get(gca, 'CLim')) ;  % get middle value of the color range

  % choose white or black for the text color of the strings to amke them
  % more easily visible over the background.
	textColors = repmat(mat(:) > midValue, 1, 3) ;
	set(hStrings, {'Color'}, num2cell(textColors, 2)) ; % Change text colors

	% Change the axes tick marks and tick labels
	set(gca, 'XTick', 1:numClasses, ...
					 'XTickLabel', labels, ...
					 'YTick', 1:numClasses, ...
					 'YTickLabel', labels, ...
					 'TickLength', [0 0]) ;
  a = get(gca,'XTickLabel') ;
  set(gca, 'YTickLabel', a, 'fontsize', 20) ;

  xtickangle(60) ;

  set(groot, 'defaultAxesTickLabelInterpreter','latex') ;
  set(groot, 'defaultLegendInterpreter','latex') ;

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

