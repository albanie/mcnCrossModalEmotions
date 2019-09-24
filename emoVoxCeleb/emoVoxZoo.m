function dag = emoVoxZoo(modelName, varargin)
%EMOVOXZOO - load EmoVoxCeleb model by name
%   DAG = EMOVOXZOO(MODELNAME, VARARGIN) - loads a pretrained
%   emotion recognition CNN given its name, MODELNAME.
%
%   EMOVOXZOO(..'name', value) accepts the following options:
%
%   `modelDir` :: fullfile(vl_rootnn, 'data/models-import', subfolder)
%    Directory to search for models
%
%   `scratch` :: false
%    If true, all parameters are randomly initialised
%
% Copyright (C) 2018 Samuel Albanie, Arsha Nagrani
% Licensed under The MIT License [see LICENSE.md for details]

  opts.scratch = false ;
  opts.dropout = false ;
  opts.numOutputs = 8 ;
  opts.numSeconds = 4 ;
  opts.lossType = 'hot-cross-ent' ;
	opts.modelDir = fullfile(vl_rootnn, 'data/models-import') ;
  opts = vl_argparse(opts, varargin) ;

  studentModels = { ...
    'emovoxceleb-student', ...
  } ;
  teacherModels = {
    'resnet50-ferplus', ...
    'senet50-ferplus', ...
  } ;

  modelNames = [studentModels teacherModels] ;
  msg = sprintf('%s: unrecognised model', modelName) ;
  assert(ismember(modelName, modelNames), msg) ;
	modelPath = fullfile(opts.modelDir, sprintf('%s.mat', modelName)) ;
  if ~exist(modelPath, 'file')
    fetchModel(modelName, modelPath) ;
  end
  net = load(modelPath) ;
  if ~opts.scratch
    if isfield(net, 'net'), net = net.net ; end
    net = fixBackwardsCompatibility(net) ;
    dag = dagnn.DagNN.loadobj(net) ;
    dag = fixInputVarnames(dag) ;
    fprintf('loaded pretrained %s model...\n', modelName) ;
    return
  end

  dag = prepareFromDagNN(net, opts.numOutputs) ;
	fprintf('\n-----------------------------------------------------\n') ;
	fprintf('Initialising parameters from scratch!                  \n') ;
	fprintf('-------------------------------------------------------\n') ;
	dag.initParams() ;

  % configure the loss layers
  dag = configureForRegression(dag, opts.lossType, ...
                              opts.dropout, opts.numOutputs) ;

  % update the pooling layer for the given duration
  dag = updatePooling(dag, opts.numSeconds, modelName) ;
	dag = fixInputVarnames(dag) ;

% ----------------------------------------------
function dag = fixInputVarnames(dag)
% ----------------------------------------------
  candidates = {'input', 'x0'} ;
  ins = dag.getInputs() ;
  for ii = 1:numel(ins)
    if ismember(ins{ii}, candidates), dag.renameVar(ins{ii}, 'data') ; end
  end

% ---------------------------------------
function fetchModel(modelName, modelPath)
% ---------------------------------------

  waiting = true ;
  prompt = sprintf(strcat('%s was not found at %s\nWould you like to ', ...
          ' download it from THE INTERNET (y/n)?\n'), modelName, modelPath) ;

  while waiting
    str = input(prompt,'s') ;
    switch str
      case 'y'
        if ~exist(fileparts(modelPath), 'dir')
          mkdir(fileparts(modelPath)) ;
        end
        fprintf(sprintf('Downloading %s ... \n', modelName)) ;
        if contains(modelName, 'emovoxceleb')
          subfolder = 'emovoxceleb' ;
        elseif contains(modelName, 'ferplus')
          subfolder = 'ferplus' ;
        end

        baseUrl = ['http://www.robots.ox.ac.uk/~albanie/models/' subfolder] ;
        url = sprintf('%s/%s.mat', baseUrl, modelName) ;
        urlwrite(url, modelPath) ;
        return ;
      case 'n', throw(exception) ;
      otherwise, fprintf('input %s not recognised, please use `y/n`\n', str) ;
    end
  end

% ----------------------------------------------------------------------------
function dag = configureForRegression(dag, lossType, dropout, numOutputs)
% ----------------------------------------------------------------------------
%CONFIGUREFORREGRESSION configures the network to train as a classifer
%  CONFIGUREFORREGRESSION(dag) adds a softmaxlog loss
%   and classerror loss on top of the fully connected output predictions
%   of the network to perform classification.
%
%   A fine tuning learning rate is set on each of the network parameters.
%   Appropriate meta information is also added for the emotion recognition
%   task

  hasDropout = any(arrayfun(@(x) isa(x.block, 'dagnn.Dropout'), dag.layers)) ;
  if dropout > 0  && ~hasDropout
    convIdx = arrayfun(@(x) isa(x.block, 'dagnn.Conv'), dag.layers) ;
    convLayers = dag.layers(convIdx) ;
    sel =  convLayers(end-2:end-1) ; % reduce aggression
    for ii = 1:numel(sel)
      prev = sel(ii).name ;
      out = sel(ii).outputs ;
      found = false ;
      for jj = 1:numel(dag.layers)
        if ismember(out, dag.layers(jj).inputs)
          found = true ;
          next = dag.layers(jj).name ;
          break ;
        end
      end
      assert(found, 'target layer was not found') ;
	  	dag = insert_dropout(dag, prev, next, dropout) ;
    end
  end

  switch lossType
    case 'euclidean'
			layer = dagnn.EuclideanLoss() ;
      inputs = {'prediction', 'logitTarget', 'instanceWeights'} ;
      % scale down a lot to prevent exploding gradients
      pIdx = dag.getParamIndex(dag.layers(end).params(1)) ;
      curr = dag.params(pIdx).value ;
      dag.params(pIdx).value = curr / 10 ;
    case 'huber'
			layer = dagnn.HuberLoss('sigma', 1) ;
      inputs = {'prediction', 'logitTarget', 'instanceWeights'} ;
    case 'softmaxlog'
			layer = dagnn.Loss('loss', 'softmaxlog') ;
      inputs = {'prediction', 'maxLabel'} ;
    case 'hot-cross-ent'
			layer = dagnn.SoftmaxCELoss('temperature', 2, 'logitTargets', true) ;
      inputs = {'prediction', 'logitTarget'} ;
    otherwise, error('unrecognised regression loss: %s\n', lossType) ;
  end
  output = 'objective' ;
  dag.addLayer('loss', layer, inputs, output) ;

  % Add class error as a proxy for performance
  layer = dagnn.VerboseLoss('loss', 'classerror') ;
  inputs = {'prediction', 'maxLabel'} ;
  output = 'classerror' ;
  dag.addLayer('classerror', layer, inputs, output)  ;

  % Add class error as a proxy for performance
	layer = dagnn.ErrorStats('numClasses', numOutputs) ;
	inputs = {'prediction', 'maxLabel'} ;
	output = 'classAccs' ;
	dag.addLayer('classAccs', layer, inputs, output)  ;

  %% Add class error as a proxy for performance
  %layer = dagnn.ClassRecall('numClasses', numOutputs) ;
  %inputs = {'prediction', 'maxLabel'} ;
  %output = 'classRec' ;
  %dag.addLayer('classRec', layer, inputs, output)  ;

  dag.rebuild()

  % modify the meta attributes of the net
  emotions = {'neutral', 'happiness', 'surprise', 'sadness', ...
              'anger', 'disgust', 'fear', 'contempt'} ;
  dag.meta.classes.name = emotions ;
  dag.meta.classes.description = emotions ;


% ----------------------------------------------
function dag = prepareFromDagNN(net, numOutputs)
% ----------------------------------------------
%PREPAREFROMDAGNN prepares a DagNN structure for classification
%  PREPARESFROMIMPLENN(dag, numOutputs) prepares a DagNN
%  network for classification by removing any old loss layers
%  and ensuring that the final fully connected "prediction"
%  layer has the correct dimensions.

  if isfield(net, 'net'), net = net.net ; end
	net = fixBackwardsCompatibility(net) ;

  % load stored network into memory
  dag = dagnn.DagNN.loadobj(net) ;

  % remove previous loss layers / softmax layers
  for l = numel(dag.layers): -1:1
    if isa(dag.layers(l).block, 'dagnn.Loss')
      dag.removeLayer(dag.layers(l).name) ;
    elseif isa(dag.layers(l).block, 'dagnn.SoftMax')
      dag.removeLayer(dag.layers(l).name) ;
    end
  end

  % modify last fully connected layer for multi-way classification
  layerOrder = dag.getLayerExecutionOrder() ;
  finalLayer = dag.layers(layerOrder(end)) ;
  numChannels = finalLayer.block.size(3) ;
  finalLayer.block.size = [1 1 numChannels numOutputs] ;

  % Initialize the params of the new prediction layer
  rng('default') ; rng(0) ;
  fScale = 1/10000 ;
  filters = fScale * randn(1, 1, numChannels, numOutputs, 'single') ;
  biases = zeros(numOutputs, 1, 'single') ;

  % Handle possible naming conventions for parameters:
  % Filter params can be called:
  % <layername>f, <layername>_filter, <layername>_f
  filterIdx = dag.getParamIndex(sprintf('%sf', finalLayer.name)) ;
  if isnan(filterIdx)
    filterIdx = dag.getParamIndex(sprintf('%s_filter', finalLayer.name)) ;
  end
  if isnan(filterIdx)
    filterIdx = dag.getParamIndex(sprintf('%s_f', finalLayer.name)) ;
  end

  % Bias params can be called: <layername>b, <layername>_bias, <layername>_b
  biasIdx = dag.getParamIndex(sprintf('%sb', finalLayer.name)) ;
  if isnan(biasIdx)
    biasIdx = dag.getParamIndex(sprintf('%s_bias', finalLayer.name)) ;
  end
  if isnan(biasIdx)
    biasIdx = dag.getParamIndex(sprintf('%s_b', finalLayer.name)) ;
  end

  dag.params(filterIdx).value = filters ;
  dag.params(biasIdx).value = biases ;

  % Rename input variable to 'input' be consistent with other networks
  firstLayer = dag.layers(layerOrder(1)) ;
  if ~strcmp(firstLayer.inputs, 'input')
    dag.renameVar(firstLayer.inputs, 'input') ;
  end

  % rename the output of the last fully connected layer to "prediction"
  predictionVar = dag.layers(dag.getLayerIndex(finalLayer.name)).outputs ;
  dag.renameVar(predictionVar, 'prediction') ;

% -------------------------------------------------------
function dag = updatePooling(dag, numSeconds, modelName)
% -------------------------------------------------------
	buckets.pool = [2 5 8 11 14 17 20 23 27 30] ;
	buckets.width = [100 200 300 400 500 600 700 800 900 1000] ;
	p1 = buckets.pool((numSeconds * 100) == buckets.width) ;
  switch modelName
    case {'vggvox_ident_net', 'vggm_bn_identif', 'emovoxceleb-student'}
      layerName = 'pool6' ;
    case 'resnet_identif', layerName = 'pool_time' ;
    otherwise, error('modelName: %s not recognised\n', modelName) ;
  end
	idx = dag.getLayerIndex(layerName) ;
  assert(numel(idx) == 1, 'expected a single pooling layer') ;
	dag.layers(idx).block.poolSize=[1 p1] ;

% --------------------------------------------------
function net = insert_dropout(net, prev, next, rate)
% --------------------------------------------------
	in = net.layers(net.getLayerIndex(prev)).outputs ;
	lname = sprintf('%s_drop', prev) ; out = lname ;
	net.addLayer(lname, dagnn.DropOut('rate', rate), in, out, {}) ;
	net.setLayerInputs(next, {out}) ;

% ----------------------------------------------------------
function net = fixBackwardsCompatibility(net)
% ----------------------------------------------------------
%FIXBACKWARDSCOMPATIBILITY - remove unsupported attributes
%  NET = FIXBACKWARDSCOMPATIBILITY(NET) enables backwards
%  compatibility by remvoing attributes that are no longer
%  supported.

  removables = {'exBackprop'} ;
  for ii = 1:numel(net.layers)
    for jj = 1:numel(removables)
      fieldname = removables{jj} ;
      if isfield(net.layers(ii).block,fieldname)
        net.layers(ii).block = rmfield(net.layers(ii).block, fieldname) ;
      end
    end
  end
