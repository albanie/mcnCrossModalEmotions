function [dag, pretrained] = ferPlusZoo(modelName, varargin)
%FERPLUSZOO - load classifier by name
%   [DAG, ISPRETRAINED] = FERPLUSZOO(MODELNAME, VARARGIN) - loads an
%   emotion recognition CNN given its name, MODELNAME.
%
%   FERPLUSZOO(..'name', value) accepts the following options:
%
%   `useBnorm` :: true
%    If true, inserts batch norm layers into a model after
%    each convolution. If the model already contains any batch norm layers,
%    this option does nothing.
%
%   `finetuneLR` :: 1
%    Sets the finetuning learning rate for all parameters except those
%    used in the classifier.
%
%   `dropoutRate` :: 0.5
%    If dropoutRate > 0, dropout layers with the given rate are inserted
%    into the model.
%
%   `lossType` :: 'distributions'
%    The loss used to train the classifier.  Can either be 'softmaxlog'
%    or 'distributions' (uses a distributional cross entropy, rather than
%    assuming 1-hot labels).
%
% Copyright (C) 2018 Samuel Albanie, Arsha Nagrani
% Licensed under The MIT License [see LICENSE.md for details]

   opts.useBnorm = false ;
   opts.finetuneLR = 1 ;
   opts.numOutputs = 7 ;
   opts.dropoutRate = 0 ;
   opts.dropooutRate = 0.5 ;
   opts.lossType = 'distributions' ;
   opts = vl_argparse(opts, varargin) ;

  vggFace2Models = {
		'resnet50_ft-dag', ...
		'resnet50_scratch-dag', ...
		'senet50_ft-dag', ...
		'senet50_scratch-dag', ...
  } ;
  standardModels = {
    'vgg-m-face-bn-fer', ...
    'vgg-m-face-bn', ...
    'vgg-vd-face-fer', ...
    'vgg-vd-face', ...
    'vgg_face', ...
		'resnet50_ft-dag', ...
  } ;
  ferModels = { ...
    'vgg-vd-face-fer', ...
    'vgg-vd-face-sfew-dag', ...
    'vgg-m-face-bn-fer', ...
  } ;
  sfewModels = {
    'vgg-vd-face-sfew', ...
    'resnet50-face-sfew', ...
  } ;
  ferPlusModels = {
    'resnet50-ferplus', ...
    'senet50-ferplus', ...
  } ;
  ferPlusModelsDev = {
    'resnet50_ft-dag-dropout-0.1', ...
    'resnet50_ft-dag-dropout-0.5', ...
    'senet50_ft-dag-distributions-dropout-0.5-aug' ...
    'senet50_ft-dag-distributions-CNTK-dropout-0.5-aug', ...
  } ;
  modelNames = [vggFace2Models standardModels ...
                ferModels sfewModels ferPlusModels ferPlusModelsDev] ;
  msg = sprintf('%s: unrecognised model', modelName) ;
  assert(ismember(modelName, modelNames), msg) ;

  subfolder = '' ;
  if ismember(modelName, vggFace2Models)
    subfolder = 'vggface2_models' ;
  elseif ismember(modelName, ferPlusModelsDev)
    subfolder = fullfile('grimaces', modelName) ;
    fprintf('loading %s...\n', modelName) ;
    switch modelName
      case 'resnet50_ft-dag-dropout-0.1'
        modelName = 'net-epoch-17' ;
      case 'resnet50_ft-dag-dropout-0.5'
        modelName = 'net-epoch-122' ;
      case 'senet50_ft-dag-distributions-dropout-0.5-aug'
        modelName = 'net-epoch-98' ;
      case 'senet50_ft-dag-distributions-CNTK-dropout-0.5-aug'
        modelName = 'net-epoch-90' ;
      otherwise, error('msg: %s\n', modelName) ;
    end
  end
  modelDir = fullfile(vl_rootnn, 'data/models-import', subfolder) ;
  modelPath = fullfile(modelDir, sprintf('%s.mat', modelName)) ;
  if ~exist(modelPath, 'file')
    dag = emoVoxZoo(modelName) ;
  else
    net = load(modelPath) ;
    if isfield(net, 'net'), net = net.net ; end
    dag = dagnn.DagNN.loadobj(net) ;
  end

  if ismember(modelName, ferPlusModels) ...
        || contains(modelName, 'net-epoch') ...
        || ismember(modelName, ferModels)
    % previously trained model
    fprintf('\n---------------------------------------------------------\n') ;
    fprintf('Loading pretrained FER/FERplus model %s', modelName) ;
    fprintf('.... not re-initing params\n') ;
    fprintf('-----------------------------------------------------------\n') ;
    dag = fixInputVarnames(dag) ;
    pretrained = true ;
    return ;
  end

	dag = prepareFromDagNN(net, numOutputs) ;

  % configure the loss layers
  dag = configureForClassification(dag, opts.finetuneLR, ...
                                   opts.dropoout, opts.lossType) ;

  % modify the network to use Batch Normalization if needed
  if opts.useBnorm, dag = insertBNLayers(dag) ; end
  dag = fixInputVarnames(dag) ;

% ----------------------------------------------
function dag = fixInputVarnames(dag)
% ----------------------------------------------
  candidates = {'input', 'x0'} ;
  ins = dag.getInputs() ;
  for ii = 1:numel(ins)
    if ismember(ins{ii}, candidates), dag.renameVar(ins{ii}, 'data') ; end
  end

% ----------------------------------------------
function dag = prepareFromDagNN(net, numOutputs)
% ----------------------------------------------
%PREPAREFROMDAGNN prepares a DagNN structure for classification
%  PREPARESFROMIMPLENN(dag, numOutputs) prepares a DagNN
%  network for classification by removing any old loss layers
%  and ensuring that the final fully connected "prediction"
%  layer has the correct dimensions.

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
  fScale = 1/100 ;
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

% ----------------------------------------------------------------------------
function dag = configureForClassification(dag, finetuneLR, dropout, lossType)
% ----------------------------------------------------------------------------
%CONFIGUREFORCLASSIFICATION configures the network to train as a classifer
%  CONFIGUREFORCLASSIFICATION(dag) adds a softmaxlog loss
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
    %sel =  convLayers(end-3:end-2) ;
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

  % set the learning rates for fine tuning
  paramIdx = dag.getParamIndex([dag.layers(1:end-2).params])  ;
  [dag.params(paramIdx).learningRate] = deal(finetuneLR)  ;

  % Add softmaxlog loss layer (for training)
  switch lossType
    case 'softmaxlog'
      layer = dagnn.Loss('loss', 'softmaxlog') ;
    case 'distributions'
      layer = dagnn.SoftmaxCELoss() ;
    otherwise, error('unrecognised loss: %s\n', lossType) ;
  end
  inputs = {'prediction', 'label'} ;
  output = 'objective' ;
  dag.addLayer('loss', layer, inputs, output) ;

  % Add class error
  layer = dagnn.Loss('loss', 'classerror') ;
  inputs = {'prediction', 'hardlabel'} ;
  output = 'classerror' ;
  dag.addLayer('classerror', layer, inputs, output)  ;

  dag.rebuild() ;

  % modify the meta attributes of the net
  emotions = {'neutral', 'happiness', 'surprise', 'sadness', ...
                'anger', 'disgust', 'fear', 'contempt'} ;
  dag.meta.classes.name = emotions ;
  dag.meta.classes.description = emotions ;

% --------------------------------------------------
function net = insert_dropout(net, prev, next, rate)
% --------------------------------------------------
	in = net.layers(net.getLayerIndex(prev)).outputs ;
	lname = sprintf('%s_drop', prev) ; out = lname ;
	net.addLayer(lname, dagnn.DropOut('rate', rate), in, out, {}) ;
	net.setLayerInputs(next, {out}) ;
