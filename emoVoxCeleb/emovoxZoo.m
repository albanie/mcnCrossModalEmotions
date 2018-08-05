function dag = emovoxZoo(modelName, varargin)
%EMOVOXZOO - load EmoVoxCeleb model by name
%   DAG = EMOVOXZOO(MODELNAME, VARARGIN) - loads an
%   emotion recognition CNN given its name, MODELNAME.
%
%   EMOVOXZOO(..'name', value) accepts the following options:
%
%   `modelDir` :: fullfile(vl_rootnn, 'data/models-import', subfolder)
%    Directory to search for models
%
% Copyright (C) 2018 Samuel Albanie, Arsha Nagrani
% Licensed under The MIT License [see LICENSE.md for details]

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
	dag = dagnn.DagNN.loadobj(load(modelPath)) ;
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
