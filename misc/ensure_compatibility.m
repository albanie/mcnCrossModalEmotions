function ensure_compatibility(varargin)
%ENSURE_COMPATIBILITY - fix public models
%   ENSURE_COMPATIBILITY modifies the public emotion recognition models
%   to make sure that they are backwards compatible with older versions
%   of MatConvNet.
%
% Copyright (C) 2018 Samuel Albanie
% Licensed under The MIT License [see LICENSE.md for details]

   opts.modelDir = fullfile(vl_rootnn, 'data/models-import') ;
   opts = vl_argparse(opts, varargin) ;

   publicModels = {'resnet50-ferplus', ...
                   'senet50-ferplus', ...
                   'emovoxceleb-student'} ;

   for ii = 1:numel(publicModels)
     modelName = publicModels{ii} ;
     fprintf('fixing %s ...\n', modelName) ;
     modelPath = fullfile(opts.modelDir, sprintf('%s.mat', modelName)) ;
     net = load(modelPath) ;
     if isfield(net, 'net'), net = net.net ; end
     for jj = 1:numel(net.layers)
       if isfield(net.layers(jj), 'block')
         if isfield(net.layers(jj).block, 'exBackprop')
           net.layers(jj).block = rmfield(net.layers(jj).block, 'exBackprop') ;
         end
       end
     end
     fprintf('saving compatible model to %s\n', modelPath) ;
     save(modelPath, '-struct', 'net') ;
   end
