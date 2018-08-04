function setup_mcnCrossModalEmotions()
%SETUP_MCNCROSSMODALEMOTIONS Sets up mcnCrossModalEmotions, by adding
% its folders to the Matlab path
%
% Copyright (C) 2018 Samuel Albanie, Arsha Nagrani
% Licensed under The MIT License [see LICENSE.md for details]

  root = fileparts(mfilename('fullpath')) ;
  addpath(root, [root '/emoVoxCeleb']) ;
