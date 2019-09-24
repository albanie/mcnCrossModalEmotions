function destPath = compute_audio_feats(destPath, varargin)
%COMPUTE_AUDIO_FEATS - compute audio features
%  DESTPATH = COMPUTE_AUDIO_FEATS(DESTPATH, VARARGIN) computes the features
%  produced by an audio model and stores them to DESTPATH.
%
%   EMO_BENCHMARKS(..'name', value) accepts the following
%   options:
%
%   `gpus` :: 1
%    The gpu device to use for processing.
%
%   `limit` :: inf
%    Limit the number of samples processed (only used for debugging).
%
%   `teacher` :: ''
%    The name of the teacher model (only required for computing features
%    on EmoVoxCeleb).
%
%   `student` :: ''
%    The name of the audio model to be evaluated.
%
%   `clobber` :: false
%    Overwrite existing cached features if already present.
%
%   `numEmotions` :: 8
%    The number of emotions predicted by the audio model.
%
%   `targetDataset` :: 'rml'
%    The dataset for which the features will be computed.
%
% Copyright (C) 2018 Samuel Albanie, Arsha Nagrani
% Licensed under The MIT License [see LICENSE.md for details]

	opts.gpus = 1 ;
  opts.limit = inf ;
  opts.teacher = '' ;
  opts.modelName = '' ;
  opts.clobber = false ;
  opts.numEmotions = 8 ;
  opts.targetDataset = 'rml' ;
  opts = vl_argparse(opts, varargin) ;

  assert(~isempty(opts.modelName), 'a model must be specified') ;

	buckets.pool = [2 5 8 11 14 17 20 23 27 30] ;
	buckets.width  = [100 200 300 400 500 600 700 800 900 1000] ;

  opts.imdbPath = fullfile(vl_rootnn, ...
     sprintf('data/xEmo18/%s_imdb/imdb.mat', opts.targetDataset)) ;

  if ~strcmp(opts.modelName, 'random')
		dag = emoVoxZoo(opts.modelName) ;
	end

  if exist(destPath, 'file') && ~opts.clobber
    fprintf('found features at %s... skipping\n', destPath) ;
    return
  end

  switch opts.targetDataset % NOTE: These need to be setup for your system
    case 'afew'
      opts.dataDir = fullfile(vl_rootnn, 'data/datasets/emotiw2016') ;
      getImdb = @(x) getAfewImdb(x, 'dropTracksWithNoDets', 1) ;
    case 'afew-6' % extracted at a stride of 6
      opts.dataDir = fullfile(vl_rootnn, 'data/datasets/emotiw2016') ;
      getImdb = @(x) getAfewImdb(x, 'dropTracksWithNoDets', 1, ...
                                     'subsampleStride', 6) ;
    case 'enterface'
      opts.dataDir = fullfile(vl_rootnn, 'data/datasets/enterface') ;
      getImdb = @(x) getEnterfaceImdb(x) ;
    case 'rml'
      opts.dataDir = fullfile(vl_rootnn, 'data/datasets/rml') ;
      getImdb = @(x) getRmlImdb(x) ;
    case 'emovoxceleb'
      dataDir = fullfile(vl_rootnn, 'data/datasets/voxceleb1/voxceleb_all') ;
      getImdb = @(x) fetch_mod_emovoxceleb_imdb(opts.teacher, dataDir) ;
    otherwise, error('unknown dataset %s\n', opts.targetDataset) ;
  end

  if exist(opts.imdbPath, 'file')
    imdb = fetchExternalImdb(opts.imdbPath) ;
  else
    fprintf('building imdb...') ; tic ;
    imdb = getImdb(opts) ;
    mkdir(fileparts(opts.imdbPath)) ;
    save(opts.imdbPath, '-struct', 'imdb') ; % avoid compression if poss
    fprintf('done in %g s\n', toc) ;
  end

  % compute for first `limit` tracks
  firstId = imdb.tracks.id(1) ; % use first in partition as offset
  numKeep = sum(imdb.tracks.id <= firstId + opts.limit) ; % number of tracks
  numIms = numKeep ;

  if strcmp(opts.modelName, 'random')
    logits = randn(numIms, opts.numEmotions, 'single') ;
    storeAudioLogitImdb(imdb, logits, numIms, destPath) ;
    return
  end

  % remove losses
  lossLayers = arrayfun(@(x) isa(x.block, 'dagnn.Loss'), dag.layers) ;
  removables = {dag.layers(lossLayers).name} ;
  for ii = 1:numel(removables)
    dag.removeLayer(removables{ii}) ;
  end
	dag.mode = 'test' ;
	if numel(opts.gpus), gpuDevice(opts.gpus) ; dag.move('gpu') ; end

	inVars = dag.getInputs() ;
	assert(numel(inVars) == 1, 'too many inputs') ;

  logits = zeros(numIms, opts.numEmotions, 'single') ;

	fprintf('processing images with %s\n', opts.modelName) ;
	for ii = 1:numIms
		tic ;
		wavPath = imdb.tracks.wavPaths{ii} ;
		inp1 = test_getinput(wavPath, buckets) ;
		s1 = size(inp1, 2) ;
		p1 = buckets.pool(s1 == buckets.width) ;
		ind1 = dag.getLayerIndex('pool6') ;

    if size(inp1, 2) > 0
      dag.layers(ind1).block.poolSize=[1 p1] ;
      dag.eval({'data', gpuArray(inp1)}) ;
      out = gather(squeeze(dag.vars(end).value)) ; % risky use of end variable
      logits(ii, :) = out' ;
      rate = 1 / toc ;
      etaStr = zs_eta(rate, ii, numIms) ;
      fprintf('processed image %d/%d at (%.3f Hz) (%.3f%% complete) (eta:%s)\n', ...
      			 ii, numIms, rate, 100 * ii/numIms, etaStr) ;
     else
       fprintf('empty audio clip\n') ; keyboard
     end
	end
  storeAudioLogitImdb(imdb, logits, numIms, destPath) ;
	if numel(opts.gpus), dag.move('cpu') ; end
end

% ------------------------------------------------------------------------
function storeAudioLogitImdb(imdb, logits, numIms, destPath)
% ------------------------------------------------------------------------
  % use the term faceLogit, even for stored audio
  faceLogits = cell(1, numIms) ;
  for ii = 1:numIms
    fprintf('splitting logits into cells (%d/%d)\n', ii, numIms) ;
    faceLogits(ii) = {logits(ii,:)} ;
  end
  imdb.faceLogits = faceLogits ;
	fprintf('saving features to %s...', destPath) ; tic ;
	if ~exist(fileparts(destPath), 'dir')
		zs_mkdirRec(fileparts(destPath)) ;
	end
	save(destPath, '-struct', 'imdb') ;
	fprintf('done in %g (s)\n', toc) ;
end

% --------------------------------------------------
function inp = test_getinput(image, buckets)
% --------------------------------------------------
  audio.window   = [0 1];
  audio.fs       = 16000;
  audio.Tw       = 25;
  audio.Ts       = 10;            % analysis frame shift (ms)
  audio.alpha    = 0.97;          % preemphasis coefficient
  audio.R        = [300 3700];  % frequency range to consider
  audio.M        = 40;            % number of filterbank channels
  audio.C        = 13;            % number of cepstral coefficients
  audio.L        = 22;            % cepstral sine lifter parameter

	audfile = [image(1:end-3),'wav'] ;
	z	= audioread(audfile) ;
  assert(size(z,2) <= 2, 'unexpected number of streams') ;
  z = z(:,1) ; % take left stream if stereo is available
	SPEC = runSpec(z, audio) ;
	mu = mean(SPEC, 2) ;
  stdev	= std(SPEC, [], 2) ;
  nSPEC	= bsxfun(@minus, SPEC, mu) ;
  nSPEC	= bsxfun(@rdivide, nSPEC, stdev) ;
  rsize	= buckets.width(find(buckets.width(:)<=size(nSPEC,2),1,'last')) ;
  rstart = round((size(nSPEC, 2) - rsize) / 2) ;
  if rstart == 0, rstart = 1 ; end
	inp(:,:) = gpuArray(single(nSPEC(:,rstart:rstart+rsize-1))) ;
end

% -----------------------------------------------------------------------------
function imdb = fetch_mod_emovoxceleb_imdb(teacher, dataDir)
% -----------------------------------------------------------------------------
%FETCH_MOD_EMOVOXCELEB_IMDB fetch a modified version of EmoVoxCeleb imdb
%  IMDB = FETCH_MOD_EMOVOXCELEB_IMDB(TEACHER) fetches a version of the
%  the EmoVoxCeleb that is compatible with audio feature construction.
    assert(~strcmp(teacher, ''), 'teacher must be set') ;
    imdb = fetch_emovoxceleb_imdb(teacher) ; % fetch original

    % conform to the other IMDB interfaces
    imdb.tracks.id = imdb.images.id ;
    imdb.tracks.wavPaths = fullfile(dataDir, imdb.images.name) ;
end

