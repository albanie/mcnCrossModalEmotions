function loadedImdb = fetch_emovoxceleb_imdb(teacher, varargin)
%FETCH_EMOVOXCELEB_IMDB - a helper function to avoid painful load times
%   LOADEDIMDB = FETCH_EMOVOXCELEB_IMDB(VARARGIN) is simply a helper
%   function that loads the predictions of a teacher network on the
%   VoxCeleb dataset.  The reason for this function is that the size of
%   the file containing the predictions as an imdb is fairly large and
%   it takes quite a long time to load from disk.  This function uses
%   global memory to keep the file cached to avoid slow reload times.
%
% Copyright (C) 2018 Samuel Albanie, Arsha Nagrani
% Licensed under The MIT License [see LICENSE.md for details]

  opts.imdbDir = fullfile(vl_rootnn, 'data/xEmo18/storedFeats') ;
  opts = vl_argparse(opts, varargin) ;

	global imdb ; % cache
  global config ; % validate the the correct imdb is being loaded

  if isequal(config, opts) && ~isempty(imdb)
		fprintf('found imdb in cache, re-using..\n') ;
  else
    imdbPath = getImdbPath(opts.imdbDir, teacher) ;
    fetchImdbFromInternet(imdbPath, 'emovoxceleb')
    if exist(imdbPath, 'file')
      fprintf('loading imdb ...\n') ; tic ;
      imdb = load(imdbPath) ;
      fprintf('done in %g s\n', toc) ;
    else
      fprintf('generating imdb (this will take a long time)...\n') ; tic ;
      imdb = buildImdb(teacher) ;
      if ~exist(opts.imdbDir, 'dir'), mkdir(opts.imdbDir) ; end
      save(imdbPath, '-struct', 'imdb') ;
      fprintf('done in %g s\n', toc) ;
    end

    fnames = fieldnames(opts) ;
    for ii = 1:numel(fnames)
      config.(fnames{ii}) = opts.(fnames{ii}) ;
    end
  end
  loadedImdb = imdb ;
end

% ------------------------------------------------------------------
function imdbPath = getImdbPath(imdbDir, teacher)
% ------------------------------------------------------------------
  template = '%s-logits' ;
	template = [template '.mat'] ;
	teacherLogits = sprintf(template, teacher) ;
	imdbPath = fullfile(imdbDir, teacherLogits) ;
end

% ------------------------------------------------------------------
function imdb = buildImdb(teacher, varargin)
% ------------------------------------------------------------------
%BUILDIMDB - store all teacher predictions in an imdb
%  IMDB = BUILDIMDB(TEACHER, VARARGIN) computes the output of the
%  teacher for almost all extracted frames of VoxCeleb (approx. 5 million
%  in total).  This will take a long time on a single GPU.

	opts.gpus = 3 ;
  opts.limit = inf ; % pick number smaller than inf for debugging
  batchSize = 128 ;
  numEmotions = 8 ;
  opts.srcImdb = fullfile(vl_rootnn, 'data/emoVoxCeleb/voxceleb-imdb.mat') ;
  opts.imdbPath = fullfile(vl_rootnn, 'data/xEmo18/dense_imdb/imdb.mat') ;
  opts.featPath = fullfile(vl_rootnn, ...
                           'data/xEmo18/featImdbs', teacher, 'imdb.mat') ;
  opts.faceDir = '/dev/shm/albanie/unzippedIntervalFaces' ;
  opts = vl_argparse(opts, varargin) ;

  [imdb, found] = dev_cache('imdbPlusDenseFrames') ;
  if ~found
    if exist(opts.imdbPath, 'file')
      fprintf('loading imdb from disk...') ; tic ;
      imdb = load(opts.imdbPath) ;
      fprintf('done in %g s\n', toc) ;
    else
      fprintf('loading src imdb...') ; tic ;
			if ~exist(opts.srcImdb, 'file')
        fprintf('VoxCeleb imdb not found...\n') ;
        fetchImdbFromInternet(opts.srcImdb, 'voxceleb')
			end
      imdb = load(opts.srcImdb) ;
      fprintf('done in %g s\n', toc) ;

      fprintf('adding frames to imdb...') ; tic ;
      imdb = addFramesToImdb(imdb, opts.faceDir) ;
      mkdir(fileparts(opts.imdbPath)) ;
      % try to avoid compression if poss here - the imdb should be approx
      % 1GB in memory
      save(opts.imdbPath, '-struct', 'imdb') ;
      fprintf('done in %g s\n', toc) ;
      dev_cache('imdbPlusDenseFrames', imdb) ;
    end
  end

  [dag, pretrained] = ferPlusZoo(teacher) ;
  assert(pretrained, 'loaded incorrect teacher model') ;

  % remove losses and configure network to predict emotions
  lossLayers = arrayfun(@(x) isa(x.block, 'dagnn.Loss'), dag.layers) ;
  removables = {dag.layers(lossLayers).name} ;
  for ii = 1:numel(removables)
    dag.removeLayer(removables{ii}) ;
  end
	dag.mode = 'test' ;
	if numel(opts.gpus), gpuDevice(opts.gpus) ; dag.move('gpu') ; end
	inVars = dag.getInputs() ;
	assert(numel(inVars) == 1, 'too many inputs') ;

  % compute for first `limit` tracks
  firstId = imdb.images.id(1) ;
  numKeep = sum(imdb.images.denseFramesWavIds <= firstId + opts.limit) ;
  numIms = min(numel(imdb.images.denseFrames), numKeep) ;
	inVars = dag.getInputs() ;
	assert(numel(inVars) == 1, 'too many inputs') ;

  logits = zeros(numIms, numEmotions, 'single') ;

	fprintf('processing images with %s\n', teacher) ;
	for ii = 1:batchSize:numIms
		tic ;
		batchStart = ii ;
		batchEnd = min(batchStart+batchSize-1, numIms) ;
		batch = batchStart:batchEnd ;
    imPaths = fullfile(opts.faceDir, imdb.images.denseFrames(batch)) ;
		data = getImageBatch(imPaths, dag) ;
		dag.eval({inVars{1}, data}) ;
		out = gather(squeeze(dag.vars(end).value)) ; % risky use of end variable
		logits(batch, :) = out' ;
		rate = numel(batch) / toc ;
    etaStr = zs_eta(rate, ii, numIms) ;
		fprintf('processed image %d/%d at (%.3f Hz) (%.3f%% complete) (eta:%s)\n', ...
					 ii, numIms, rate, 100 * ii/numIms, etaStr) ;
	end

  % save the logits alongside the wav files for easier processing
  % when training
  numWavs = numel(imdb.images.name) ;
  wavLogits = cell(1, numWavs) ;
  numLogits = min(numWavs, opts.limit) ;
  for ii = 1:numLogits
    fprintf('splitting logits into cells for wav %d/%d\n', ii, numLogits) ;
    idx = imdb.images.id(ii) ;
    wavLogits(ii) = {logits(imdb.images.denseFramesWavIds == idx,:)} ;
  end
  imdb.wavLogits = wavLogits ;
end

% --------------------------------------------------------------------------
function data = getImageBatch(imagePaths, dag)
% --------------------------------------------------------------------------
  opts.prefetch = false ; % can optimise this
  imageSize = dag.meta.normalization.imageSize(1:2) ;

  % we need to do a slightly odd rescaling/grayscale tx to match training
	% (we use bilinear to reproduce the resize operation used during training
  % as faithfully as possible)
  args = {imagePaths, ...
          'NumThreads', 10, ...
          'Pack', ...
          'Interpolation', 'bilinear', ...
          'CropSize', 1 / 1.6, ...
          'CropLocation', 'center', ...
          'Resize', imageSize} ;

  if opts.prefetch
    vl_imreadjpeg(args{:}, 'prefetch') ;
    data = [] ;
  else
    data = vl_imreadjpeg(args{:}) ;
  end

  % throw away color, since it was not used during training
	data = uint8(data{1}) ;
	data_ = permute(data, [1 2 4 3]) ; sz = size(data_) ;
	data_ = reshape(data_, sz(1), sz(2) * sz(3), 3) ;
	gray = rgb2gray(data_) ;
	gray = single(reshape(gray, sz(1), sz(2), 1, [])) ;
	data = normalizeFace(gray, dag) ;
	if strcmp(dag.device, 'gpu'), data = gpuArray(data) ; end
end

% --------------------------------------------------------------------
function face = normalizeFace(greyFace, dag)
% --------------------------------------------------------------------
% Normalizes the network (done differently for different models)
  greyFace = imresize(greyFace, dag.meta.normalization.imageSize(1:2)) ;
  face = repmat(greyFace, [1 1 3]) ;
  avgIm = reshape(dag.meta.normalization.averageImage, 1, 1, 3) ;
  face = bsxfun(@minus, face, avgIm) ;
end

% ----------------------------------------------------------------------
function imdb = addFramesToImdb(imdb, faceDir)
% ----------------------------------------------------------------------

  [numIms, found] = dev_cache('numIms') ;
  if ~found
    % The `find` call is purely used to compute the number of images as a
    % sanity check and to avoid running out of memory later.  After some
    % trial and error, it seems that calling the system find command is
    % about an order of magnitude faster than using MATLAB to find the
    % files (there are approx 5 mil files in total)

    % note the trailing slash in the command (it will fail without a trailing
    % slash, but will be fine with two slashes)
    cmd = sprintf('find %s/ -name "*.jpg"', faceDir) ;
    fprintf('finding image paths...\n') ; tic ;
    [status, out] = system(cmd) ;
    faceUrl = ['http://www.robots.ox.ac.uk/~vgg/research/CMBiometrics/data' ...
                '/dense-face-frames.tar.gz'] ;
    msg = ['find command failed, did you unpack the face images into %s ?' ...
           '\n (they can be found at %s)'] ;
    assert(status == 0, sprintf(msg, faceDir, faceUrl)) ;
    fprintf('done in %g (s)\n', toc) ;

    fprintf('finished find call, splitting into paths...') ; tic ;
    imPaths = strsplit(out, '\n') ;
    if isempty(imPaths{end}), imPaths(end) = [] ; end % drop last cell if empty
    numIms = numel(imPaths) ;
    assert(numIms == 5078961, 'unexpected number of face images') ;
    fprintf('done in %g (s)\n', toc) ;
    dev_cache('numIms', numIms) ;
  end

  % We now arrange frames in cells such that each cell corresponds to one
  % audio clip. Note that since the frames have been extracted at a lower
  % frame rate than was used in the original dataset, there are now a small
  % number of audio tracks that do not have any corresponding face frames.
  % There are also a small number of frames that are not registered to
  % tracks (there 1217 such frames).  Both are dropped for the distillation
  % process.
  framePaths = cell(numIms, 1) ;
  wavIds = zeros(numIms, 1, 'single') ;
  numWavs = numel(imdb.images.name) ;
  offset = 1 ;
  for ii = 1:numWavs
    tic ;
    wavPath = imdb.images.name{ii} ;
    video = imdb.images.video{ii} ;
    track = imdb.images.track(ii) ;
    tokens = strsplit(imdb.images.name{ii}, '/') ;
    celeb = tokens{1} ;
    frameDir = fullfile(faceDir, celeb, '1.6', video, num2str(track)) ;
    frames = zs_getImgsInDir(frameDir, 'jpg') ;
    numTrackFrames = numel(frames) ;
    wavIds(offset:offset+numTrackFrames-1) = ii ;
    framePaths(offset:offset+numTrackFrames-1) = frames ;
    offset = offset + numTrackFrames ;
    if numTrackFrames == 0
      fprintf('no frames found for wavfile...%s\n', wavPath) ;
    end
    rate = 1 / toc ;
    etaStr = zs_eta(rate, ii, numWavs) ;
    fprintf('processed wav %d/%d at (%.3f Hz) (%.3f%% complete) (eta:%s)\n', ...
               ii, numWavs, rate, 100 * ii/numWavs, etaStr) ;
  end

  % drop tracks without frames and remove completely from imdb
  wavsWithFrames = unique(wavIds) ;
  allWavIds = 1:numWavs ;
  dropWavs = ~ismember(allWavIds, wavsWithFrames) ; % 134 of these
  fnames = fieldnames(imdb.images) ;
  for ii = 1:numel(fnames)
    imdb.images.(fnames{ii})(dropWavs) = [] ;
  end

  % drop "unclaimed" frames
  dropUnclaimed = (wavIds == 0) ;
  wavIds(dropUnclaimed) = [] ;
  framePaths(dropUnclaimed) = [] ;

  % finally, we ensure that the paths are relative
  if ~strcmp(faceDir(end), '/')
    faceDir = [faceDir '/'] ; % make certain that there is a trailing slash
  end
  % subtract absolute path to faceDir from all paths
  framePathsRelative = cellfun(@(x) {erase(x, faceDir)}, framePaths) ;

  % store corresponding frames on the imdb object
  imdb.images.denseFrames = framePathsRelative ;
  imdb.images.denseFramesWavIds = wavIds ;
end

% -------------------------------------------------
function fetchImdbFromInternet(destPath, imdbName)
% -------------------------------------------------

  waiting = true ;
  prompt = sprintf(...
        strcat('VoxCeleb Imdb was not found at %s\nWould you like to ', ...
        ' download it from THE INTERNET (y/n)?\n'), destPath) ;

  baseUrl = 'http://www.robots.ox.ac.uk/~albanie/data/cross-modal-emotions' ;

  switch imdbName
    case 'voxceleb'
      url = [baseUrl '/voxceleb-imdb.mat'] ;
    case 'emovoxceleb'
      url = [baseUrl '/senet50-ferplus-logits.mat'] ;
    otherwise
      error('did not recognise imdb name %s\n', imdbName) ;
  end

  if exist(destPath, 'file')
    fprintf('file already exists at destination, skipping...\n') ;
    return
  end

  while waiting
    str = input(prompt,'s') ;
    switch str
      case 'y'
        if ~exist(fileparts(destPath), 'dir'), mkdir(fileparts(destPath)) ; end
        fprintf('VoxCelebImdb ... \n') ;
        urlwrite(url, destPath) ;
        return ;
      case 'n', throw(exception) ;
      otherwise, fprintf('input %s not recognised, please use `y/n`\n', str) ;
    end
  end
end
