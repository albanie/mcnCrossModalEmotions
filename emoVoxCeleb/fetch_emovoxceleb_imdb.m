function loadedImdb = fetch_emovoxceleb_imdb(teacher, varargin)
%FETCH_EMOVOXCELEB_IMDB - a helper function to avoid painful load times
%   LOADEDIMDB = FETCH_EMOVOXCELEB_IMDB(VARARGIN) is simply a helper
%   function that loads the predictions of a teacher network on the
%   VoxCeleb dataset.  The reason for this function is that the size of
%   the file containing the predictions as an imdb is fairly large and
%   it takes quite a long time to load from disk.  This function uses
%   global memory to keep the file cached to avoid slow reload times.

  opts.imdbDir = fullfile(vl_rootnn, 'data/xEmo18/storedFeats') ;
  opts = vl_argparse(opts, varargin) ;

	global imdb ; % cache
  global config ; % validate the the correct imdb is being loaded

  if isequal(config, opts) && ~isempty(imdb)
		fprintf('found imdb in cache, re-using..\n') ;
  else
    imdbPath = getImdbPath(opts.imdbDir, teacher) ;
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
	template = '%s-logits-combined-tagged' ;
	template = [template '.mat'] ;
	teacherLogits = sprintf(template, teacher) ;
	imdbPath = fullfile(imdbDir, teacherLogits) ;
end

% ------------------------------------------------------------------
function imdb = buildImdb(teacher, varargin)
% ------------------------------------------------------------------
	opts.gpus = 1 ;
  batchSize = 128 ;
  opts.limit = inf ;
  numEmotions = 8 ;
  opts.srcImdb = fullfile(vl_rootnn, 'data/voxceleb/imdb.mat') ;
  opts.imdbPath = fullfile(vl_rootnn, 'data/xEmo18/dense_imdb/imdb.mat') ;
  opts.featPath = fullfile(vl_rootnn, ...
                           'data/xEmo18/featImdbs', teacher, 'imdb.mat') ;
  opts.faceDir = '/dev/shm/albanie/unzippedIntervalFaces' ;
  opts = vl_argparse(opts, varargin) ;


  % work relative to audio files
  %if ~exist('imdb', 'var')
    if exist(opts.imdbPath, 'file')
      fprintf('loading imdb from memory...') ; tic ;
      imdb = load(opts.imdbPath) ;
      fprintf('done in %g s\n', toc) ;
    else
      fprintf('loading src imdb...') ; tic ;
      imdb = load(opts.srcImdb) ;
      fprintf('done in %g s\n', toc) ;

      fprintf('adding frames to imdb...') ; tic ;
      imdb = addFramesToImdb(imdb, opts.faceDir) ;
      mkdir(fileparts(opts.imdbPath)) ;
      save(opts.imdbPath, '-struct', 'imdb') ; % do not use compression if poss
      fprintf('done in %g s\n', toc) ;
    end
  %end
  keyboard

  numWavs = numel(imdb.images.name) ;
  paritionSize = round(numWavs / 4) ;
  % partition based on expId, keeping it basic
  keep1 = 1:paritionSize ;
  keep2 = paritionSize+1:2*paritionSize ;
  keep3 = 2*paritionSize+1:3*paritionSize ;
  keep4 = 3*paritionSize+1:numWavs ;
  kept = [keep1 keep2 keep3 keep4] ;
  assert(numel(unique(kept)) == numel(kept), 'repeated idx') ;
  assert(numel(kept) == numWavs, 'missing idx') ;

  switch expId
    case 1, keep = keep1 ;
    case 2, keep = keep2 ;
    case 3, keep = keep3 ;
    case 4, keep = keep4 ;
    otherwise, error('unexpected id: %d\n', expId) ;
  end

  % extract partitionImdb
  fnames = {'name', 'id', 'set', 'sp', 'video', 'track'} ;
  for ii = 1:numel(fnames)
    fname = fnames{ii} ;
    partitionImdb.images.(fname) = imdb.images.(fname)(keep) ;
  end
  fnames = {'denseFrames', 'denseFramesWavIds'} ;
  for ii = 1:numel(fnames)
    fname = fnames{ii} ;
    frameKeep = ismember(imdb.images.denseFramesWavIds, keep) ;
    partitionImdb.images.(fname) = imdb.images.(fname)(frameKeep) ;
  end

  if ~exist('dag', 'var')
		[dag, pretrained] = grimaces_zoo(opts.modelName, 0, 0, 0) ;
    assert(pretrained, 'loaded incorrect model') ;
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

  % compute for first `limit` tracks
  firstId = partitionImdb.images.id(1) ; % use first in partition as offset
  numKeep = sum(partitionImdb.images.denseFramesWavIds <= firstId + opts.limit) ;
  numIms = min(numel(partitionImdb.images.denseFrames), numKeep) ;
	inVars = dag.getInputs() ;
	assert(numel(inVars) == 1, 'too many inputs') ;

  logits = zeros(numIms, numEmotions, 'single') ;

	fprintf('processing images with %s\n', opts.modelName) ;
	for ii = 1:batchSize:numIms
		tic ;
		batchStart = ii ;
		batchEnd = min(batchStart+batchSize-1, numIms) ;
		batch = batchStart:batchEnd ;
    imPaths = fullfile(opts.faceDir, partitionImdb.images.denseFrames(batch)) ;
		data = getImageBatch(imPaths, dag) ;
		dag.eval({inVars{1}, data}) ;
		out = gather(squeeze(dag.vars(end).value)) ; % risky use of end variable
		logits(batch, :) = out' ;
		rate = numel(batch) / toc ;
    etaStr = zs_eta(rate, ii, numIms) ;
		fprintf('processed image %d/%d at (%.3f Hz) (%.3f%% complete) (eta:%s)\n', ...
					 ii, numIms, rate, 100 * ii/numIms, etaStr) ;
	end

  % save the logits alongside the wav files for easier processing during training
  numWavs = numel(partitionImdb.images.name) ;
  wavLogits = cell(1, numWavs) ;
  numLogits = min(numWavs, opts.limit) ;
  for ii = 1:numLogits
    fprintf('splitting logits into cells for wav %d/%d\n', ii, numLogits) ;
    idx = partitionImdb.images.id(ii) ;
    sel = find(partitionImdb.images.denseFramesWavIds == idx) ;
    wavLogits(ii) = {logits(sel,:)} ;
  end

  partitionImdb.wavLogits = wavLogits ;
  imdb = partitionImdb ;
	%fprintf('saving features to %s...', destPath) ; tic ;
	%if ~exist(fileparts(destPath), 'dir')
		%zs_mkdirRec(fileparts(destPath)) ;
	%end
	%save(destPath, '-struct', 'partitionImdb', '-v7.3') ;
	%fprintf('done in %g (s)\n', toc) ;
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
    assert(status == 0, 'find command failed...') ;
    fprintf('done in %g (s)\n', toc) ;

    fprintf('finished find call, splitting into paths...') ; tic ;
    imPaths = strsplit(out, '\n') ;
    if isempty(imPaths{end}), imPaths(end) = [] ; end % drop last cell if empty
    numIms = numel(imPaths) ;
    assert(numIms == 5078961, 'unexpected number of face images') ;
    fprintf('done in %g (s)\n', toc) ;
    dev_cache('numIms', numIms) ;
  end
  keyboard

  % need to order frames to match tracks
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

  % drop empty frames
  drop = (wavIds == 0) ;
  wavIds(drop) = [] ;
  framePaths(drop) = [] ;
end
