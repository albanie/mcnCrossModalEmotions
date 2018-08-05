function destPath = compute_visual_feats(destPath, varargin)

	opts.gpus = 2 ;
  opts.limit = inf ;
  opts.clobber = false ;
  opts.batchSize = 128 ;
  opts.numEmotions = 8 ;
  opts.targetDataset = 'rml' ;
  opts.modelName = 'senet50-ferplus' ;
  opts = vl_argparse(opts, varargin) ;
  opts.imdbPath = fullfile(vl_rootnn, ...
             sprintf('data/xEmo18/%s_imdb/imdb.mat', opts.targetDataset)) ;

  if exist(destPath, 'file') && ~opts.clobber
    fprintf('found features at %s... skipping\n', destPath) ;
    return
  end

  switch opts.targetDataset
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
    otherwise, error('unknown dataset %s\n', opts.targetDataset) ;
  end

  % work relative to audio files
  if exist(opts.imdbPath, 'file')
    fprintf('loading imdb from memory...') ; tic ;
    imdb = load(opts.imdbPath) ;
    fprintf('done in %g s\n', toc) ;
  else
    fprintf('building imdb...') ; tic ;
    imdb = getImdb(opts) ;
    mkdir(fileparts(opts.imdbPath)) ;
    save(opts.imdbPath, '-struct', 'imdb') ; % avoid compression if poss
    fprintf('done in %g s\n', toc) ;
  end

  [dag, pretrained] = ferPlusZoo(opts.modelName) ;
  assert(pretrained, 'loaded incorrect model') ;

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

  % flatten frames into list, and keep track of the mapping to cells
  frames = zs_flattenCell(imdb.tracks.paths) ;
  frameIdx = arrayfun(@(x) ...
                     {x*ones(1, numel(imdb.tracks.paths{x}))}, ...
                     1:numel(imdb.tracks.paths)) ;
  frameIdx = [frameIdx{:}] ;
  assert(numel(frames) == numel(frameIdx), 'incorrect idx pairing') ;

  % compute for first `limit` tracks
  firstId = frameIdx(1) ; % use first in partition as offset
  numKeep = sum(frameIdx <= firstId + opts.limit) ; % number of tracks
  numIms = min(numel(frames), numKeep) ;
  meta = dag.meta ;
	inVars = dag.getInputs() ;
	assert(numel(inVars) == 1, 'too many inputs') ;

  logits = zeros(numIms, opts.numEmotions, 'single') ;

	fprintf('processing images with %s\n', opts.modelName) ;
	for ii = 1:opts.batchSize:numIms
		tic ;
		batchStart = ii ;
		batchEnd = min(batchStart+opts.batchSize-1, numIms) ;
		batch = batchStart:batchEnd ;
    imPaths = fullfile(opts.dataDir, frames(batch)) ;
		data = getImageBatch(imPaths, dag, meta) ;
		dag.eval({inVars{1}, data}) ;
		out = gather(squeeze(dag.vars(end).value)) ; % risky use of end variable
		logits(batch, :) = out' ;
		rate = numel(batch) / toc ;
    etaStr = zs_eta(rate, ii, numIms) ;
		fprintf(['processed image %d/%d at (%.3f Hz) ' ...
             '(%.3f%% complete) (eta:%s)\n'], ...
             ii, numIms, rate, 100 * ii/numIms, etaStr) ;
	end

  % save the logits alongside the frames for easier processing during training
  keptIdx = frameIdx(1:numIms) ; msg = 'unexpected indices' ;
  assert(isequal(1:numel(unique(keptIdx)), unique(keptIdx)), msg) ;
  numKeptTracks = numel(unique(keptIdx)) ;

  faceLogits = cell(1, numKeptTracks) ;
  for ii = 1:numKeptTracks
    fprintf('splitting logits into cells (%d/%d)\n', ii, numKeptTracks) ;
    sel = frameIdx == ii ;
    faceLogits(ii) = {logits(sel,:)} ;
  end

	fprintf('saving features to %s...', destPath) ; tic ;
  imdb.faceLogits = faceLogits ;
	if ~exist(fileparts(destPath), 'dir')
		zs_mkdirRec(fileparts(destPath)) ;
	end
	save(destPath, '-struct', 'imdb') ;
	fprintf('done in %g (s)\n', toc) ;
	if numel(opts.gpus), dag.move('cpu') ; end
end

% --------------------------------------------------------
function data = getImageBatch(imagePaths, dag, meta)
% --------------------------------------------------------
  numThreads = 10 ;
  opts.prefetch = false ; % can optimise this
  imageSize = meta.normalization.imageSize(1:2) ;

  % Afew is already tightly cropped
  args{1} = {imagePaths, ...
             'NumThreads', numThreads, ...
             'Pack', ...
             'Interpolation', 'bilinear', ... % bilinear to reproduce trainig
             'Resize', imageSize, ...
             'CropLocation', 'center'} ; % centre crop for testing
  args = horzcat(args{:}) ;

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
  greyFace = imresize(greyFace, dag.meta.normalization.imageSize(1:2)) ;
  face = repmat(greyFace, [1 1 3]) ;

  % handle average image consisting of color channel averages
  avgIm = reshape(dag.meta.normalization.averageImage, 1, 1, 3) ;
  face = bsxfun(@minus, face, avgIm) ;
end
