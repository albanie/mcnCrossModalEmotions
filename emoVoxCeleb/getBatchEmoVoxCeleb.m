function inputs = getBatchEmoVoxCeleb(opts, useGpu, imdb, batch)
%GETBATCHEMOVOXCELEB get a batch of data from EmoVoxCeleb
%   INPUTS = GETBATCHEMOVOXCELEB(OPTS, USEGPU, IMDB, BATCH) fetches
%   the samples from EmoVoxCeleb (found in IMDB) corresponding to the
%   indices specified in BATCH. USEGPU is used to denote whether the
%   audio samples should be loaded onto the GPU.  OPTS is a structure
%   with settings for the preprocessing.
%
% Copyright (C) 2018 Samuel Albanie, Arsha Nagrani
% Licensed under The MIT License [see LICENSE.md for details]

  images = imdb.images.name(batch) ;
  logits = imdb.wavLogits(batch) ;
  isVal = ~isempty(batch) && imdb.images.set(batch(1)) ~= 1  ;
  timeOffsets = [] ;

  if ~isVal % training
    [im,lgo] = cnn_get_batch_wav_emo(images, imdb.wavDir, opts, logits, ...
                                timeOffsets, ...
                                'prefetch', nargout == 0)  ;
  else % validation: disable data augmentation
    [im,lgo] = cnn_get_batch_wav_emo(images, imdb.wavDir, opts, logits, ...
                                timeOffsets, ...
                                'prefetch', nargout == 0, ...
                                'transformation', ['v' opts.transformation])  ;
  end

  if nargout > 0
    if useGpu, im = gpuArray(im) ; end
    lgo = lgo(:,:,1:opts.numPredEmotions,:) ; % select relevant emotions
    inputs = {'data', im} ;
    [~, maxLabel] = max(lgo, [], 3) ;
    switch opts.lossType
      case 'softmaxlog'
        inputs = [inputs {'maxLabel', maxLabel}] ;
      case 'euclidean'
        weights = ones(1, 1, 1, numel(batch)) ; % no re-weighting required
        inputs = [inputs {'logitTarget', lgo, 'instanceWeights', ...
                                weights, 'maxLabel', maxLabel}] ;
      case 'hot-cross-ent'
        inputs = [inputs {'logitTarget', lgo, 'maxLabel', maxLabel}] ;
      otherwise, error('unrecognised loss type: %s\n', opts.lossType) ;
    end
  end

% ---------------------------------------------------------------------------
function [imo, lgo] = cnn_get_batch_wav_emo(images, wavDir, opts, logits, ...
                                            timeOffsets, varargin)
% ---------------------------------------------------------------------------

% opts.imageSize(2) should be 400 to sample a 4s long audio sample randomly
% the size of cell arrays images and logits should be the same
% wavDir should contain the path to all the wav files

% RETURNS: imo: 512 x 400 x 1 x batchsize
%          lgo: 1 x 1 x 8 x batchsize

  %varargin{1}         = rmfield(varargin{1},'meta');
  opts.prefetch = false ;
  opts = vl_argparse(opts, varargin) ;
  meta = opts.meta ;

  [chspeed, inputnorm, noisy] = findSettings(opts.transformation, opts) ;

  % compute (expected) audio time in seconds. If an audio track is shorter
  % than the desired time, it can be padded.
  audTime = 0.01 * opts.imageSize(2) + 0.001 * meta.audio.Tw - 0.001 ;
  audSamp = audTime * meta.audio.fs ;

  % preallocate
  im = cell(1, numel(images)) ;
  lg_sampled = cell(1, numel(images)) ;


  %% ----- Fetch wav files ("images") -----
  for ii = 1: numel(images)

    audfile = fullfile(wavDir,[images{ii}(1:end-3),'wav']) ;
    info  = audioinfo(audfile) ;

    if ~opts.fixedSegments
      % no clip in our version of the dataset is longer than this many seconds,
      % so we treat the audio accordingly
      DATASET_LIMIT = 19.9 ;

      % threshold the duration of the audio
      maxSamples = info.SampleRate * DATASET_LIMIT ;
      info.TotalSamples = min(maxSamples, info.TotalSamples) ;
    end

    if opts.fixedSegments
      wr = timeOffsets(ii) * meta.audio.fs + 1 ;
      wend = wr + audSamp -1 ;
      if wend > info.TotalSamples
        % occasionally padding is required - should only affect a small number
        % of samples
        tmp  = audioread(audfile, [wr info.TotalSamples]) ;
        z = vertcat(tmp, zeros(audSamp - numel(tmp), 1)) ; % pad with zeros
      else
        z  = audioread(audfile,[wr wend]) ;
      end
    elseif chspeed
      speedR    = 0.95 + rand(1) * 0.1 ;
      audSampR  = round(audSamp * speedR) ;
      wd        = info.TotalSamples-audSampR ;
      wr        = randi(wd) ;
      zo        = audioread(audfile,[wr wr+audSampR-1]) ;
      z         = resample(zo,round(meta.audio.fs/speedR),meta.audio.fs) ;
    else
      % random crop coord and read audio
      wd = info.TotalSamples - audSamp ;
      if wd >= 1 % standard case, uniformly pick random offfset
        wr = randi(wd) ;
        z  = audioread(audfile,[wr wr+audSamp-1]) ;
      else % handle cases when the audio is shorter than the desired length
        wr = 1 ; % required for starttime computation
        tmp  = audioread(audfile) ;
        z = vertcat(tmp, zeros(audSamp - numel(tmp), 1)) ; % pad with zeros
      end
    end

    if noisy
      % read noise
      Nir = randi(meta.noise.noisenum) ;
      Nwr = randi(meta.noise.noiselen-numel(z)) ;
      y = audioread(sprintf('%s/%02d.wav',meta.noise.noisedir,Nir),[Nwr Nwr+numel(z)-1]) ;

      % mix ratio
      Nratio = rand(1) * meta.noise.noisevol ;
      z = (z+y.*Nratio) ;
    end

    %get logits
    lgts = logits{ii} ;

    if opts.fixedSegments
      lgts_sampled = lgts ;
    else
      % the randomly selected start and end time for the spectrogram, note
      % these are in seconds
      starttime = wr / meta.audio.fs ;
      endtime = (wr + audSamp - 1) / meta.audio.fs ;

      % get mapping from seconds to logit space
      startIdx = time2idx(starttime) ;
      endIdx = time2idx(endtime) ;


      % Note that if ffmpeg did not produce the exact number of specified frames,
      % we may oversample from the pre-computed logits.  A simple solution
      % is to limit the range of logits to be sampled to lie in the precomputed
      % array.  In practice, this may mean that we occasionally sample logits from
      % an interval slightly shorter than the desired number of seconds.
      endIdx = min(endIdx, size(lgts, 1)) ;
      %fprintf('sampling interval (%d,%d) from %d\n', startIdx, endIdx, size(lgts,1)) ;

      %sample the logits that correspond to the audio segment
      lgts_sampled = lgts(startIdx:endIdx,:) ;
    end

    % generate spectogram
    SPEC = runSpec(z, meta.audio) ;

    if inputnorm
      mu = mean(SPEC,2) ;
      stdev = std(SPEC,[],2) ;
      SPEC = bsxfun(@minus, SPEC, mu) ;
      SPEC = bsxfun(@rdivide, SPEC, stdev) ;
    end

    if numel(SPEC) ~= opts.imageSize(1) * opts.imageSize(2)
      fprintf('WARNING: Image size mismatch %d %d.\n', size(SPEC,1), size(SPEC,2)) ;
    end

    im{ii} = single(SPEC) ;

    % TAKING THE MEAN, THIS SHOULD RETURN A 1x8 VECTOR for every single
    % audio sample, CHANGE TO MAX IF REQUIRED
    switch opts.logitAggregator
      case 'mean', aggregator = @(x) mean(x, 1) ;
      case 'max', aggregator = @(x) max(x, [], 1) ;
      otherwise, error('unreccognised aggregator %s\n', opts.logitAggregator) ;
    end
    %pooled_lgts = mean(lgts_sampled, 1) ;
    pooled_lgts = aggregator(lgts_sampled) ;

    % subsample the desired number of emotions
    pooled_lgts = pooled_lgts(1:opts.numPredEmotions) ;
    if any(isnan(pooled_lgts))
      fprintf('NaN lgts...\n') ;
      keyboard
    end
    lg_sampled{ii} = pooled_lgts ;
  end

  % ----- Concatenate -----
  imo = gpuArray(cat(4, im{:})) ;
  lgo = gpuArray(cat(4, lg_sampled{:})) ;

  % arrange emotions along channel dimension
  lgo = reshape(lgo, 1, 1, opts.numPredEmotions, numel(images)) ;

  % ----- Subtract average -----
  if ~isempty(opts.averageImage)
    offset = gpuArray(opts.averageImage(1)) ;
    imo(:,:,:,:) = bsxfun(@minus, imo(:,:,:,:), offset) ;
  end

% ----------------------------------------
function idx = time2idx(time)
% ----------------------------------------
% TIME2IDX - convert time (in seconds) to frame index
  fps = 25 ; stride = 6 ;
  idx = floor(max(time * fps - 1, 0) / stride) + 1 ;

% ---------------------------------------------------------------------------
function [chspeed, inputnorm, noisy] = findSettings(transformation, opts)
% ---------------------------------------------------------------------------
  %% ----- Find settings -----
  isVal = any(strfind(transformation,'v')) ;

  if any(strfind(transformation,'S')) && ~isVal
    chspeed = true ;
    fprintf('S')
  else
    chspeed = false ;
    fprintf('-')
  end

  if any(strfind(transformation,'I'))
    inputnorm = true ;
    opts.averageImage = 0 ;
    fprintf('I')
  else
    inputnorm = false ;
    fprintf('-')
  end

  if any(strfind(transformation,'N')) && ~isVal
    noisy = true ;
    fprintf('N ')
  else
    noisy = false ;
    fprintf('- ')
  end
