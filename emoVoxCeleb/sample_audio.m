function sample_audio(varargin)
%SAMPLE_AUDIO - generate some audio samples
%   SAMPLE_AUDIO(VARARGIN) generates a collection of audio samples
%   from EmoVoxCeleb, together with their associated "peak" frames
%   in the dataset and corresponding aggregate teacher predictions.
%
%   SAMPLE_AUDIO(..'name', value) accepts the following options:
%
%   `clobber` :: false
%    Overwrite existing samples.
%
%   `samplePeaks` :: true
%    Store the approximate peak frame associated with each audio segment.
%
%   `sampleFrqmeSeq` :: false
%    Store every frame in the track associated with each audio sample.
%
%   `teacher` :: 'senet50-ferplus'
%    The name of the teacher model used to generate the frame logits.
%
%   `ignore` :: {'disgust', 'contempt', 'fear'}
%    Avoid sampling emotions with poor representation in the dataset
%    (otherwise will get a bunch of repeats).
%
%   `wavDir` :: fullfile(vl_rootnn, 'data/ramdisk/voxceleb_all')
%    Directory containing the wavfiles of VoxCeleb.
%
%   `faceDir` :: fullfile(vl_rootnn, ...
%                           'data/datasets/voxceleb1/unzippedIntervalFaces') ;
%    Directory containing the extracted face frames from VoxCeleb.
%
% Copyright (C) 2018 Samuel Albanie, Arsha Nagrani
% Licensed under The MIT License [see LICENSE.md for details]

  opts.vis = false ;
  opts.clobber = false ;
  opts.samplePeaks = true ;
  opts.sampleFrameSeq = false ;
  opts.ignore = {'disgust', 'contempt', 'fear'} ;
  opts.teacher = 'senet50-ferplus' ;
  opts.wavDir = fullfile(vl_rootnn, 'data/datasets/voxceleb1/voxceleb_all') ;
  opts.aviDir = '/datasets/voxceleb1/avi' ;
  opts.faceDir = fullfile(vl_rootnn, ...
                           'data/datasets/voxceleb1/unzippedIntervalFaces') ;
  opts = vl_argparse(opts, varargin) ;

  destFolder = fullfile(vl_rootnn, 'data/mcnCrossModalEmotions/samples', ...
                        opts.teacher) ;

  % pick some colors for visualisation purposes
	colors = { ...
		[202, 202, 202], ... % neutral
		[250, 190, 190], ... % happiness
		[230, 190, 255], ... % surprise
		[88, 112, 209], ... % sadness
		[230, 88, 88], ... % anger
		[32, 162, 102], ... % disgust
		[0, 128, 128], ... % fear
		[0, 0, 0], ... % contempt
	} ;

	emotions = {'neutral', 'happiness', 'surprise', 'sadness', ...
								'anger', 'disgust', 'fear', 'contempt'} ;
  emotionIdx = find(~ismember(emotions, opts.ignore)) ;
  emotionLabels = emotions(emotionIdx) ;
  imdb = fetch_emovoxceleb_imdb(opts.teacher) ;

  % logits used for sampling
  [~,maxIdx] = cellfun(@(x) max(x(:)), imdb.wavLogits, 'Uni', 0) ;
  [frameIdx,tags] = cellfun(@(x, y) ind2sub(size(x), y), ...
                            imdb.wavLogits, maxIdx) ;

  % logits used for visuals
  maxedLogits = cellfun(@(x) {max(x, [], 1)}, imdb.wavLogits) ;
  samplesPerEmo = 20 ;
  sampledWavs = cell(1, numel(emotions)) ;
  sampledFrames = cell(1, numel(emotions)) ;
  sampledLogits = cell(1, numel(emotions)) ;

  rng(0) ; % for consistency
  cont = confirmSamplingProcess(destFolder, opts.clobber) ;
  if ~cont, fprintf('exiting sampler\n') ; return ; end

  for ii = 1:numel(emotionLabels)
    emoIdx = emotionIdx(ii) ;
    tagged = find(tags == emoIdx) ;
    fprintf('found %d audio segments for %s, picking %d\n', ...
       numel(tagged), emotionLabels{ii}, samplesPerEmo) ;
    picks = randsample(numel(tagged), min(numel(tagged), samplesPerEmo)) ;
    samples = tagged(picks) ;
    sampledWavs{ii} = imdb.images.name(samples) ;

    sampleIds = imdb.images.id(samples) ; % map to ids
    allSampledFrames = arrayfun(@(x) ...
    {imdb.images.denseFrames(imdb.images.denseFramesWavIds == x)}, sampleIds) ;
    peakFrames{ii} = cellfun(@(x,y) {x{y}}, allSampledFrames, ...
                                           num2cell(frameIdx(samples))) ; %#ok
    sampledFrames{ii} = allSampledFrames ;
    sampledLogits{ii} = maxedLogits(samples) ;
  end

  % store samples in convenient layout
  for ii = 1:numel(emotionLabels)
    baseDir = fullfile(destFolder, emotionLabels{ii}) ;

    % copy wavs, logits.txt and store avi path
    wavPaths = sampledWavs{ii} ;
    for jj = 1:numel(wavPaths)
      subfolder = num2str(jj) ;
      wavPath = wavPaths{jj} ;
      [relPath, fname,~] = fileparts(wavPath) ;

      % store logits and aviPath in a textfile
      origAviPath = fullfile(relPath, [fname '.avi'] ) ;
      aviFilePath = fullfile(baseDir, subfolder, 'meta.txt') ;
      zs_mkdirRec(fileparts(aviFilePath)) ;
      fid = fopen(aviFilePath, 'w');
      fprintf(fid, 'aviPath: %s\n', origAviPath) ;
      logitTemplate = repmat('%.4f ', 1, 7) ;
      fprintf(fid, [logitTemplate '\n'], sampledLogits{ii}{jj}) ;
      fclose(fid) ;

      srcPath = fullfile(opts.wavDir, wavPath) ;
      wavFname = strrep(wavPaths{jj}, '/', '-') ;
      destPath = fullfile(baseDir, subfolder, wavFname) ;
      zs_mkdirRec(fileparts(destPath)) ;
      fprintf('(%s) copying wav samples: %s -> %s\n', emotionLabels{ii}, ...
              srcPath, destPath)
      copyfile(srcPath, destPath) ;

      aviSrcPath = fullfile(opts.aviDir, origAviPath) ;
      aviFname = strrep(origAviPath, '/', '-') ;
      aviDestPath = fullfile(baseDir, subfolder, aviFname) ;
      fprintf('(%s) copying avi samples: %s -> %s\n', emotionLabels{ii}, ...
              aviSrcPath, aviDestPath)

      dist = sampledLogits{ii}{jj} ;

      clf ; b = bar(diag(dist), 'stacked') ;
      for cc = 1:numel(colors)
        b(cc).FaceColor = colors{cc} / 255 ;
        b(cc).EdgeColor = 'None' ;
      end

			% Change the axes tick marks and tick labels
      stubs = cellfun(@(x) {[upper(x(1)) lower(x(2:3))]}, emotions) ;
			set(gca, 'XTick', 1:numel(emotions), ...
          'XTickLabel', stubs, ...
          'YTickLabel', [], ...
          'TickLength', [0 0], ...
          'XAxisLocation','top', ...
          'fontsize', 24) ;
      xlim([0.5 8.5]) ;
      ylim([min(-3, min(dist)) max(10, max(dist))]) ;

      if opts.vis
        if exist('zs_dispFig', 'file'), zs_dispFig ; end
      end
      figPath = fullfile(baseDir, subfolder, 'distribution.png') ;
      fig = gcf ;
      set(gcf, 'color', 'w') ;
      fig.PaperUnits = 'inches' ;
      fig.PaperPosition = [0 0 8 4] ;
      print(figPath, '-dpng') ;
    end

    if opts.samplePeaks
      peakFramePaths = peakFrames{ii} ;
      for jj = 1:numel(peakFramePaths)
        subfolder = num2str(jj) ;
        srcPath = fullfile(opts.faceDir, peakFramePaths{jj}) ;
        destPath = fullfile(baseDir, subfolder, 'peakFrame.jpg') ;
        zs_mkdirRec(fileparts(destPath)) ;
        fprintf('(%s) copying peak frame: %s -> %s\n', emotions{ii}, ...
                srcPath, destPath)
        copyfile(srcPath, destPath) ;
      end
    end

    if opts.sampleFrameSeq
      framePathSets = sampledFrames{ii} ;
      for jj = 1:numel(framePathSets)

        framePathSet = sort(framePathSets{jj}) ;
        for kk = 1:numel(framePathSets{jj})
          subfolder = num2str(jj) ;
          srcPath = fullfile(opts.faceDir, framePathSet{kk}) ;
          destName = sprintf('%05d.jpg', kk) ;
          destPath = fullfile(baseDir, subfolder, 'frames', destName) ;

          zs_mkdirRec(fileparts(destPath)) ;
          fprintf('(%s) copying all frames: %s -> %s\n', emotions{ii}, ...
                  srcPath, destPath)
          copyfile(srcPath, destPath) ;
        end
      end
    end
  end
end

% --------------------------------------------------------------
function cont = confirmSamplingProcess(folder, clobber)
% --------------------------------------------------------------
	cont = true ;
  rmFolderCmd = sprintf('rm -rf %s\n', folder) ;
  if exist(folder, 'dir')
    if clobber, system(rmFolderCmd) ; return ; end
		waiting = true ;
    fprintf('destination directory at %s already exists\n', folder) ;
    prompt = 'Would you like to wipe it? (y/n)\n' ;
		while waiting
			str = input(prompt,'s') ;
			switch str
				case 'y', system(rmFolderCmd) ; return
				case 'n', cont = false ; return
				otherwise
          fprintf('input %s unrecognised, please use `y` or `n`\n', str) ;
			end
		end
  end
end
