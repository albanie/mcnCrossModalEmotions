function student_stats(varargin)
%STUDENT_STATS - compute the statistics for student predictions
%   STUDENT_STATS(VARARGIN) computes the similarity bewteen
%   predictions made by the student model (which operates on voices)
%   and those made by the teacher (which operates on faces) across
%   the EmoVoxCeleb dataset.
%
%   STUDENT_STATS(..'name', value) accepts the following options:
%
%   `refresh` :: false
%    If true, recomputes all features on EmoVoxCeleb using the student
%    model.
%
%   `partition` :: 'unheardTest'
%    The partition of the data to visualize (can be one of 'train',
%    'unheardVal', 'unheardTest', 'heardVal' and 'heardTest').
%
%   `visHist` :: false
%    If true, visualise the distribution of dominant predictions made by
%    the teacher.
%
%   `ignore` :: {'fear', 'contempt', 'disgust'}
%    For analysis purposes, certain partitions have very few samples
%    of some of the rarer emotions.  Using `ignore`, it is possible to
%    specify a list of emotions to drop from the visualisations if
%    desired.
%
%   `teacher` :: 'senet50-ferplus'
%    The name of the teacher model.
%
%   `student` :: 'emovoxceleb-student'
%    The name of the stduent model.
%
%   NOTES: vl_roc (from the vlfeat toolbox) is required to plot the curves.
%
% Copyright (C) 2018 Samuel Albanie, Arsha Nagrani
% Licensed under The MIT License [see LICENSE.md for details]

  opts.refresh = false ;
  opts.visHist = false ;
  opts.partition = 'all' ;
  opts.student = 'emovoxceleb-student' ;
  opts.teacher = 'senet50-ferplus' ;
  opts.ignore = {'fear', 'contempt', 'disgust'} ;
  opts.figDir = fullfile(fileparts(mfilename('fullpath')), ...
                                                       'emovoxceleb-figs') ;
  opts.cachePath = fullfile(vl_rootnn, ...
         'data/mcnCrossModalEmotions/cache/emovoxceleb-student-stats.mat') ;
  opts.expRoot = fullfile(vl_rootnn, '/data/xEmo18') ;
  opts = vl_argparse(opts, varargin) ;

  imdbDir = fullfile(vl_rootnn, 'data/mcnCrossModalEmotions', ...
                        sprintf('cachedFeats-audio')) ;
  featPath = fullfile(imdbDir, ...
         sprintf('%s-emovoxceleb-feats.mat', opts.student)) ;
  compute_audio_feats(featPath, 'modelName', opts.student, ...
                                'targetDataset', 'emovoxceleb', ...
                                'teacher', opts.teacher) ;

  stored = load(featPath) ;

  % note that confusingly, the student predictions are stored as
  % the faceLogits attribute.
  studentLogits = vertcat(stored.faceLogits{:}) ;
  [~,maxLogits] = max(studentLogits, [], 2) ;

  if opts.visHist
    histogram(maxLogits) ;
    title('histogram of dominant emotions (predicted by student)') ;
    if exist('zs_dispFig', 'file'), zs_dispFig ; end
   end

  if ~exist('loadedImdb', 'var')
    fprintf('loading EmoVoxCeleb Imdb...') ; tic ;
    loadedImdb = fetch_emovoxceleb_imdb(opts.teacher) ;
    fprintf('done in %g s\n', toc) ;
  end

  setIdx = 1:3 ;
  keys = {'train', 'unheardVal', 'heardVal'} ;
  setMap = containers.Map(keys, setIdx) ;

  if ~strcmp(opts.partition, 'all')
    partitions = {opts.partition} ;
  else
    partitions = keys ;
  end

  for ii = 1:numel(partitions)
    partition = partitions{ii} ;
    fprintf('compute stats for %s (%d/%d)...\n', partition, ii, numel(partitions)) ;

    % compute AUC using max logits as the target label
    keep = (loadedImdb.images.set == setMap(partition)) ;
    normedLogits = vl_nnsoftmaxt(studentLogits, 'dim', 2) ;
    subsetLogits = normedLogits(keep,:) ;
    [~,teacherMaxLogits] = cellfun(@(y) max(max(y,[], 1)), stored.wavLogits(keep)) ;

    if opts.visHist %  visualise histogram if required
      histogram(teacherMaxLogits) ;
      if exist('zs_dispFig', 'file'), zs_dispFig ; end
    end

    % compute AUC per class and visualise ROC
    if ~exist(opts.figDir, 'dir'), mkdir(opts.figDir) ; end
    net = emoVoxZoo(opts.student) ;
    emotions = net.meta.classes.name ;
    auc = zeros(1, numel(emotions)) ;
    for jj = 1:numel(emotions)
      classIdx = jj ;
      labels = -1 * ones(1, numel(teacherMaxLogits)) ;
      labels(teacherMaxLogits == classIdx) = 1 ;
      scores = subsetLogits(:, classIdx) ;
      [~,~,info] = vl_roc(labels, scores) ;
      auc(jj) = info.auc ;

      vl_roc(labels, scores) ; % visualise ROC curve if required
      title(sprintf('%s (%s)', emotions{jj}, partition)) ;
      destPath = fullfile(opts.figDir, ...
                     sprintf('%s-%s.jpg', emotions{jj}, partition)) ;
      if ~ismember(emotions{jj}, opts.ignore)
        saveas(1, destPath) ;
        if exist('zs_dispFig', 'file'), zs_dispFig ; end
      end
    end

    for jj = 1:numel(emotions)
      fprintf('%s: %g\n', emotions{jj}, auc(jj)) ;
    end

    if ~exist(fileparts(opts.cachePath), 'dir')
      mkdir(fileparts(opts.cachePath)) ;
    end
    if ~exist(opts.cachePath, 'file')
      cache.emotions = emotions ;
    else
      cache = load(opts.cachePath) ;
    end

    % compute mean average precision for emotions that are represented
    representedEmotions = unique(teacherMaxLogits) ;
    drop = find(ismember(emotions, opts.ignore)) ;
    representedEmotions(ismember(representedEmotions, drop)) = [] ;
    meanAuc = mean(auc(representedEmotions)) ;
    fprintf('meanAuc: %g\n', meanAuc) ;
    if ~isfield(cache, partition)
      cache.(partition) = auc ;
    end
    save(opts.cachePath, '-struct', 'cache') ;
  end
