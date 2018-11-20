function teacher_stats(varargin)
%TEACHER_STATS - compute the distribution of teacher predictions
%   TEACHER_STATS(VARARGIN) computes the distribution of emotion predictions
%   made by the teahcer model on the EmoVoxCeleb dataset and the Afew6.0
%   dataset [1] and combines them in a figure.
%
%   TEACHER_STATS(..'name', value) accepts the following options:
%
%   `figurePath` :: fullfile(vl_rootnn, 'data/xEmo18/emovoxceleb-figure.pdf')
%    Path to location where figure will be stored.
%
%    References:
%     [1] Dhall, Abhinav, et al. "Collecting large, richly annotated
%       facial-expression databases from movies." IEEE multimedia 19.3%
%       (2012): 34-41.
%
% Copyright (C) 2018 Samuel Albanie, Arsha Nagrani
% Licensed under The MIT License [see LICENSE.md for details]

  opts.figurePath = fullfile(vl_rootnn, ...
                       'data/emoVoxCeleb/emovoxceleb-figure.pdf') ;
  opts.afewLogits = fullfile(vl_rootnn, 'data/emoVoxCeleb/afew-logits.mat') ;
  opts = vl_argparse(opts, varargin) ;

  fprintf('loading imdb of predictions on EmoVoxCeleb...') ; tic ;
  opts.teacher = 'senet50-ferplus' ;
  imdb = fetch_emovoxceleb_imdb(opts.teacher) ;
  allLogits = vertcat(imdb.wavLogits{:}) ;
  [~, emoVoxPreds] = max(allLogits, [], 2) ;
  fprintf('done in %g (s) \n', toc) ;

  fprintf('loading imdb of teacher predictions on AFEW...') ; tic ;
  % Note: These are stored online as a convenience but can be recomputed
  % if desired.
  if ~exist(fileparts(opts.afewLogits), 'dir')
		mkdir(fileparts(opts.afewLogits)) ;
	end
	fetchLogitsFromInternet(opts.afewLogits, 'afew-logits') ;
  xx = load(opts.afewLogits) ;
  emos = vertcat(xx.faceLogits{:}) ;
  [~, afewPreds] = max(emos, [], 2) ;
  fprintf('done in %g (s) \n', toc) ;
  plotHistogram(emoVoxPreds, afewPreds, opts.figurePath) ;
end

% --------------------------------------------------------------------
function plotHistogram(emoVoxPreds, afewPreds, figurePath)
% --------------------------------------------------------------------

  ferPlusEmotions = {'Neutral', 'Happiness', 'Surprise', 'Sadness', ...
                    'Anger', 'Disgust', 'Fear', 'Contempt'} ;

  olive = [ 125 ; 168 ; 50 ] / 255 ; % pick a colour scheme
  set(0,'defaulttextinterpreter','latex') ;

  h = figure(1) ;
  [emoCeleb, ~] = histcounts(emoVoxPreds, 0.5:8.5) ;
  [compared, edges] = histcounts(afewPreds, 0.5:8.5) ;
  label1 = 'EmoVoxCeleb' ;
  label2 = 'Afew 6.0' ;
	ctrs = edges(1)+(1:length(edges)-1).*diff(edges);   % Create Centres
	b = bar(ctrs, [emoCeleb ; compared]', 1, 'BarWidth', 1) ;
  b(1).EdgeColor = olive ; b(1).FaceColor = olive ; b(1).FaceAlpha = 0.8 ;
  b(2).EdgeColor = 'red'; b(2).FaceColor = 'red'; b(2).FaceAlpha = 0.8 ;

	set(gca,'YScale','log') ; grid on ;
	set(gca, 'xtick', ctrs, 'xticklabel', ferPlusEmotions) ;
  ylabel('Number of frames') ;
  set(groot, 'defaultAxesTickLabelInterpreter','latex') ;
  set(groot, 'defaultLegendInterpreter','latex') ;
  legend({label1, label2}) ;
  a = get(gca,'XTickLabel') ;
  set(gca,'XTickLabel', a, 'fontsize', 15) ;
  xlim([1 9]) ;
	xtickangle(60) ;

  if exist('zs_dispFig', 'file'), zs_dispFig ; end

  set(h,'Units','Inches');
  pos = get(h,'Position');
  set(h,'PaperPositionMode', 'Auto', ...
         'PaperUnits', 'Inches', 'PaperSize',[pos(3), pos(4)]) ;
  print(figurePath, '-dpdf','-r0') ;
end

% -------------------------------------------------
function fetchLogitsFromInternet(destPath, imdbName)
% -------------------------------------------------

  waiting = true ;
  prompt = sprintf(...
        strcat('Logits were not found at %s\nWould you like to ', ...
        ' download them from THE INTERNET (y/n)?\n'), destPath) ;

  baseUrl = 'http://www.robots.ox.ac.uk/~albanie/data/cross-modal-emotions' ;

  switch imdbName
    case 'afew-logits'
      url = [baseUrl '/afew-logits.mat'] ;
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
