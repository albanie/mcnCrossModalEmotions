function audio_benchmarks(varargin)

  opts.clobber = false ;
  opts.scratch = false ;
  opts.numEpochs = 200 ;
  opts.official = false ;
  opts.manualEpoch = 0 ;
  opts.testMode = false ;
  opts.strategies = {'max'} ;
  opts.teacherType = 'CNTK' ;
  opts.affBiases = {true} ;
  opts.evaluationMethod = 'cross-val' ;
  opts.figDir = fullfile(vl_rootnn, 'data/affine-figs-audio-splits') ;
  opts.datasets = {'rml', 'enterface'} ;
  opts = vl_argparse(opts, varargin) ;

  modelPairs = getAudioModels('teacherType', opts.teacherType) ;
  if ~exist(opts.figDir, 'dir'), mkdir(opts.figDir) ; end

  % The labels learned by the student on EmoVoxCeleb
	modelEmoLabels = {'neutral', 'happiness', 'surprise', ...
										'sadness', 'anger', 'disgust', ...
										'fear', 'contempt'} ;

  for ii = 1:numel(opts.datasets)

    dataset = opts.datasets{ii} ;
    datasetLabels = {'Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise'} ;

    for jj = 1:numel(modelPairs)
      modelName = modelPairs{jj}{1} ;
      numSrcEmotions = modelPairs{jj}{2} ;
      origExpDir = '~/coding/libs/mcn/contrib-matconvnet/data/xEmo18/' ;
      modelDir = [origExpDir modelName] ;

      if ~strcmp(modelName, 'random')
        [~,epoch] = audio_zoo(modelDir, opts.scratch, ...
                       'manualEpoch', opts.manualEpoch) ;
      else
        epoch = 0 ;
      end


      numTargetEmotions = 7 ;
      targetEmoLabels = modelEmoLabels(1:numTargetEmotions) ;

      numExps = numel(opts.strategies) * numel(opts.affBiases) ;
      storedW = zeros(numSrcEmotions, numTargetEmotions, numExps) ;
      storedB = zeros(numTargetEmotions, numExps) ;
      accs = zeros(1, numExps) ; counter = 1 ;
      expNames = {} ;

      for bb = 1:numel(opts.affBiases)
        affBias = opts.affBiases{bb} ;

        for kk = 1:numel(opts.strategies)
          strategy = opts.strategies{kk} ;
          expName = sprintf('%s-%s-epoch%d', modelName, strategy, epoch) ;
          if ~affBias
            expName = [expName '-no-bias'] ; %#ok
          end
          fprintf('learning affine weights...') ;
          [miniImdb(counter),expDirs,valIdxSets] = run_cross_val(...
                     'modelName', modelName, ...
                     'audioModelDir', modelDir, ...
                     'official', opts.official, ...
                     'targetDataset', dataset, ...
                     'aggregator', strategy, ...
                     'refreshCkpts', opts.clobber, ...
                     'numSrcEmotions', numSrcEmotions, ...
                     'manualEpoch', opts.manualEpoch, ...
                     'evaluationMethod', opts.evaluationMethod, ...
                     'numEpochs', opts.numEpochs, ...
                     'affBias', affBias, ...
                     'testMode', opts.testMode, ...
                     'modality', 'audio') ;
          foldAccs = zeros(1, numel(expDirs)) ;

          % note, neutral class should be empty, but we leave it for here
          % for simplicity
          confSum = zeros(7, 7) ;

          for ee = 1:numel(expDirs)

            expDir = expDirs{ee} ;
            bestEpoch = findBestEpoch(expDir, 'prune', false) ;
            bestNet = fullfile(expDir, sprintf('net-epoch-%d.mat', bestEpoch)) ;

            %bestEpoch = findBestEpoch(expDir, 'prune', true) ;
            %bestNet = fullfile(expDir, sprintf('net-epoch-%d.mat', bestEpoch)) ;
            tmp = load(bestNet) ; net = Net(tmp.net) ;
            w = net.getValue('prediction_filters') ;
            w = reshape(w, numSrcEmotions, numTargetEmotions) ;
            storedW(:,:,counter) = w ;

            % ----------------------------
						% compute requested metrics
            % ----------------------------
            valIdx = valIdxSets{ee} ;
						X = miniImdb.fusedLogits(valIdx,:) ;
						preds = X * w ;

						if affBias
							b = net.getValue('prediction_biases') ;
							storedB(:,counter) = b ;
							preds = bsxfun(@plus, preds, b') ; % include biases
						end

            labels = miniImdb.labels(valIdx) ;
            [~,cls] = max(preds, [], 2) ;
            matches = cls == labels' ;
            acc = sum(matches) / numel(matches) ;
            confmat = confusionmat(labels', cls, 'Order', 1:7) ;
            confSum = confSum + confmat ;

            fprintf('(fold %d/%d) recomputed accuracy: %.1f\n', ...
                  ee, numel(expDirs), 100 * acc) ;

            accs(counter) = 100 * (1 - tmp.stats.val(end).error) ;
            fprintf('(fold %d/%d) accuracy for %s: %.1f\n', ee, numel(expDirs), ...
                                                     expName, accs(counter)) ;
            expNames{end+1} = expName ; %#ok
            if strcmp(opts.evaluationMethod, 'cross-val')
              foldAccs(ee) = accs(counter) ;
            end
          end

          if strcmp(opts.evaluationMethod, 'cross-val') % do not use with x-val
            fprintf('-----------------------------\n') ;
            fprintf('DATASET: %s\n', dataset) ;
            fprintf('MODEL: %s\n', modelName) ;
            fprintf('cross-validation score: %g, std %g \n', mean(foldAccs), std(foldAccs)) ;
            fprintf('-----------------------------\n') ;
          end

          if strcmp(opts.evaluationMethod, 'cross-val') % do not use with x-val
            fprintf('-----------------------------\n') ;
            fprintf('DATASET: %s\n', dataset) ;
            fprintf('MODEL: %s\n', modelName) ;
            fprintf('cross-validation score: %g, std %g \n', mean(foldAccs), std(foldAccs)) ;
            fprintf('-----------------------------\n') ;
          end

          fprintf('confusion matrix:\n') ;
          disp(confSum) ;

          prunedConfSum = confSum(2:end,2:end) ;
          fprintf('confusion matrix (minus neutral):\n') ;
          disp(prunedConfSum) ;

          fprintf('normalized confusion matrix (minus neutral):\n') ;
          normed = bsxfun(@rdivide, prunedConfSum, sum(prunedConfSum,2)) ;
          disp(normed) ;

          % figure creation
					h = figure(1) ;
          confmat_colors(normed, datasetLabels) ;
          %titleStr = sprintf('Confusion Matrix for %s\n', dataset) ;
          %title(titleStr, 'interpreter', 'latex', 'fontsize', 24) ;

          figName = sprintf('%s-%s.pdf', dataset, modelName) ;
          figPath = fullfile(opts.figDir, 'confmat', figName) ;
          if ~exist(fileparts(figPath), 'dir'), mkdir(fileparts(figPath)) ; end

					set(h,'Units','Inches');
					pos = get(h,'Position');
					set(h,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])

					print(figPath, '-dpdf','-r0') ;
          fprintf('saving figure at %s\n', figPath) ;
          zs_dispFig ;

          %% sanity check on weights
					X = miniImdb.fusedLogits ;
					preds = X * w ;

          if affBias
            b = net.getValue('prediction_biases') ;
            storedB(:,counter) = b ;
            preds = bsxfun(@plus, preds, b') ; % include biases
          end

          if false % debug only
            gt = zeros(size(preds)) ; %#ok
            for zz = 1:size(preds,1)
              gt(zz,miniImdb(counter).labels(zz)) = 1 ;
            end
            clf ;
            subplot(1,2,1) ; imagesc(preds) ;
            subplot(1,2,2) ; imagesc(gt) ;
            [~,cls] = max(preds, [], 2) ;
            trainIdx = (miniImdb(counter).images.set == 1) ;
            valIdx = (miniImdb(counter).images.set == 2) ;
            matches = cls == miniImdb(counter).labels' ;
            trAcc = sum(matches(trainIdx)) / numel(matches(trainIdx)) ;
            valAcc = sum(matches(valIdx)) / numel(matches(valIdx)) ;
          end

          counter = counter + 1 ;
        end
			end
		end
  end
end
