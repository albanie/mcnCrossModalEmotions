function generateBaseImdb(varargin)
%GENERATEBASEIMDB - remap the VoxCeleb IMDB for EmoVoxCeleb
%   GENERATEBASEIMDB - starting from the VoxCeleb IMDB file,
%   this function modifies the IMDB such that the train and
%   validation splits match the descriptions in the paper
%
%     S. Albanie, A. Nagrani, A. Vedaldi, A. Zisserman,
%     Emotion Recognition in Speech using Cross-Modal Transfer
%     in the Wild, ACM Multimedia 2018
%
%   These splits are designed to preseve the "Unseen-Unheard" test
%   split of 250 identities, and the smaller "Seen-Heard" test split
%   of all remaining identities proposed in Section 5 of the paper:
%
%     A. Nagrani, S. Albanie, A. Zisserman, Learnable PINs: Cross-Modal
%     Embeddings for Person Identity, ECCV 2018
%     https://arxiv.org/pdf/1805.00833.pdf
%
%   NOTE: this code should not need to be re-run, since the sets are
%   included with the public IMDB released with EmoVoxCeleb, but the script
%   is included here for completeness.
%
% Copyright (C) 2018 Samuel Albanie, Arsha Nagrani
% Licensed under The MIT License [see LICENSE.md for details]

  opts.numTracks = 153486 ;
  opts.numIdentities = 1251 ;
  opts.voxcelebImdb = fullfile(vl_rootnn, 'data/voxceleb/imdb.mat') ;
  opts.emoVoxCelebImdb = fullfile(vl_rootnn, 'data/emoVoxCeleb/imdb.mat') ;
  opts.eccv_mapData = fullfile(vl_rootnn, 'data/voxceleb/eccv_mapData.mat') ;

  fprintf('loading ECCV mapData...') ; tic ;
  mapData = load(opts.eccv_mapData) ;
  fprintf('done in %g s\n', toc) ;

  fprintf('loading VoxCeleb imdb...') ; tic ;
  imdb = load(opts.voxcelebImdb) ;
  fprintf('done in %g s\n', toc) ;
  setMaps = retrieveSetIdx(mapData, opts.numTracks, opts.numIdentities) ;

  assert(isequal(setMaps.trackSpIds, imdb.images.sp), ...
                                             'speakers are not aligned') ;
  assert(isequal(imdb.images.name, setMaps.trackWavs), ...
                                              'wavPaths are not aligned') ;

  % merge "val (US-UH)" from ECCV into the training split (118,485 in total)
  setMaps.trackSets(setMaps.trackSets == 2) = 1 ;

  % keep "test (US-UH)" as is (30,496 in total)
  setMaps.trackSets(setMaps.trackSets == 4) = 2 ;

  % keep "test (S-H)" as is (4505 in total).  Note that the "total" numbers
  % here differ slightly from the publication, because there is one additional
  % filtering step that is performed before the EmoVoxCeleb IMDB is ready
  % for usage.
  setMaps.trackSets(setMaps.trackSets == 3) = 3 ;

  partitions = {'train', 'test (US-US)', 'test (S-H)'} ;
  for ii = 1:numel(partitions)
    numTracksInPartition = sum(setMaps.trackSets == ii) ;
    fprintf('tracks in %s: %d\n', partitions{ii}, numTracksInPartition) ;
  end

  imdb.images.set = setMaps.trackSets ;
  if ~exist(fileparts(opts.emoVoxCelebImdb), 'dir')
    mkdir(fileparts(opts.emoVoxCelebImdb)) ;
  end
  fprintf('saving emoVoxCeleb imdb...') ; tic ;
  save(opts.emoVoxCelebImdb, '-struct', 'imdb') ;
  fprintf('done in %g s\n', toc) ;

% ----------------------------------------------------------------------
function setMaps = retrieveSetIdx(mapData, numTracks, numIdentities)
% ----------------------------------------------------------------------
%RETRIEVESETIDX - generate splits for IMDB
%   SETMAPS = RETRIEVESETIDX(MAPDATA, NUMTRACKS, NUMIDENTITIES) retrieves
%   the setIdx from the ECCV mapping. MAPDATA contains the mapping from
%   identities to splits, NUMTRACKS is the expected number of tracks and
%   NUMIDENTITIES is the expected number of identities (used for sanity
%   checks.

  % The mapData should be a collection of mappings between identities and sets
  assert(all(isfield(mapData, {'mapvoice', 'speakset'})), ...
                      'missing expected fields in mapData') ;
  assert(size(mapData.mapvoice, 1) == numIdentities, ...
                      'unexpected number of identities') ;

  trackSets = arrayfun(@(x) {mapData.mapvoice(x).set}, 1:numIdentities) ;
  trackWavs = arrayfun(@(x) {mapData.mapvoice(x).name}, 1:numIdentities) ;
  trackSpIds = arrayfun(@(x) {ones(1, numel(trackSets{x})) * x}, ...
                                                       1:numIdentities) ;

  % flatten
  trackSets = [trackSets{:}] ;
  trackSpIds = [trackSpIds{:}] ;
  trackWavs = [trackWavs{:}] ;

  % sanity checks
  assert(numel(trackSets) == numTracks, 'unexpected num. of tracks') ;
  assert(numel(trackWavs) == numTracks, 'unexpected num. of wav paths') ;
  assert(numel(trackSpIds) == numTracks, 'unexpected num. of speaker ids') ;

  setMaps.trackSets = trackSets ;
  setMaps.trackWavs = trackWavs ;
  setMaps.trackSpIds = trackSpIds ;
