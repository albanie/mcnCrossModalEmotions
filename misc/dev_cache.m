function [val, found] = dev_cache(key, value)
%DEV_CACHE - store state in global variables during development
%   [VAL, FOUND] = DEV_CACHE(KEY, VALUE) provides a global
%   key-value store for development purposes.  It can be useful
%   when working with files that take several minutes to load
%   from disk.

  global cache ;
  val = [] ;

  if nargin == 1
    mode = 'get' ;
  else
    mode = 'put' ;
  end

  switch mode
    case 'get'
      found = ~isempty(cache) && isfield(cache, key) ;
      if found
        fprintf('found key-value pair in cache, retrieving..\n') ;
        val = cache.(key) ;
      end
    case 'put'
      cache.(key) = value ;
  end
