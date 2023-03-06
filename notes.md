# Notes during implementation 

## Alternatives

- https://stackoverflow.com/questions/16463582/memoize-to-disk-python-persistent-memoization
- https://stackoverflow.com/questions/17999204/lru-cache-on-hard-drive-python

## APIs

- Cache
  - get(hsh)
    - update last accessed in metadata
    - store.get(...)
  - put(hsh, item)
    - append to metadata
    - store.put(..)
    - evict cache if necessary
- Hasher
  - function kwargs + id
  - https://stackoverflow.com/questions/830937/python-convert-args-to-kwargs
- CacheItem
  - hash = ...
  - 
- MetadataStorage
  - add(item)



## MetadataStorage

- How to store metadata?
  - id
  - byte_size
  - last_accessed
  - last_modified
  - ...

Implementation options:

- sqlite3
    - is it thread-safe? does it need a lock?
    - https://stackoverflow.com/a/2894830/2135504
    - https://stackoverflow.com/a/6969938/2135504
- sqlite dict
    - https://github.com/RaRe-Technologies/sqlitedict
    - hmm... more of a key-value store than RDBM
- pandas and sqlite?
    - https://datacarpentry.org/python-ecology-lesson/09-working-with-sql/index.html
    - https://stackoverflow.com/a/36029761/2135504
    - ensure closing DB
    - https://codereview.stackexchange.com/a/182706/202014
    - ah, might actually work because can select columns!
    - maybe not as great, because we don`t fix schema?! Also not as flexible (e.g. select only last accessed! or modify only one thing...)
- tinydb  
    - https://github.com/msiemens/tinydb (4.7k stars)
    - nice, but rather for document management
- sqlalchemy and sqlite is likely the solution  
    - https://docs.sqlalchemy.org/en/20/orm/quickstart.html
    - A bit more tricky, but seems like correct choice