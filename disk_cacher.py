#%%
import utils

utils.setup_ipython()


# %%
import inspect
from assertpy import assert_that


def resolve_args(func, *args, **kwargs):
    sig = inspect.signature(func)
    bounds = sig.bind(*args, **kwargs)
    bounds.apply_defaults()
    return bounds.arguments


def test_resolve_args():
    def func(x, y, o1="foo", o2=123):
        pass

    a = resolve_args(func, 1, 33, o2={"foo": 1222})
    e = {"x": 1, "y": 33, "o1": "foo", "o2": {"foo": 1222}}
    assert_that(a).is_equal_to(e)


if utils.run_as_cell(__name__):
    test_resolve_args()

# %%
from deepdiff import DeepHash


def hash_obj(obj):
    """Hashes arbitrary objects using deepdiff.DeepHash

    Note: This is surprisingly difficult, see
    - https://stackoverflow.com/questions/5884066/hashing-a-dictionary
    - https://hynek.me/articles/hashes-and-equality/

    Just hoping that DeepHash is somewhat sensible!
    """
    return DeepHash(obj)[obj]


def test_hash_obj():
    a = hash_obj({"x": 1, "y": 33, "o1": "foo", "o2": {"foo": 1222}})
    e = "b1c3136e7ec7e5833a1babc76628be37cd49a6f62379fa47e7a912006ac8785a"
    assert_that(a).is_equal_to(e)


if utils.run_as_cell(__name__):
    test_hash_obj()

#%% tinydb
from tinydb import middlewares, storages, TinyDB
import pandas as pd
import filelock
import time
from contextlib import contextmanager


class MyTinyDB:
    """TinyDB with caching and filelocking (for safe parallel processing)."""

    def __init__(self, path) -> None:
        self.db = TinyDB(
            path,
            storage=middlewares.CachingMiddleware(storages.JSONStorage),
        )
        self.lock = filelock.SoftFileLock(Path(path).with_suffix(".lock"))

    @contextmanager
    def open(self) -> TinyDB:
        with self.lock:
            with self.db as db:
                yield db


if utils.run_as_cell(__name__):
    database = MyTinyDB("db.json")

    def read(x):
        with database.open() as db:
            print("entering")
            db.all()  # verify no error
            time.sleep(1)
            print("exiting")

    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(3) as exec:
        future = exec.map(read, range(4))
        list(future)

#%%

p = Path("mycache/da1806f8b460e832332f69395d1810c07bac441834fbdd2791d5c0d946626eaf.pkl")


# %%
from pathlib import Path
import pickle
import logging
import shutil
import tempfile
import abc
from typing import Any

LOGGER = logging.getLogger(__name__)


class AbstractStore(abc.ABC):
    @abc.abstractmethod
    def get(self, id_: str) -> Any:
        ...

    @abc.abstractmethod
    def put(self, obj: Any, id_: str) -> int:
        ...

    @abc.abstractmethod
    def delete(self, id_: str) -> None:
        ...

    @abc.abstractmethod
    def clear(self) -> None:
        ...


class PickleStore(AbstractStore):
    def __init__(self, dirpath: Path):
        assert isinstance(dirpath, Path)
        dirpath = dirpath.absolute()
        if dirpath.exists():
            # TODO: accept non-empty dir only if has metadata matching dir?!
            LOGGER.warning(f"{dirpath=} already exists")
        else:
            dirpath.mkdir(exist_ok=True)

        self.dirpath = dirpath

    def get(self, id_: str) -> Any:
        p = self._get_path(id_)
        with open(p, "rb") as f:
            return pickle.load(f)

    def put(self, obj: Any, id_: str) -> int:
        p = self._get_path(id_)
        if p.exists():
            raise FileExistsError(f"{id_=} already exists!")
        with open(p, "wb") as f:
            pickle.dump(obj, f)
        return p.stat().st_size

    def delete(self, id_: str) -> None:
        p = self._get_path(id_)
        p.unlink()

    def clear(self):
        shutil.rmtree(self.dirpath)
        self.dirpath.mkdir()

    def _get_path(self, id_: str) -> Path:
        return self.dirpath / f"{id_}.pkl"


def test_pickle_store():

    import geopandas as gpd
    from geopandas.testing import assert_geodataframe_equal

    dirpath = tempfile.mkdtemp()
    obj = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    hsh = "some_random_hash"
    store = PickleStore(Path(dirpath))

    store.put(obj, hsh)
    reloaded = store.get(hsh)
    assert_geodataframe_equal(obj, reloaded)

    store.clear()  # to remove the tempdir, not strictly necessary


if utils.run_as_cell(__name__):
    test_pickle_store()

#%%
import pandas as pd

df = pd.DataFrame(
    [
        {"val": 1, "name": "hello"},
        {"val": 2, "name": "hello"},
        {"val": 3, "name": "hello"},
    ]
)
df

#%% Also store metadata
import time
from tinydb import Query
import pandas as pd


class Cache:
    def __init__(
        self, dirpath: Path, max_byte_size: int = 0, max_item_count: int = 0
    ) -> None:
        self.dirpath = dirpath
        self.store: AbstractStore = PickleStore(dirpath)
        self.db = MyTinyDB(self.db_path)

        self.max_byte_size = max_byte_size
        self.max_item_count = max_item_count

    @property
    def db_path(self) -> Path:
        return self.dirpath / "db.json"

    def put(self, obj, id_: str):
        byte_size = self.store.put(obj, id_)
        info = {
            "id_": id_,
            "byte_size": byte_size,
            "last_accessed_time": time.time_ns(),
            "created_time": time.time_ns(),
        }
        with self.db.open() as db:
            db.insert(info)
        self._evict_objects()

    def get(self, id_):
        obj = self.store.get(id_)
        query = Query()
        with self.db.open() as db:
            print(f"Udpating {id_}")
            db.update(
                {"last_accessed_time": time.time_ns()},
                query.id_ == id_,
            )
        return obj

    def clear(self):
        self.store.clear()
        self.db_path.unlink(missing_ok=True)

    def _evict_objects(self):
        with self.db.open() as db:
            df = pd.DataFrame(db.all())
        # earliest time should be top
        df.sort_values(by="last_accessed_time", ascending=True, inplace=True)

        delete_list = []
        while self._requires_eviction(df):
            delete_list.append(df.loc[df.index[0], "id_"])
            df = df.iloc[1:]

        for id_ in delete_list:
            print(f"deleting {id_}")
            # Prevent deleting an object, which another process is currently reading.
            with self._get_lock(id_):
                self.store.delete(id_)

                query = Query()
                with self.db.open() as db:
                    db.remove(query.id_ == id_)

    def _requires_eviction(self, df: pd.DataFrame) -> bool:
        if self.max_byte_size and df["byte_size"].sum() > self.max_byte_size:
            return True
        if self.max_item_count and len(df) > self.max_item_count:
            return True
        return False

    @contextmanager
    def _get_lock(self, id_):
        with filelock.SoftFileLock(self.dirpath / f"{id_}.lock"):
            yield


class FakeStore(AbstractStore):
    def get(self, id_: str) -> Any:
        return "fake object"

    def put(self, obj: Any, id_: str) -> int:
        return 100  # bytes

    def delete(self, id_: str) -> None:
        return

    def clear():
        return


if utils.run_as_cell(__name__):
    dirpath = Path("mycache")
    cache = Cache(dirpath, max_byte_size=250)
    cache.clear()
    cache.store = FakeStore()

    def pc():
        with cache.db.open() as db:
            print(db.all())

    cache.put("", "1")
    cache.put("", "2")
    cache.get("1")
    cache.put("", "3")
    pc()


#%%
import functools
from pathlib import Path

# TODO: we could also put this to the same class?!
class CacheDecorator:
    def __init__(self, dirpath: Path, logfunc=print) -> None:
        self.dirpath = dirpath
        self.store = PickleStore(dirpath)
        self.logfunc = logfunc

    def __call__(self, func):
        @functools.wraps(func)
        def new_func(*args, **kwargs):
            resolved_args = resolve_args(func, *args, **kwargs)
            hash_str = hash_obj(resolved_args)

            # Use a filelock to prevent two parallel processes working on the same hash value.
            # FIXME: Use the _get_lock function from Cache
            with filelock.SoftFileLock(self.dirpath / f"{hash_str}.lock"):
                try:
                    out = self.store.get(hash_str)
                    self.logfunc("cache hit")
                except FileNotFoundError:
                    self.logfunc("cache miss")
                    out = func(*args, **kwargs)
                    self.store.put(out, hash_str)
            return out

        return new_func


def test_persistent_cache():
    logs = []
    cache = CacheDecorator(
        dirpath=Path(__file__).parent / "temp_cache",
        logfunc=logs.append,
    )
    cache.store.clear()

    @cache
    def hi(x, y=123):
        time.sleep(2)
        return x

    out = [hi(2), hi(3), hi(2, y=123)]
    assert_that(out).is_equal_to([2, 3, 2])
    assert_that(logs).is_equal_to(["cache miss", "cache miss", "cache hit"])
    cache.store.clear()


if utils.run_as_cell(__name__):
    test_persistent_cache()
