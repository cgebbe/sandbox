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


test_hash_obj()

# %%
from pathlib import Path
import pickle
import logging
import shutil
import tempfile
import geopandas as gpd


LOGGER = logging.getLogger(__name__)


class PickleStore:
    def __init__(self, dirpath: Path):
        assert isinstance(dirpath, Path)
        dirpath = dirpath.absolute()
        if dirpath.exists():
            # TODO: accept existing dirs only if
            # args are as expected?!
            # ... has metadata and corresponding files
            # ... is empty
            LOGGER.warning(f"{dirpath=} already exists")
        else:
            dirpath.mkdir(exist_ok=True)

        self.dirpath = dirpath

    def get(self, hash_value):
        p = self._get_path(hash_value)
        with open(p, "rb") as f:
            return pickle.load(f)

    def put(self, obj, hash_value: str):
        p = self._get_path(hash_value)
        if p.exists():
            raise FileExistsError(f"{hash_value=} already exists, but should not.")
        with open(p, "wb") as f:
            return pickle.dump(obj, f)

    def clear(self):
        shutil.rmtree(self.dirpath)
        self.dirpath.mkdir()

    def _get_path(self, hash_value: str):
        return self.dirpath / f"{hash_value}.pkl"


from geopandas.testing import assert_geodataframe_equal


def test_pickle_store():
    dirpath = tempfile.mkdtemp()
    obj = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    hsh = "some_random_hash"
    store = PickleStore(Path(dirpath))

    store.put(obj, hsh)
    reloaded = store.get(hsh)
    assert_geodataframe_equal(obj, reloaded)

    store.clear()  # to remove the tempdir, not strictly necessary


test_pickle_store()

#%%
import functools


class PersistentCache:
    def __init__(self, dirpath: Path, logfunc=print) -> None:
        self.store = PickleStore(dirpath)
        self.logfunc = logfunc

    def __call__(self, func):
        @functools.wraps(func)
        def new_func(*args, **kwargs):
            resolved_args = resolve_args(func, *args, **kwargs)
            hash_val = hash_obj(resolved_args)
            try:
                res = self.store.get(hash_val)
                self.logfunc("cache hit")
            except FileNotFoundError:
                self.logfunc("cache miss")
                res = func(*args, **kwargs)
                self.store.put(res, hash_val)
            return res

        return new_func


def test_persistent_cache():
    logs = []
    cache = PersistentCache(
        dirpath=Path(__file__).parent / "mycache",
        logfunc=logs.append,
    )
    cache.store.clear()

    @cache
    def hi(x, y=123):
        return x

    out = [hi(2), hi(3), hi(2, y=123)]
    assert_that(out).is_equal_to([2, 3, 2])
    assert_that(logs).is_equal_to(["cache miss", "cache miss", "cache hit"])


test_persistent_cache()

# %%


# # %% [markdown]
# # # TODO: MetaDataStorage

# # %%
# from sqlalchemy import create_engine
# engine = create_engine("sqlite:///sqlite.db", echo=True)


# # %%
# from typing import List
# from typing import Optional
# from sqlalchemy import ForeignKey
# from sqlalchemy import String
# from sqlalchemy.orm import DeclarativeBase
# from sqlalchemy.orm import Mapped
# from sqlalchemy.orm import mapped_column
# from sqlalchemy.orm import relationship

# class User(DeclarativeBase):
#     __tablename__ = "user_account"

#     id: Mapped[int] = mapped_column(primary_key=True)
#     name: Mapped[str] = mapped_column(String(30))
#     fullname: Mapped[Optional[str]]

#     def __repr__(self) -> str:
#         return f"User(id={self.id!r}, name={self.name!r}, fullname={self.fullname!r})"


# from sqlalchemy.orm import Session
# with Session(engine) as session:
#      spongebob = User(
#          name="spongebob",
#          fullname="Spongebob Squarepants",
#      )
#      session.add_all([spongebob])
#      session.commit()


# # %%
# import sqlite3

# class SQLDB:
#     """ContextManager for sqlite3 DB.

#     Taken from https://codereview.stackexchange.com/a/182706/202014
#     """
#     def __init__(self, path):
#         self.connection = sqlite3.connect(path)
#         self.cursor=self.connection.cursor()

#     def __enter__(self):
#         return self

#     def __exit__(self, ext_type, exc_value, traceback):
#         self.cursor.close()
#         if isinstance(exc_value, Exception):
#             self.connection.rollback()
#         else:
#             self.connection.commit()
#         self.connection.close()


# DB_PATH="foo.sqlite"
# with SQLDB(DB_PATH) as db:
#     df.to_sql("my_table",con=db, if_exists="replace")

# # %%
# !ls -lah

# # %%
# with SqliteDict("foo.sqlite") as db:
#     print(list(db.items()))
