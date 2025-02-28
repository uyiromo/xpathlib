from __future__ import annotations

import json
import os
import pathlib
import typing
from copy import deepcopy
from fnmatch import fnmatch
from logging import Logger
from os import remove
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from .io import TxRawIO, TxTextIO
from .logger import args2str, getlg
from .ssh import SSHContext, SSHFile, scp_from, scp_to, ssh_ls, ssh_mv, ssh_rm, ssh_stat

lg: Logger = getlg()

# A marker indicating a file exists ONLY on remote
DEVZERO: pathlib.Path = pathlib.Path("/dev/zero")

# A marker indicating a file has been removed from local
DEVNULL: pathlib.Path = pathlib.Path("/dev/null")


class Path(os.PathLike):
    """pathlib.Path wrapper

    see: https://docs.python.org/3/library/pathlib.html#pure-paths

    """

    def __init__(self):
        self._sshctxt: SSHContext
        self._cacheroot: pathlib.Path
        self._remoteroot: pathlib.Path
        self._relpath: pathlib.Path
        self._always_keep: List[str]

        raise NotImplementedError("xpathlib.Path must be initialized by xpathlib.Path.init()")

    @property
    def lpath(self) -> pathlib.Path:
        """pathlib.Path: local cache path"""
        return self._cacheroot / self._relpath

    @property
    def rpath(self) -> pathlib.Path:
        """pathlib.Path: remote path"""
        return self._remoteroot / self._relpath

    #
    #
    # cache-related ops
    #
    #
    def _is_keep(self) -> bool:
        """True if a file pointed by `self` must be kept in local cache"""
        lp: Path = self.lpath
        return any((lp.full_match(pat) or fnmatch(lp.name, pat) or (lp.name == pat)) for pat in self._always_keep)

    def _is_cached(self) -> bool:
        """True if a file pointed by `self` has been cached (exists and not DEVZERO)"""
        lp: Path = self.lpath
        return lp.exists() and (lp.is_file() or lp.is_dir())

    def _is_removed(self) -> bool:
        """True if local file has been removed (symlink to DEVNULL)"""
        lp: Path = self.lpath
        return lp.exists() and (lp.is_symlink() and lp.readlink() == DEVNULL)

    def _cache(self):
        """cache a file pointed by `self`

        - If NOT exists on both local and cache, `touch`
        - If exists ONLY on local, do nothing (cache is already LATEST)
        - If exists ONLY on remote, retrieve

        """
        lg.debug(args2str(locals()))

        lp: Path = self.lpath
        rp: Path = self.rpath
        lg.debug(f"lp: '{lp}', rp: '{rp}'")

        if not lp.exists():
            # marker does not exist (= not exist on remote)
            lg.debug("lp does not exist. touch()")
            lp.touch()
        else:
            # marker exists (= exist on remote)
            if self._is_cached():
                lg.debug("lp is cached. do nothing")
                pass
            else:
                lg.debug("lp is UNcached. retrieve from remote")
                lp.unlink(missing_ok=True)
                scp_from(self._sshctxt, rp, lp)

        return

    def _uncache(self):
        """uncache a file pointed by `self`"""
        lg.debug(args2str(locals()))
        assert self._is_cached(), f"{self.lpath} is not cached!"

        lp: Path = self.lpath

        remove(lp)
        lp.symlink_to(DEVZERO)

        return

    def _remove(self) -> None:
        """Remove local file and mark it as removed (symlink to DEVNULL)"""
        lp: Path = self.lpath

        # remove local files
        dirpath: str
        dirnames: List[str]
        filenames: List[str]
        for dirpath, dirnames, filenames in os.walk(lp, topdown=False):
            for filename in filenames:
                p: pathlib.Path = pathlib.Path(os.path.join(dirpath, filename))
                if p.is_symlink():
                    p.unlink()
                else:
                    remove(p)

            for dirname in dirnames:
                os.rmdir(os.path.join(dirpath, dirname))

        # mark as removed
        lp.symlink_to(DEVNULL)

        return

    def _sync_core(self) -> None:
        """core ops for sync(), do ssh ops without WOL"""

        # each file state is one of the following:
        # 1. removed
        #    -> remove from remote and unmark as REMOVED
        # 2. dir
        #    -> recursively sync
        # 3. file
        #    -> sync if cached
        if self._is_removed():
            ssh_rm(self._sshctxt, self.rpath, do_wol=False)
            self.lpath.unlink(missing_ok=False)

        elif self.lpath.is_dir():
            lg.info(f"_sync_core: {self}")
            for f in self.iterdir():
                f._sync_core()

        else:
            if self._is_cached():
                scp_to(self._sshctxt, self.lpath, self.rpath, do_wol=False)

                # delete cache if necessary
                if self._is_keep():
                    pass
                else:
                    self._uncache()
            else:
                # do nothing if not yet cached
                pass

        return

    def sync(self) -> None:
        """Sync all files under `self`"""
        lg.debug(args2str(locals()))
        self._sshctxt.wol()
        self._sync_core()

        return

    def _buildcache_core(self) -> None:
        """core ops for _buildcache(), do ssh ops without WOL"""
        lg.debug(args2str(locals()))

        self.lpath.mkdir(exist_ok=True, parents=True)

        file: SSHFile
        for file in ssh_ls(self._sshctxt, self.rpath, do_wol=False):
            lg.debug(f"file: {file}")

            p: Path = self / file.name
            if file.isdir:
                lg.info(f"_buildcache_core: {p}")
                p._buildcache_core()
            else:
                # if not exists, create a marker
                lp: Path = p.lpath
                rp: Path = p.rpath

                # mark as "exists on remote"
                if not lp.exists():
                    lp.symlink_to(DEVZERO)
                else:
                    pass

                # retrieve if required
                keep: bool = p._is_keep()
                lg.debug(f"keep: {keep}")
                if keep:
                    lg.info(f"_buildcache_core: keep::{p}")
                    scp_from(self._sshctxt, rp, lp, do_wol=False)
                else:
                    pass

        # end for

        return

    def _buildcache(self) -> None:
        """Build local cache for a file pointed by `self`"""
        lg.debug(args2str(locals()))
        self._sshctxt.wol()
        self._buildcache_core()

        return

    def _clone(self) -> Path:
        new: Path = object.__new__(type(self))
        new._sshctxt: SSHContext = self._sshctxt
        new._cacheroot: pathlib.Path = self._cacheroot
        new._remoteroot: pathlib.Path = self._remoteroot
        new._relpath: Path = self._relpath
        new._always_keep: List[str] = self._always_keep

        return new

    @classmethod
    def init(
        cls,
        config: str | os.PathLike,
        relpath: str | os.PathLike,
        always_keep: List[str],
    ) -> Path:
        """Create new Path and build local cache

        Args:
            config (:obj:`str` | `os.PathLike`): config file (.json)
            relpath (:obj:`str` | `os.PathLike`): relative path to cacheroot/remoteroot
            always_keep (:obj:`List[str]`): file name patterns always kept in local cache

        Returns:
            Path: Path object

        Notes:
            config has the following keys:
            - SSHHOST
              - ssh host name (should exist in ~/.ssh/config)
            - WOLMACADDR
              - Wake-on-LAN MAC address
            - WOLIPADDR
              - Wake-on-LAN IP address
            - WOLBROADCASTADDR
              - Wake-on-LAN broadcast address
            - CACHEROOT
              - local cache root directory
            - REMOTEROOT
              - remote root directory
            example:
                {
                    "SSHHOST": "myserver",
                    "WOLMACADDR": "00:11:22:33:44:55",
                    "WOLIPADDR": "192.168.xxx.yyy",
                    "WOLBROADCASTADDR": "192.168.xxx.255",

                    "CACHEROOT": "/home/workspace",
                    "REMOTEROOT": "/mnt/backup"
                }

        """
        lg.debug(args2str(locals()))

        # parse config file
        data: Dict[str, str] = json.loads(pathlib.Path(config).expanduser().read_text())

        # to Path
        remoteroot: pathlib.Path = pathlib.Path(data["REMOTEROOT"]).expanduser()
        cacheroot: pathlib.Path = pathlib.Path(data["CACHEROOT"]).expanduser()
        relpath: pathlib.Path = pathlib.Path(relpath).expanduser()

        sshctxt: SSHContext = SSHContext(
            data["SSHHOST"],
            data["WOLMACADDR"],
            data["WOLIPADDR"],
            data["WOLBROADCASTADDR"],
        )
        lg.debug(f"config: {str(config)}")

        self: Path = object.__new__(cls)
        self._sshctxt = sshctxt
        self._cacheroot = cacheroot
        self._remoteroot = remoteroot
        self._relpath = relpath
        self._always_keep = deepcopy(always_keep)
        self._always_keep.append(".built")

        #
        # build local cache
        #
        marker: pathlib.Path = cacheroot / relpath / ".built"
        if not marker.exists():
            self._buildcache()
        marker.touch()

        return self

    #
    # os.PathLike
    #
    def __fspath__(self) -> str:
        return self.lpath.__fspath__()

    def __str__(self) -> str:
        return str(self._relpath)

    #
    # PurePath: General properties
    #
    def __eq__(self, other: Path) -> bool:
        assert isinstance(other, Path), f"unsupported operand type(s) for ==: 'xpathlib.Path' and '{type(other)}'"
        return self.__hash__() == other.__hash__()

    def __hash__(self) -> int:
        return hash(self.lpath)

    def __lt__(self, other: Path) -> bool:
        assert isinstance(other, Path), f"unsupported operand type(s) for <: 'xpathlib.Path' and '{type(other)}'"
        return self.lpath < other.lpath

    #
    # PurePath: Operators
    #
    def __truediv__(self, subpath: str | os.PathLike) -> Path:
        new: Path = self._clone()
        new._relpath: Path = self._relpath / subpath

        return new

    #
    # PurePath: Accessing individual parts
    #
    @property
    def parts(self) -> Tuple[str, ...]:
        return self._relpath.parts

    #
    # PurePath: Methods and properties
    #
    @property
    def parser(self) -> pathlib.PurePosixPath:
        return self._relpath.parser

    @property
    def drive(self) -> str:
        return self._relpath.drive

    @property
    def root(self) -> str:
        return self._relpath.root

    @property
    def anchor(self) -> str:
        return self._relpath.anchor

    @property
    def parents(self) -> pathlib.Path:
        return self._relpath.parents

    @property
    def parent(self) -> pathlib.Path:
        return self._relpath.parent

    @property
    def name(self) -> str:
        return self._relpath.name

    @property
    def suffix(self) -> str:
        return self._relpath.suffix

    @property
    def suffixes(self) -> List[str]:
        return self._relpath.suffixes

    @property
    def stem(self) -> str:
        return self._relpath.stem

    def as_posix(self) -> str:
        raise NotImplementedError()

    def is_absolute(self) -> bool:
        raise NotImplementedError()

    def is_relative_to(self, other: str) -> bool:
        raise NotImplementedError()

    def joinpath(self, *args: str) -> Path:
        raise NotImplementedError()

    def full_match(self, pattern: str, *, case_sensitive: Optional[bool] = None) -> bool:
        return self._relpath.full_match(pattern, case_sensitive=case_sensitive)

    def match(self, pattern: str, *, case_sensitive: Optional[bool] = None) -> bool:
        return self._relpath.match(pattern)

    def relative_to(self, other: Path, walk_up: bool = False) -> Path:
        raise NotImplementedError()

    def with_name(self, name: str) -> Path:
        new: Path = self._clone()
        new._relpath = self._relpath.with_name(name)

        return new

    def with_stem(self, stem: str) -> Path:
        new: Path = self._clone()
        new._relpath = self._relpath.with_stem(stem)

        return new

    def with_suffix(self, suffix: str) -> Path:
        new: Path = self._clone()
        new._relpath = self._relpath.with_suffix(suffix)

        return new

    def with_segments(self, segments: List[str]) -> Path:
        raise NotImplementedError()

    #
    # Concrete pathes: Parsing and generating URIs
    #
    @classmethod
    def from_uri(cls, uri: str) -> Path:
        raise NotImplementedError()

    def as_uri(self) -> str:
        raise NotImplementedError()

    #
    # Concrete pathes: Expanding and resolving paths
    #
    @classmethod
    def home(cls) -> Path:
        raise NotImplementedError()

    def expanduser(self) -> Path:
        new: Path = self._clone()
        new._relpath = self._relpath.expanduser()

        return new

    @classmethod
    def cwd(cls) -> Path:
        raise NotImplementedError()

    def absolute(self) -> Path:
        raise NotImplementedError()

    def resolve(self, strict: bool = False) -> Path:
        new: Path = self._clone()
        new._relpath = self._relpath.resolve(strict=strict)

        return new

    def readlink(self) -> Path:
        new: Path = self._clone()
        new._relpath = self._relpath.readlink()

        return new

    #
    # Concrete pathes: Querying file type and status
    #
    def stat(self, *, follow_symlinks: bool = True) -> os.stat_result:
        if self._is_cached():
            return self.lpath.stat()
        else:
            return ssh_stat(self._sshctxt, self.rpath)

    def lstat(self) -> os.stat_result:
        raise NotImplementedError()

    def exists(self) -> bool:
        """Return True if the file exists and NOT removed"""
        return self._is_cached() or (self._lpath.exists() and not self._is_removed())

    def is_file(self, *, follow_symlinks: bool = True) -> bool:
        assert self._is_cached(), f"{self.lpath} is not cached!"
        return self.lpath.is_file()

    def is_dir(self, *, follow_symlinks: bool = True) -> bool:
        assert self._is_cached(), f"{self.lpath} is not cached!"
        return self.lpath.is_dir()

    def is_symlink(self) -> bool:
        raise NotImplementedError()

    def is_junction(self) -> bool:
        raise NotImplementedError()

    def is_mount(self) -> bool:
        raise NotImplementedError()

    def is_socket(self) -> bool:
        raise NotImplementedError()

    def is_fifo(self) -> bool:
        raise NotImplementedError()

    def is_block_device(self) -> bool:
        raise NotImplementedError()

    def is_char_device(self) -> bool:
        raise NotImplementedError()

    def samefile(self, other: Path) -> bool:
        raise NotImplementedError()

    #
    # Concrete pathes: Reading and writing files
    #
    def open(
        self,
        mode: str = "r",
        buffering: int = -1,
        encoding: Optional[str] = None,
        errors: Optional[str] = None,
        newline: Optional[str] = None,
    ) -> typing.IO:
        self._cache()

        if "b" in mode:
            return TxRawIO(self.lpath, mode, buffering, None, None, None)
        else:
            return TxTextIO(self.lpath, mode, buffering, encoding, errors, newline)

    def read_text(
        self, encoding: Optional[str] = None, errors: Optional[str] = None, newline: Optional[str] = None
    ) -> str:
        with self.open("rt", encoding=encoding, errors=errors, newline=newline) as f:
            return f.read()

    def read_bytes(self) -> bytes:
        with self.open("rb") as f:
            return f.read()

    def write_text(
        self,
        data: str,
        encoding: Optional[str] = None,
        errors: Optional[str] = None,
        newline: Optional[str] = None,
    ) -> int:
        with self.open("wt", encoding=encoding, errors=errors, newline=newline) as f:
            return f.write(data)

    def write_bytes(self, data: bytes) -> int:
        with self.open("wb") as f:
            return f.write(data)

    #
    # Concrete pathes: Reading directories
    #
    def iterdir(self) -> Iterable[Path]:
        for p in self.lpath.iterdir():
            yield self / p.name

    def glob(
        self,
        pattern: str,
        *,
        case_sensitive: Optional[bool] = None,
        recurse_symlinks: bool = False,
    ) -> Iterable[Path]:
        raise NotImplementedError()

    def rglob(
        self,
        pattern: str,
        *,
        case_sensitive: Optional[bool] = None,
        recurse_symlinks: bool = False,
    ) -> Iterable[Path]:
        raise NotImplementedError()

    def walk(
        self,
        top_down: bool = True,
        on_error: Optional[Callable[[OSError], None]] = None,
        follow_symlinks: bool = False,
    ) -> Iterable[Tuple[Path, List[str], List[str]]]:
        raise NotImplementedError()

    #
    # Concrete pathes: Creating files and directories
    #
    def touch(self, mode: int = 0o666, exist_ok: bool = True) -> None:
        """touch(). retrieve self._relpath if has not been retrieved"""
        lg.debug(args2str(locals()))

        self._cache()
        self.lpath.touch(mode=mode, exist_ok=exist_ok)

    def mkdir(self, mode: int = 0o777, parents: bool = False, exist_ok: bool = False) -> None:
        lpath: Path = self.lpath
        lpath.mkdir(mode=mode, parents=parents, exist_ok=exist_ok)

        return

    def symlink_to(self, target: str | os.PathLike, target_is_directory: bool = False) -> None:
        raise NotImplementedError()

    def hardlink_to(self, target: str | os.PathLike) -> None:
        raise NotImplementedError()

    #
    # Renaming and deleting
    #
    # def rename(self, target: str | os.PathLike) -> None:
    def rename(self, target: Path) -> Path:
        """rename() both on local and remote"""
        ssh_mv(self._sshctxt, self.rpath, target.rpath)
        self.lpath.rename(target.lpath)

        return target

    # def replace(self, target: str | os.PathLike) -> None:
    def replace(self, target: Path) -> Path:
        ssh_mv(self._sshctxt, self.rpath, target.rpath)
        self.lpath.replace(target.lpath)

        return target

    def unlink(self, missing_ok: bool = False) -> None:
        lp: pathlib.Path = self.lpath

        # remove and mark as REMOVED
        lp.unlink(missing_ok=missing_ok)
        lp.symlink_to(DEVNULL)

        return

    def rmdir(self) -> None:
        lp: pathlib.Path = self.lpath

        # remove and mark as REMOVED
        lp.rmdir()
        lp.symlink_to(DEVNULL)

        return

    #
    # Concrete pathes: Permissions and ownership
    #
    def owner(self, *, follow_symlinks: bool = True) -> str:
        raise NotImplementedError()

    def group(self, *, follow_symlinks: bool = True) -> str:
        raise NotImplementedError()

    def chmod(self, mode: int, *, follow_symlinks: bool = True) -> None:
        raise NotImplementedError()

    def lchmod(self, mode: int) -> None:
        raise NotImplementedError()


#
# register as a member of os.PathLike
#
os.PathLike.register(Path)
