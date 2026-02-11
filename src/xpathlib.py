from __future__ import annotations

import fnmatch
import functools
import logging
import pathlib
import re
import stat
import types
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, IntEnum
from threading import Lock

import paramiko
from paramiko import SFTPAttributes, SFTPClient
from paramiko.client import SSHClient

logger: logging.Logger = logging.getLogger('xpathlib')

# Logging formatters
FMT_INFO = logging.Formatter('%(asctime)s | %(name)s | %(message)s', datefmt='%H:%M:%S')
FMT_DEBUG = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')


def INFO(msg: str) -> None:
    logger.info(msg, stacklevel=2)


def DEBUG(msg: str) -> None:
    logger.debug(msg, stacklevel=2)


def ERROR(msg: str) -> None:
    logger.error(msg, stacklevel=2)


# tx
SUFFIX_PART: str = '.xpathlib_part'


def resolve_tx(path: pathlib.Path) -> pathlib.Path:
    return path.with_suffix(path.suffix + SUFFIX_PART)


class LogLevel(Enum):
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


@dataclass
class SSHConfig:
    host: str
    port: int
    username: str
    identity: pathlib.Path


@dataclass
class Context(AbstractContextManager):  # pyright: ignore[reportMissingTypeArgument]
    ssh_config: SSHConfig
    cachedir: pathlib.Path
    remotedir: pathlib.Path

    logger: logging.Logger = logger

    depth: int = 0
    lock: Lock = Lock()

    ssh_client: SSHClient | None = None
    sftp_client: SFTPClient | None = None

    keeps: list[re.Pattern[str]] = field(default_factory=list)

    # for sync
    # already created remote directories
    already_exist_dirs: set[pathlib.Path] = field(default_factory=set)

    def is_xpath(self, path: pathlib.Path) -> bool:
        """Return True if the given path is inside the xpathlib cachedir."""
        cachedir_s: str = str(self.cachedir)
        if not cachedir_s.endswith('/'):
            cachedir_s += '/'

        return str(path).startswith(cachedir_s)

    def resolve_remote(self, local_path: pathlib.Path) -> pathlib.Path:
        """Resolve the remote path corresponding to the given local path."""
        assert self.is_xpath(local_path)

        relative_path: pathlib.Path = local_path.relative_to(self.cachedir)
        remote_path: pathlib.Path = self.remotedir / relative_path

        return remote_path

    def should_keep(self, filename: str) -> bool:
        return any(pat.match(filename) for pat in self.keeps)

    def push(self, funcname: str, *, is_debug: bool = False) -> None:
        if not is_debug:
            INFO(f'>> {funcname}')
        else:
            pass

        DEBUG(f'>> {funcname}')

        return

    def pop(self, funcname: str, *, is_debug: bool = False) -> None:
        if not is_debug:
            INFO(f'<< {funcname}')
        else:
            pass

        DEBUG(f'<< {funcname}')

        return

    def __enter__(self) -> Context:
        with self.lock:
            if self.depth == 0:
                self.ssh_client = SSHClient()

                # WARNING: insecure, only for dev
                self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

                self.ssh_client.connect(
                    hostname=self.ssh_config.host,
                    port=self.ssh_config.port,
                    username=self.ssh_config.username,
                    key_filename=str(self.ssh_config.identity),
                )

                self.sftp_client = self.ssh_client.open_sftp()

            else:
                # already connected
                pass

            self.depth += 1

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> bool:
        assert self.sftp_client is not None
        assert self.ssh_client is not None

        with self.lock:
            self.depth -= 1

            if self.depth > 0:
                return False
            else:
                # close connections
                self.sftp_client.close()
                self.ssh_client.close()

                self.sftp_client = None
                self.ssh_client = None

        return False

    def sftp_get(self, local_path: pathlib.Path, remote_path: pathlib.Path) -> None:
        assert self.sftp_client is not None

        local_path_tx: pathlib.Path = resolve_tx(local_path)
        self.sftp_client.get(str(remote_path), str(local_path_tx))

        pathlib_rename(local_path_tx, local_path)

        return

    def sftp_put(self, local_path: pathlib.Path, remote_path: pathlib.Path) -> None:
        assert self.sftp_client is not None

        # create parents on remote
        for p in reversed(remote_path.parents):
            if p in self.already_exist_dirs:
                continue

            try:
                # if success, dir exists
                self.sftp_client.stat(str(p))
            except FileNotFoundError:
                DEBUG(f'Creating remote directory: {p}')
                self.sftp_client.mkdir(str(p))

            self.already_exist_dirs.add(p)

        remote_path_tx: pathlib.Path = resolve_tx(remote_path)
        self.sftp_client.put(str(local_path), str(remote_path_tx))

        self.sftp_client.posix_rename(str(remote_path_tx), str(remote_path))

        return

    def sftp_rename(self, remote_oldpath: pathlib.Path, remote_newpath: pathlib.Path) -> None:
        assert self.sftp_client is not None

        self.sftp_client.posix_rename(str(remote_oldpath), str(remote_newpath))

        return

    def sftp_remove(self, remote_path: pathlib.Path) -> None:
        assert self.sftp_client is not None

        self.sftp_client.remove(str(remote_path))

        return

    def sftp_rmdir(self, remote_path: pathlib.Path) -> None:
        assert self.sftp_client is not None

        self.sftp_client.rmdir(str(remote_path))
        self.already_exist_dirs.discard(remote_path)

        return


# Global context
ctx: Context | None = None


#
# cache ops
#
# Modified   : original file on REMOTE, LOCAL one is modified
# Shared     : original file on REMOTE, LOCAL one is the same as REMOTE one
# Invalidated: original file on REMOTE, LOCAL one is NOT cached
# New        : original file on LOCAL
# Deleted    : original file on REMOTE, LOCAL one is deleted
class CacheState(IntEnum):
    M = 0o664
    S = 0o666
    I = 0o100  # noqa: E741 (Ambiguous name)
    N = 0o644
    D = 0o000

    @staticmethod
    def from_path(p: pathlib.Path) -> CacheState:
        perm: int = p.stat().st_mode & 0o777
        return CacheState(perm)


# state transitions:
#
# |  state \ op | read | write | delete |
# |           M |    M |     M |      D |
# |           S |    S |     M |      D |
# |           I |    S |     M |      D |
# |           N |    N |     N |      D |
# |           D |    S |     M |      D |
# | (not exist) |    - |     N |      - |


def is_read(mode: str) -> bool:
    return 'r' in mode


def is_write(mode: str) -> bool:
    return ('w' in mode) or ('a' in mode) or ('x' in mode)


def is_append(mode: str) -> bool:
    return 'a' in mode


def is_remotefile(path: pathlib.Path) -> bool:
    """Return TRUE if the file is shared with remote"""
    state: CacheState = CacheState.from_path(path)
    return state in {
        CacheState.M,
        CacheState.S,
        CacheState.I,
        CacheState.D,
    }


def is_remotedir(path: pathlib.Path) -> bool:
    """Return TRUE if any file is shared with remote"""
    b: bool = False

    for p in path.iterdir():
        if p.is_dir():
            b |= is_remotedir(p)
        elif p.is_file():
            b |= is_remotefile(p)
        else:
            pass

    return b


def transit_cachestate(path: pathlib.Path, mode: str) -> None:
    if not path.exists():
        path.touch(mode=CacheState.N)
    else:
        current: CacheState = CacheState.from_path(path)

        match current:
            case CacheState.M:
                path.chmod(CacheState.M)
            case CacheState.S:
                if is_write(mode):
                    path.chmod(CacheState.M)
                else:
                    path.chmod(CacheState.S)
            case CacheState.I:
                if is_write(mode):
                    path.chmod(CacheState.M)
                else:
                    path.chmod(CacheState.S)
            case CacheState.N:
                path.chmod(CacheState.N)
            case CacheState.D:
                if is_write(mode):
                    path.chmod(CacheState.M)
                else:
                    path.chmod(CacheState.S)
            case _:
                raise RuntimeError(f'Unsupported file perms: {path} ({oct(current)})')

        # end match

    return


def _build_core(cachedir: pathlib.Path, remotedir: pathlib.Path) -> None:
    assert ctx is not None
    assert ctx.sftp_client is not None

    ctx.push('_build_core', is_debug=True)

    attr: SFTPAttributes
    for attr in ctx.sftp_client.listdir_attr(str(remotedir)):
        local_path: pathlib.Path = cachedir / attr.filename
        remote_path: pathlib.Path = remotedir / attr.filename
        DEBUG(f"  local_path='{local_path}' remote_path='{remote_path}'")

        # 1. If dir, recurse
        # 2. If file
        #    a. If exists, do nothing (local file is ALWAYS up to date)
        #    b. If not, touch as "I"
        #       and if keep, download and set to "S"
        assert attr.st_mode is not None
        if stat.S_ISDIR(attr.st_mode):
            DEBUG('  is directory')
            _build_core(local_path, remote_path)
        elif stat.S_ISREG(attr.st_mode):
            DEBUG('  is regular file')

            local_path.parent.mkdir(parents=True, exist_ok=True)
            if local_path.exists():
                DEBUG('  already cached, skipping')
                pass
            else:
                DEBUG('  not cached, creating NOTCACHED marker')
                local_path.touch(mode=CacheState.I)

                if ctx.should_keep(attr.filename):
                    INFO(f"  keep: '{remote_path}'")
                    DEBUG('  matches keep pattern')
                    ctx.sftp_get(local_path, remote_path)
                    local_path.chmod(CacheState.S)
                else:
                    DEBUG('  does not match keep pattern, leaving as NOTCACHED')
        else:
            raise RuntimeError(f'Unsupported file type: {remote_path} ({attr.st_mode})')

    ctx.pop('_build_core', is_debug=True)
    return


def build(cachedir: pathlib.Path, remotedir: pathlib.Path, *, force: bool = False) -> None:
    """
    Build local cache

    Args:
        cachedir (pathlib.Path): The directory to use for caching.
        remotedir (pathlib.Path): The remote directory to cache from.
        force (bool): Whether to force rebuild the cache. (Default: False)

    """
    assert ctx is not None
    ctx.push('build')
    INFO(f'   cachedir: "{cachedir}"')
    INFO(f'  remotedir: "{remotedir}"')
    INFO(f'      force: "{force}"')

    marker: pathlib.Path = cachedir / '.built'
    if marker.exists() and not force:
        INFO('  Cache already built, skipping.')
    else:
        INFO('  Building cache...')

        s: datetime = datetime.now()
        with ctx:
            _build_core(cachedir, remotedir)
        e: datetime = datetime.now()

        elapsed: float = (e - s).total_seconds()
        INFO(f'  Built cache in {elapsed:.2f} seconds.')
        marker.touch()

    ctx.pop('build')
    return


def cache(local_path: pathlib.Path, remote_path: pathlib.Path) -> None:
    assert ctx is not None
    ctx.push('cache', is_debug=True)
    DEBUG(f'   local_path: {local_path}')
    DEBUG(f'  remote_path: {remote_path}')

    if local_path.exists():
        state: CacheState = CacheState.from_path(local_path)
        DEBUG(f'  existing file with state "{state.name}"')

        match state:
            case CacheState.M:
                DEBUG('  modified. do nothing')
                pass
            case CacheState.S:
                DEBUG('  shared. do nothing')
                pass
            case CacheState.I:
                DEBUG('  invalidated. caching...')
                with ctx:
                    ctx.sftp_get(local_path, remote_path)

                local_path.chmod(CacheState.S)
            case CacheState.N:
                DEBUG('  new. do nothing')
                pass
            case CacheState.D:
                DEBUG('  deleted. create as new')
                pathlib_unlink(local_path)
                pathlib_touch(local_path, mode=CacheState.N)
                pass
            case _:
                raise RuntimeError(f'Unsupported file state: {local_path} ({state.name})')
    else:
        DEBUG('  not exists. create as new')
        local_path.parent.mkdir(parents=True, exist_ok=True)
        pathlib_touch(local_path, mode=CacheState.N)

    ctx.pop('cache', is_debug=True)
    return


def _sync_file(local_path: pathlib.Path, remote_path: pathlib.Path, *, dry_run: bool = False) -> None:
    assert ctx is not None

    if local_path.suffix == SUFFIX_PART:
        DEBUG('  skipping .tx file')
        return
    else:
        if dry_run:
            pass
        else:
            state: CacheState = CacheState.from_path(local_path)
            INFO(f"  sync: '{local_path}' -> '{remote_path}' (state={state.name})")
            ctx.sftp_put(local_path, remote_path)

            if ctx.should_keep(str(local_path)):
                DEBUG('  SHOULD keep.')
                DEBUG('  keep modified file as SHARED')
                local_path.chmod(CacheState.S)
            else:
                DEBUG('  NOT keep.')
                pathlib_unlink(local_path)
                pathlib_touch(local_path, mode=CacheState.I)

    return


def _sync_core(local_path: pathlib.Path, remote_path: pathlib.Path, *, dry_run: bool = False) -> None:
    assert ctx is not None
    ctx.push('_sync_core', is_debug=True)
    DEBUG(f'   local_path: "{local_path}"')
    DEBUG(f'  remote_path: "{remote_path}"')
    DEBUG(f'      dry_run: "{dry_run}"')

    for p in local_path.iterdir():
        local_p: pathlib.Path = p
        remote_p: pathlib.Path = remote_path / p.name

        if p.is_dir():
            _sync_core(local_p, remote_p, dry_run=dry_run)
        elif p.is_file():
            state: CacheState = CacheState.from_path(local_p)

            match state:
                case CacheState.M:
                    INFO(f"      modified: '{local_p}' -> '{remote_p}'")
                    _sync_file(local_p, remote_p, dry_run=dry_run)

                case CacheState.S:
                    DEBUG(f"       shared: '{local_p}'")
                    _sync_file(local_p, remote_p, dry_run=dry_run)
                    pass

                case CacheState.I:
                    DEBUG(f"  invalidated: '{local_p}'")
                    pass

                case CacheState.N:
                    INFO(f"          new: '{local_p}' -> '{remote_p}'")
                    _sync_file(local_p, remote_p, dry_run=dry_run)
                case CacheState.D:
                    INFO(f"  deleting remote file: '{remote_p}'")
                    if dry_run:
                        pass
                    else:
                        ctx.sftp_remove(remote_p)
                        pathlib_unlink(p)

                case _:
                    raise RuntimeError(f'Unsupported file state: {local_p} ({state.name})')
        else:
            raise RuntimeError(f'Unsupported file type: {local_p}')

    ctx.pop('_sync_core', is_debug=True)
    return


def sync_xpathlib(*, dry_run: bool = False) -> None:
    """
    Sync local cache to remote
    """
    assert ctx is not None
    ctx.push('sync_xpathlib')

    s: datetime = datetime.now()
    with ctx:
        _sync_core(ctx.cachedir, ctx.remotedir, dry_run=dry_run)
    e: datetime = datetime.now()

    elapsed: float = (e - s).total_seconds()
    INFO(f'  sync in {elapsed:.2f} seconds.')

    ctx.pop('sync_xpathlib')
    return


#
# hooks
#
pathlib_open: callable = pathlib.Path.open
pathlib_touch: callable = pathlib.Path.touch
pathlib_symlink_to: callable = pathlib.Path.symlink_to
pathlib_rename: callable = pathlib.Path.rename
pathlib_replace: callable = pathlib.Path.replace
pathlib_unlink: callable = pathlib.Path.unlink
pathlib_rmdir: callable = pathlib.Path.rmdir


def copy_if(p1: pathlib.Path, p2: pathlib.Path) -> None:
    """Make a copy if p1 exists to p2"""

    p2.parent.mkdir(parents=True, exist_ok=True)
    if p1.exists():
        # use pathlib_* to avoid xpathlib
        with pathlib_open(p1, 'rb') as fr, pathlib_open(p2, 'wb') as fw:
            fw.write(fr.read())
    else:
        pathlib_touch(p2)

    return


class XFile:
    def __init__(self, path: pathlib.Path, mode: str) -> None:
        assert ctx is not None

        self._is_xpath: bool = ctx.is_xpath(path)
        self._path: pathlib.Path = path
        self._mode: str = mode

        self._path_tx: pathlib.Path | None = None

        DEBUG('XFile.__init__:')
        DEBUG(f'      path: {self._path}')
        DEBUG(f'      mode: {self._mode}')
        DEBUG(f'  is_xpath: {self._is_xpath}')

        if not self._is_xpath:
            # out of xpathlib dir
            self._f = pathlib_open(self._path, mode)
        else:
            # in xpathlib dir
            local_path: pathlib.Path = self._path
            remote_path: pathlib.Path = ctx.remotedir / local_path.relative_to(ctx.cachedir)
            cache(local_path, remote_path)

            # transactional update
            if is_write(mode):
                self._path_tx = resolve_tx(self._path)

                # make a copy
                if is_append(mode):
                    copy_if(self._path, self._path_tx)
                else:
                    pass

                self._f = pathlib_open(self._path_tx, mode)
            else:  # read
                self._f = pathlib_open(self._path, mode)

        return

    def __del__(self) -> None:
        self.close()
        return

    def close(self) -> None:
        assert ctx is not None
        DEBUG('XFile.close:')
        DEBUG(f'      path: {self._path}')
        DEBUG(f'      mode: {self._mode}')
        DEBUG(f'  is_xpath: {self._is_xpath}')

        if not self._f.closed:
            self._f.close()

        if not self._is_xpath:
            pass
        else:
            # commit
            if self._path_tx and self._path_tx.exists():
                if self._path.exists():
                    self._path.chmod(self._path.stat().st_mode)
                else:
                    self._path.chmod(CacheState.N)
                transit_cachestate(self._path_tx, self._mode)

                pathlib_replace(self._path_tx, self._path)
            else:
                # here, read-only
                pass

        return

    # context manager
    def __enter__(self) -> XFile:
        assert ctx is not None

        DEBUG('XFile.__enter__')
        DEBUG(f'      path: {self._path}')
        DEBUG(f'      mode: {self._mode}')

        self._f.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> bool:
        assert ctx is not None

        DEBUG('XFile.__exit__')
        DEBUG(f'      path: {self._path}')
        DEBUG(f'      mode: {self._mode}')

        ret: bool = self._f.__exit__(exc_type, exc_value, traceback)
        self.close()

        return ret

    # delegate
    def __getattr__(self, name: str) -> object:
        attr = getattr(self._f, name)

        if callable(attr):
            # keep self to prevent early gc
            @functools.wraps(attr)
            def method(*args, **kwargs):  # pyright: ignore[reportMissingParameterType]
                _self: XFile = self
                return attr(*args, **kwargs)

            return method
        else:
            return attr


def xpathlib_open(
    self: pathlib.Path,
    mode: str = 'r',
    buffering: int = -1,
    encoding: str | None = None,
    errors: str | None = None,
    newline: str | None = None,
    **kwargs,  # pyright: ignore[reportMissingParameterType]
) -> XFile:
    assert ctx is not None
    ctx.push('xpathlib_open', is_debug=True)
    DEBUG(f'       self: {self}')
    DEBUG(f'       mode: {mode}')
    DEBUG(f'  buffering: {buffering}')
    DEBUG(f'   encoding: {encoding}')
    DEBUG(f'     errors: {errors}')
    DEBUG(f'    newline: {newline}')
    DEBUG(f'     kwargs: {kwargs}')

    ctx.pop('xpathlib_open', is_debug=True)
    return XFile(self, mode)


def xpathlib_touch(
    self: pathlib.Path,
    mode: int = 0o666,
    exist_ok: bool = True,
) -> None:
    assert ctx is not None
    ctx.push('xpathlib_touch', is_debug=True)
    DEBUG(f'       self: {self}')
    DEBUG(f'       mode: {oct(mode)}')
    DEBUG(f'   exist_ok: {exist_ok}')

    if not ctx.is_xpath(self):
        pathlib_touch(self, mode=mode, exist_ok=exist_ok)
    else:
        if self.exists():
            if not exist_ok:
                raise FileExistsError(f'File exists: {self}')
            else:
                pass

            cache(self, ctx.resolve_remote(self))
        else:
            self.parent.mkdir(parents=True, exist_ok=True)
            pathlib_touch(self, mode=CacheState.N)

    ctx.pop('xpathlib_touch', is_debug=True)
    return


def xpathlib_symlink_to(
    self: pathlib.Path,
    target: pathlib.Path,
    target_is_directory: bool = False,
) -> None:
    assert ctx is not None
    ctx.push('xpathlib_symlink_to', is_debug=True)
    DEBUG(f'                 self: {self}')
    DEBUG(f'               target: {target}')
    DEBUG(f'  target_is_directory: {target_is_directory}')

    if not ctx.is_xpath(self):
        pathlib_symlink_to(self, target, target_is_directory=target_is_directory)
    else:
        raise NotImplementedError('symlink_to is not supported in xpathlib cachedir')

    ctx.pop('xpathlib_symlink_to', is_debug=True)
    return


def xpathlib_rename(self: pathlib.Path, target: pathlib.Path) -> pathlib.Path:
    assert ctx is not None
    ctx.push('xpathlib_rename', is_debug=True)
    DEBUG(f'       self: {self}')
    DEBUG(f'     target: {target}')

    both_not_xpath: bool = not ctx.is_xpath(self) and not ctx.is_xpath(target)
    both_xpath: bool = ctx.is_xpath(self) and ctx.is_xpath(target)

    if both_not_xpath:
        result = pathlib_rename(self, target)
    elif both_xpath:
        # if M/S/I/D, also rename on remote
        if self.is_file():
            # file
            oldpath: pathlib.Path = ctx.resolve_remote(self)
            newpath: pathlib.Path = ctx.resolve_remote(target)

            if is_remotefile(self):
                DEBUG(f'  is remotefile: {self}')
                INFO(f"  renaming remote file: '{oldpath}' -> '{newpath}'")
                with ctx:
                    ctx.sftp_rename(oldpath, newpath)
            else:
                DEBUG(f'  is localfile: {self}')
                pass

            result = pathlib_rename(self, target)
        else:
            # dir
            if is_remotedir(self):
                # if any file inside is M/S/I/D, need to rename on remote
                oldpath: pathlib.Path = ctx.resolve_remote(self)
                newpath: pathlib.Path = ctx.resolve_remote(target)

                INFO(f"  renaming remote directory: '{oldpath}' -> '{newpath}'")
                with ctx:
                    ctx.sftp_rename(oldpath, newpath)
            else:
                DEBUG(f'  is localdir: {self}')

            result = pathlib_rename(self, target)

    else:
        raise NotImplementedError('cross-xpath rename is not supported')

    ctx.pop('xpathlib_rename', is_debug=True)
    return result


def xpathlib_replace(self: pathlib.Path, target: pathlib.Path) -> pathlib.Path:
    return xpathlib_rename(self, target)


def xpathlib_unlink(self: pathlib.Path, missing_ok: bool = False) -> None:
    assert ctx is not None
    ctx.push('xpathlib_unlink', is_debug=True)
    DEBUG(f'       self: {self}')

    if not ctx.is_xpath(self):
        pathlib_unlink(self, missing_ok=missing_ok)
    else:
        if is_remotefile(self):
            pathlib_unlink(self, missing_ok=missing_ok)
            pathlib_touch(self, mode=CacheState.D)
        elif CacheState.from_path(self) == CacheState.N:
            pathlib_unlink(self, missing_ok=missing_ok)
        else:
            raise RuntimeError(f'Unsupported file perms: {self} ({...})')

    ctx.pop('xpathlib_unlink', is_debug=True)
    return


def xpathlib_rmdir(self: pathlib.Path) -> None:
    assert ctx is not None
    ctx.push('xpathlib_rmdir', is_debug=True)
    DEBUG(f'  self: {self}')

    if not ctx.is_xpath(self):
        pathlib_rmdir(self)
    else:
        if is_remotedir(self):
            remote_path: pathlib.Path = ctx.resolve_remote(self)
            INFO(f"  remote_path: '{remote_path}'")
            with ctx:
                ctx.sftp_rmdir(remote_path)
        else:
            pathlib_rmdir(self)

    ctx.pop('xpathlib_rmdir', is_debug=True)
    return


#
# API
#
def logger_xpathlib() -> logging.Logger:
    """
    Return the logger for xpathlib

    """
    global logger

    return logger


def enable_xpathlib(
    ssh_config: SSHConfig,
    cachedir: pathlib.Path,
    remotedir: pathlib.Path,
    keeps: list[str],
    loglevel: LogLevel = LogLevel.INFO,
    logdir: pathlib.Path | None = None,
) -> pathlib.Path:
    """
    Initialize xpathlib

    Args:
        ssh_config (SSHConfig): The SSH configuration to use.
        cachedir (pathlib.Path): The directory to use for caching.
        remotedir (pathlib.Path): The remote directory to cache from.
        keeps (list[str]): List of regex patterns for files to keep cached.
        loglevel (LogLevel): The logging level to use. (Default: LogLevel.INFO)
        logdir (pathlib.Path | None): The directory to use for logging. If None, logs are only streamed.

    Returns:
        pathlib.Path: The cache directory used.

    """
    global logger
    global ctx

    if not cachedir.is_absolute():
        raise ValueError('"cachedir" must be an absolute path')
    if not remotedir.is_absolute():
        raise ValueError('"remotedir" must be an absolute path')

    ctx = Context(
        ssh_config=ssh_config,
        cachedir=cachedir,
        remotedir=remotedir,
        keeps=[re.compile(fnmatch.translate(pattern)) for pattern in keeps],
    )

    # Set up cache directory
    if not cachedir.exists():
        cachedir.mkdir(parents=True)

    # Setup logger
    logger.setLevel(loglevel.value)

    # Stream
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(FMT_INFO)
    logger.addHandler(stream_handler)

    # Set up logging
    if logdir is not None:
        logdir.mkdir(parents=True, exist_ok=True)
        now: str = datetime.now().strftime('%Y%m%d-%H%M%S')

        fh_info = logging.FileHandler(logdir / f'info.{now}.log')
        fh_info.setLevel(logging.INFO)
        fh_info.setFormatter(FMT_INFO)
        logger.addHandler(fh_info)

        fh_debug = logging.FileHandler(logdir / f'debug.{now}.log')
        fh_debug.setLevel(logging.DEBUG)
        fh_debug.setFormatter(FMT_DEBUG)
        logger.addHandler(fh_debug)
    else:
        pass

    INFO('xpathlib initialized')
    INFO(f'  cachedir: {cachedir}')
    INFO(f'  loglevel: {loglevel.name}')
    INFO(f'    logdir: {logdir}')
    INFO('     keeps:')
    for pattern in ctx.keeps:
        INFO(f'       {pattern}')

    INFO('Building cache...')
    build(ctx.cachedir, ctx.remotedir)

    # monkey patch
    pathlib.Path.open = xpathlib_open  # pyright: ignore[reportAttributeAccessIssue]
    pathlib.Path.touch = xpathlib_touch  # pyright: ignore[reportAttributeAccessIssue]
    pathlib.Path.symlink_to = xpathlib_symlink_to  # pyright: ignore[reportAttributeAccessIssue]
    pathlib.Path.rename = xpathlib_rename  # pyright: ignore[reportAttributeAccessIssue]
    pathlib.Path.replace = xpathlib_replace  # pyright: ignore[reportAttributeAccessIssue]
    pathlib.Path.unlink = xpathlib_unlink  # pyright: ignore[reportAttributeAccessIssue]
    pathlib.Path.rmdir = xpathlib_rmdir  # pyright: ignore[reportAttributeAccessIssue]

    return cachedir


def disable_xpathlib() -> None:
    """
    Disable xpathlib and restore original pathlib behavior.
    """
    global ctx
    global logger
    logger.info('xpathlib disabled')

    ctx = None
    logger.handlers.clear()

    pathlib.Path.open = pathlib_open  # pyright: ignore[reportAttributeAccessIssue]
    pathlib.Path.touch = pathlib_touch  # pyright: ignore[reportAttributeAccessIssue]
    pathlib.Path.symlink_to = pathlib_symlink_to  # pyright: ignore[reportAttributeAccessIssue]
    pathlib.Path.rename = pathlib_rename  # pyright: ignore[reportAttributeAccessIssue]
    pathlib.Path.replace = pathlib_replace  # pyright: ignore[reportAttributeAccessIssue]
    pathlib.Path.unlink = pathlib_unlink  # pyright: ignore[reportAttributeAccessIssue]
    pathlib.Path.rmdir = pathlib_rmdir  # pyright: ignore[reportAttributeAccessIssue]

    return
