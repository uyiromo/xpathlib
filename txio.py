from __future__ import annotations

import io
import collections
import os
import pathlib
from hashlib import sha512
import typing


class TxIO(io.IOBase):

    def __init__(
        self,
        file: pathlib.Path,
        mode: str,
        buffering: int,
        encoding: typing.Optional[str],
        errors: typing.Optional[str],
        newline: typing.Optional[str],
    ):
        self._orgfile: pathlib.Path = file
        self._tmpfile: typing.Optional[pathlib.Path] = None
        self._f: typing.IO

        # Start transaction if appending
        if "a" in mode:
            tmpname: str = sha512(str(self._orgfile.absolute()).encode()).hexdigest()

            cachedir: pathlib.Path = pathlib.Path(os.environ["HOME"]) / ".cache" / "txio"
            cachedir.mkdir(parents=True, exist_ok=True)
            self._tmpfile = cachedir / tmpname

            # copy original file to tmpfile
            if self._orgfile.exists():
                self._tmpfile.write_bytes(self._orgfile.read_bytes())
            else:
                self._tmpfile.touch()

            self._f = self._tmpfile.open(mode, buffering, encoding, errors, newline)
        else:
            self._f = self._orgfile.open(mode, buffering, encoding, errors, newline)

    #
    # stubs
    #
    def fileno(self) -> int:
        return self._f.fileno()

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        return self._f.seek(offset, whence)

    def truncate(self, size: typing.Optional[int] = None) -> int:
        return self._f.truncate(size)

    #
    # Mixin methods
    #
    def close(self) -> None:
        """End transaction"""
        if self._tmpfile:
            self._f.close()
            self._tmpfile.replace(self._orgfile)
        else:
            self._f.close()

        return

    @property
    def closed(self) -> bool:
        return self._f.closed

    def __enter__(self) -> TxIO:
        # do nothing because file has already opened in __init__
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

        return exc_type is None

    def flush(self) -> None:
        self._f.flush()

    def isatty(self) -> bool:
        return self._f.isatty()

    def __iter__(self) -> typing.Iterator[typing.Any]:
        return iter(self._f)

    def __next__(self) -> typing.Any:
        return next(self._f)

    def readable(self) -> bool:
        return self._f.readable()

    def readline(self, size: int | None = -1) -> typing.AnyStr:  # type: ignore[type-var]
        return self._f.readline(size if size is not None else -1)

    def readlines(self, hint: int = -1) -> list[str]:  # type: ignore[misc, override]
        return self._f.readlines(hint)

    def seekable(self) -> bool:
        return self._f.seekable()

    def tell(self) -> int:
        return self._f.tell()

    def writable(self) -> bool:
        return self._f.writable()

    def writelines(self, lines: typing.Iterable[typing.Any]) -> None:
        self._f.writelines(lines)


class TxRawIO(io.RawIOBase, TxIO):

    def __init__(
        self,
        file: pathlib.Path,
        mode: str,
        buffering: int,
        encoding: typing.Optional[str],
        errors: typing.Optional[str],
        newline: typing.Optional[str],
    ):
        super().__init__(file, mode, buffering, encoding, errors, newline)
        self._f: typing.BinaryIO  # type: ignore[assignment]

    #
    # stubs
    #

    def write(self, b: collections.abc.Buffer) -> int:
        return self._f.write(b)

    #
    # Mixin methods
    #
    def read(self, size: int | None = -1) -> bytes:
        return self._f.read(size if size is not None else -1)

    def readall(self) -> bytes:
        return self._f.read()


class TxTextIO(io.TextIOBase, TxIO):

    def __init__(
        self,
        file: pathlib.Path,
        mode: str,
        buffering: int,
        encoding: typing.Optional[str],
        errors: typing.Optional[str],
        newline: typing.Optional[str],
    ):
        super().__init__(file, mode, buffering, encoding, errors, newline)
        self._f: typing.TextIO  # type: ignore[assignment]

    # @property
    # def encoding(self) -> str:
    #    return self._f.encoding

    # @property
    # def errors(self) -> typing.Optional[str]:
    #    return self._f.errors

    # @property
    # def newlines(self) -> typing.Any:
    #    return self._f.newlines

    # @property
    # def buffer(self) -> io.BufferedIOBase:
    #    return self._f.buffer

    # def detach(self) -> typing.BinaryIO:  # type: ignore[attr-defined]
    #    return self._f.detach()

    def read(self, size: int | None = -1) -> str:
        return self._f.read(size if size else -1)

    def readline(self, size: int | None = -1) -> str:  # type: ignore[override, arg-type]
        return self._f.readline(size if size else -1)

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        return self._f.seek(offset, whence)

    def tell(self) -> int:
        return self._f.tell()

    def write(self, s: str) -> int:
        return self._f.write(s)
