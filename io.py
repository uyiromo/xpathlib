from __future__ import annotations

import io
import os
import pathlib
from hashlib import sha512
from typing import IO, List, Optional, Tuple


class TxIO(io.IOBase):

    def __init__(
        self,
        file: pathlib.Path,
        mode: str,
        buffering: int,
        encoding: Optional[str],
        errors: Optional[str],
        newline: Optional[str],
    ):
        self._orgfile: pathlib.Path = file
        self._tmpfile: Optional[pathlib.Path] = None
        self._f: io.IO

        # Start transaction if writeable
        if ("w" in mode) or ("a" in mode) or ("x" in mode):
            tmpname: str = sha512(str(self._orgfile.absolute()).encode()).hexdigest()
            self._tmpfile = pathlib.Path("tmp") / tmpname

            # copy original file to tmpfile
            if self._orgfile.exists():
                self._tmpfile.write_bytes(self._orgfile.read_bytes())
            else:
                self._tmpfile.touch()

            self._f: IO = self._tmpfile.open(mode, buffering, encoding, errors, newline)
        else:
            self._f: IO = self._orgfile.open(mode, buffering, encoding, errors, newline)

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

    def fileno(self) -> int:
        return self._f.fileno()

    def flush(self) -> None:
        self._f.flush()

    def isatty(self) -> bool:
        return self._f.isatty()

    def readable(self) -> bool:
        return self._f.readable()

    def readline(self, size: int = -1) -> bytes:
        return self._f.readline(size)

    def readlines(self, hint: int = -1) -> List[bytes]:
        return self._f.readlines(hint)

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        return self._f.seek(offset, whence)

    def seekable(self) -> bool:
        return self._f.seekable()

    def tell(self) -> int:
        return self._f.tell()

    def truncate(self, size: Optional[int] = None) -> int:
        return self._f.truncate(size)

    def writable(self) -> bool:
        return self._f.writable()

    def writelines(self, lines: List[bytes]) -> None:
        return self._f.writelines(lines)


class TxRawIO(TxIO):

    def read(self, size: int = -1) -> bytes:
        return self._f.read(size)

    def readall(self) -> bytes:
        return self._f.readall()

    def readinto(self, b: bytearray) -> int:
        return self._f.readinto(b)

    def write(self, b: bytes) -> int:
        return self._f.write(b)


class TxTextIO(TxIO):

    @property
    def encoding(self) -> str:
        return self._f.encoding

    @property
    def errors(self) -> str:
        return self._f.errors

    @property
    def newlines(self) -> Optional[str | Tuple[str]]:
        return self._f.newlines

    # @property
    # def buffer(self) -> io.BufferedIOBase:
    #    return self._f.buffer

    def detach(self) -> io.RawIOBase:
        return self._f.detach()

    def read(self, size: int = -1) -> str:
        return self._f.read(size)

    def readline(self, size: int = -1) -> str:
        return self._f.readline(size)

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        return self._f.seek(offset, whence)

    def tell(self) -> int:
        return self._f.tell()

    def write(self, s: str) -> int:
        return self._f.write(s)
