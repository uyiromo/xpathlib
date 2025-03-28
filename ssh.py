"""ssh backend"""

import os
import pathlib
import struct
from base64 import b64decode, b64encode
from logging import Logger
from pickle import loads
from socket import AF_INET, IPPROTO_UDP, SO_BROADCAST, SOCK_DGRAM, SOL_SOCKET, socket
from subprocess import CompletedProcess, run
from time import sleep
from typing import Iterable, List, Tuple

from .logger import args2str, getlg

lg: Logger = getlg()


NULLSTR: str = "\x00"

# use ControlMaster with 10m timeout
CM_ARGS: str = "-o 'ControlMaster auto' -o 'ControlPath %d/.ssh/mux-%r@%h:%p' -o 'ControlPersist 600'"


def escape(rpath: pathlib.Path, extra: bool = False) -> str:
    """escape for Linux shell"""
    unsafe: List[str] = [" ", "　", "(", ")", "[", "]", "'", "&", ";"]

    rpath_safe: str = str(rpath)
    for s in unsafe:
        rpath_safe = rpath_safe.replace(f"{s}", f"\\{s}")

    if extra:
        rpath_safe = rpath_safe.replace("`", "\\\\\\`")
        rpath_safe = rpath_safe.replace("$", "\\\\\\$")
    else:
        rpath_safe = rpath_safe.replace("`", "\\`")
        rpath_safe = rpath_safe.replace("$", "\\$")

    return rpath_safe


def runcmd(cmd: str) -> CompletedProcess:
    lg.debug(args2str(locals()))
    return run(cmd, shell=True, capture_output=True, text=True)


class SSHContext:
    def __init__(self, host: str, macaddr: str, ipaddr: str, broadcastaddr: str) -> None:
        self.host: str = host
        self.macaddr: str = macaddr
        self.ipaddr: str = ipaddr
        self.broadcastaddr: str = broadcastaddr

        # build wolpacket
        # 0xFF x6 + macaddr x16
        self.wolpacket: bytes = struct.pack("!6B", *[0xFF] * 6) + (bytes.fromhex(macaddr.replace(":", "")) * 16)

        # build nccmd
        self.nccmd: str = f"nc -v -w 1 {self.ipaddr} -z 22"

    def wol(self) -> None:
        """Wake-on-LAN"""
        lg.debug(args2str(locals()))

        lg.debug(f"send packet: {self.wolpacket!r} to '{self.broadcastaddr}'")
        with socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP) as sock:
            sock.setsockopt(SOL_SOCKET, SO_BROADCAST, 1)
            sock.sendto(self.wolpacket, (self.broadcastaddr, 7))

        # wait for wakeup (up to 300s)
        for i in range(30):
            cp: CompletedProcess = run(self.nccmd, shell=True, capture_output=True, text=True, check=False)
            lg.debug(f"send nc: '{self.nccmd}', returncode: '{cp.returncode}'")
            if cp.returncode == 0:
                break
            else:
                sleep(10)

        _ = run(self.nccmd, shell=True, capture_output=True, text=True, check=True)
        return

    def run_sshcmd(self, cmd: str, do_wol: bool) -> CompletedProcess:
        lg.debug(args2str(locals()))
        if do_wol:
            self.wol()
        else:
            pass

        sshbin: str
        if "SSHBIN" in os.environ:
            sshbin = os.environ["SSHBIN"]
        else:
            sshbin = "ssh"

        return runcmd(f'{sshbin} {self.host} {CM_ARGS} "{cmd}"')

    def run_scpfrom(self, rpath: pathlib.Path, lpath: pathlib.Path, do_wol: bool) -> CompletedProcess:
        lg.debug(args2str(locals()))
        if do_wol:
            self.wol()
        else:
            pass

        scpbin: str
        if "SCPBIN" in os.environ:
            scpbin = os.environ["SCPBIN"]
        else:
            scpbin = "scp"

        return runcmd(f"{scpbin} {CM_ARGS} {self.host}:{escape(rpath.absolute())} {escape(lpath.absolute())}")

    def run_scpto(self, lpath: pathlib.Path, rpath: pathlib.Path, do_wol: bool) -> CompletedProcess:
        lg.debug(args2str(locals()))
        if do_wol:
            self.wol()
        else:
            pass

        scpbin: str
        if "SCPBIN" in os.environ:
            scpbin = os.environ["SCPBIN"]
        else:
            scpbin = "scp"

        return runcmd(f"{scpbin} {CM_ARGS} {escape(lpath.absolute())} {self.host}:{escape(rpath.absolute())}")


def ssh_rsync(
    ctxt: SSHContext, ldirpath: pathlib.Path, rdirpath: pathlib.Path, filenames: List[str], do_wol: bool = True
) -> None:
    """rsync

    Args:
        ctxt (obj:`SSHContext`): SSHContext
        ldirpath (obj:`pathlib.Path`): local dir path
        rdirpath (obj:`pathlib.Path`): remote dir path
        filenames (obj:`List[str]`): filenames
        do_wol (obj:`bool`): True if wake-on-LAN

    Returns:
        None

    """
    lg.debug(args2str(locals()))
    assert (
        ldirpath.is_absolute() and rdirpath.is_absolute()
    ), f"ldirpath and rdirpath must be absolute ones: {ldirpath}, {rdirpath}"

    # rpath's dir may not exist
    _ = ctxt.run_sshcmd(f"mkdir -p {escape(rdirpath, extra=True)}", do_wol)

    rdirpath_safe: str = escape(rdirpath)
    rdirpath_safe = rdirpath_safe.replace("[", "\\[")
    rdirpath_safe = rdirpath_safe.replace("]", "\\]")

    cmd: str = f"rsync -avz --files-from=- --from0 -e ssh {escape(ldirpath)} {ctxt.host}:{rdirpath_safe}"

    lg.debug(f"cmd: {cmd}")
    _ = run(cmd, shell=True, capture_output=True, text=True, check=True, input=NULLSTR.join(filenames))

    return


def ssh_walk(
    ctxt: SSHContext, remotetop: pathlib.Path, do_wol: bool = True
) -> Iterable[Tuple[pathlib.Path, List[str], List[str]]]:
    """os.walk(topdown=True)

    Args:
        ctxt (obj:`SSHContext`): SSHContext
        remotetop (obj:`pathlib.Path`): remote top directory path
        do_wol (obj:`bool`): True if wake-on-LAN

    Returns:
        Iterable[Tuple[pathlib.Path, List[pathlib.Path], List[pathlib.Path]]]: (dirpath, dirnames, filenames)

    """
    lg.debug(args2str(locals()))
    assert remotetop.is_absolute(), f"remotetop must be absolute one: {remotetop}"

    dirs: List[pathlib.Path] = list()
    dirs.append(remotetop)

    while dirs:
        dirpath: pathlib.Path = dirs.pop(0)

        # dirnames
        cmd_d: str = f"cd {escape(dirpath, extra=True)} && find . -mindepth 1 -maxdepth 1 -type d -print0"
        cp_d: CompletedProcess = ctxt.run_sshcmd(cmd_d, do_wol)
        dirnames: List[str] = sorted([name for name in cp_d.stdout.strip(NULLSTR).split(sep=NULLSTR) if name])
        lg.debug(f"# of dirnames: {len(dirnames)}")

        # filenames
        cmd_f: str = f"cd {escape(dirpath, extra=True)} && find . -mindepth 1 -maxdepth 1 -type f -print0"
        cp_f: CompletedProcess = ctxt.run_sshcmd(cmd_f, do_wol)
        filenames: List[str] = sorted([name for name in cp_f.stdout.strip(NULLSTR).split(sep=NULLSTR) if name])
        lg.debug(f"# of filenames: {len(filenames)}")

        yield (dirpath, sorted(dirnames), sorted(filenames))

        # push dirs
        dirs.extend([dirpath / name for name in dirnames])

    # end of while
    return None


def ssh_rm(ctxt: SSHContext, rpath: pathlib.Path, do_wol: bool = True) -> None:
    """remove

    Args:
        ctxt (obj:`SSHContext`): SSHContext
        rpath (obj:`pathlib.Path`): remote file/directory paths
        do_wol (obj:`bool`): True if wake-on-LAN

    Returns:
        None

    """
    lg.debug(args2str(locals()))
    assert rpath.is_absolute(), f"rpath must be absolute one: {rpath}"

    cmd: str = f"rm -rf {escape(rpath, extra=True)}"
    _: CompletedProcess = ctxt.run_sshcmd(cmd, do_wol)

    return


def scp_from(ctxt: SSHContext, rpath: pathlib.Path, lpath: pathlib.Path, do_wol: bool = True) -> None:
    """scp

    Args:
        ctxt (obj:`SSHContext`): SSHContext
        rpath (obj:`pathlib.Path`): remote path
        lpath (obj:`pathlib.Path`): local path
        do_wol (obj:`bool`): True if wake-on-LAN

    Returns:
        None

    """
    lg.debug(args2str(locals()))
    assert rpath.is_absolute() and lpath.is_absolute(), f"rpath and lpath must be absolute ones: {rpath}, {lpath}"

    # lpath's dir may not exist
    lpath.parent.mkdir(parents=True, exist_ok=True)
    _ = ctxt.run_scpfrom(rpath, lpath, do_wol)

    return


def scp_to(ctxt: SSHContext, lpath: pathlib.Path, rpath: pathlib.Path, do_wol: bool = True) -> None:
    """scp

    Args:
        ctxt (obj:`SSHContext`): SSHContext
        lpath (obj:`pathlib.Path`): local path
        rpath (obj:`pathlib.Path`): remote path
        do_wol (obj:`bool`): True if wake-on-LAN

    Returns:
        None

    """
    lg.debug(args2str(locals()))
    assert lpath.is_absolute() and rpath.is_absolute(), f"lpath and rpath must be absolute ones: {lpath}, {rpath}"

    # rpath's dir may not exist
    _ = ctxt.run_sshcmd(f"mkdir -p {escape(rpath.parent, extra=True)}", do_wol)
    _ = ctxt.run_scpto(lpath, rpath, do_wol)

    return


def ssh_mv(ctxt: SSHContext, rpath_src: pathlib.Path, rpath_dst: pathlib.Path, do_wol: bool = True) -> None:
    """mv

    Args:
        ctxt (obj:`SSHContext`): SSHContext
        rpath_src (obj:`pathlib.Path`): remote src path (must be absolute)
        rpath_dst (obj:`pathlib.Path`): remote dst path (must be absolute)
        do_wol (obj:`bool`): True if wake-on-LAN

    Returns:
        None

    """
    lg.debug(args2str(locals()))
    assert (
        rpath_src.is_absolute() and rpath_dst.is_absolute()
    ), f"rpath_src and rpath_dst must be absolute ones: {rpath_src}, {rpath_dst}"

    # rpath_dst's dir may not exist
    _ = ctxt.run_sshcmd(f"mkdir -p {escape(rpath_dst.parent, extra=True)}", do_wol)
    _ = ctxt.run_sshcmd(f"mv {escape(rpath_src)} {escape(rpath_dst)}", do_wol)

    return


def ssh_stat(ctxt: SSHContext, rpath: pathlib.Path, do_wol: bool = True) -> os.stat_result:
    """stat

    Args:
        ctxt (obj:`SSHContext`): SSHContext
        rpath (obj:`pathlib.Path`): remote path (must be absolute)
        do_wol (obj:`bool`): True if wake-on-LAN

    Returns:
        os.stat_result: stat result

    """
    lg.debug(args2str(locals()))
    assert rpath.is_absolute(), f"rpath must be absolute one: {rpath}"

    script: str = f"""
from pickle import dumps, loads
from pathlib import Path
import os
from base64 import b64decode, b64encode
r: os.stat_result = Path("{rpath}").stat()
r_str: str = b64encode(dumps(r)).decode("ascii")
print(r_str)
"""
    script_s: str = b64encode(script.encode()).decode("ascii")
    stat_s: str = ctxt.run_sshcmd(f"python3 -c '$(echo '{script_s}' | base64 -d -)'", do_wol).stdout

    return loads(b64decode(stat_s))
