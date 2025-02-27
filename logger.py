"""Common utils"""

from logging import DEBUG, Logger, getLogger
from typing import Any, Dict


def args2str(args: Dict[str, Any]) -> str:
    args_str: str = ", ".join(f"{k}:'{v}'" for k, v in args.items())
    return f"Args: {args_str}"


def getlg() -> Logger:
    lg: Logger = getLogger("xpathlib")
    lg.setLevel(DEBUG)

    return lg
