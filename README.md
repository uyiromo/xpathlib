# xpathlib

monkey patch for `pathlib.Path` to support following features.

- local cache with ssh backend
- transactional write (both local update, and remote push)

## usage

- enable xpathlib
  - after enabling, following `pathlib.Path` methods are patched:
    - `Path.open`
    - `Path.touch`
    - `Path.symlink_to`
    - `Path.rename`
    - `Path.replace`
    - `Path.unlink`
    - `Path.rmdir`

```python
from xpathlib import enable_xpathlib, SSHConfig

# - SSHConfig
#   - ssh <username>@<host> -p <port> -i <identity>
# - cachedir
#   - local cache directory path
# - remotedir
#   - remote directory path
# - keeps
#   - list of glob patterns to keep in local cache when syncing from remote
enable_xpathlib(
    SSHConfig(
          host="example.com",
          port=22,
          username="user",
          identity=Path("/path/to/private/key"),
    ),
    cachedir=Path("cachedir"),
    remotedir=Path("remotedir"),
    keeps=["*.log", "*.txt"],
)
```

- disable xpathlib

```python
from xpathlib import disable_xpathlib

disable_xpathlib()
```

- sync with remote

```python
from xpathlib import sync_xpathlib

sync_xpathlib()
```

- get logger
  - can get `Logger` instance for xpathlib

```python
import logging
from xpathlib import logger_xpathlib

logger: logging.Logger = logger_xpathlib()
```
