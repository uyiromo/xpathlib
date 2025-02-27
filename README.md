# xpathlib.Path

- A wrapper class of `pathlib.Path` with some additional features.
  - local caching of file contents using SSH backend.
  - transactional file operations.

## Logger

- You can get `Logger` by the `getlg()` method.
  - It returns a logger with the name `xpathlib`.

- Example1: Emit logs to stdout:

  ```python
  from logging import DEBUG, StreamHandler, Formatter
  from sys import stdout

  fmt: Formatter = Formatter('%(asctime)s [%(levelname)s] %(filename)s::%(funcName)s:%(lineno)d - %(message)s')
  hdlr: StreamHandler = StreamHandler(stream=stdout)
  hdlr.setLevel(DEBUG)
  hdlr.setFormatter(fmt)

  getlg().addHandler(hdlr)

  ```

- Example2: Emit logs to 'xpathlib.log':

  ```python
  from logging import DEBUG, FileHandler, Formatter

  fmt: Formatter = Formatter('%(asctime)s [%(levelname)s] %(filename)s::%(funcName)s:%(lineno)d - %(message)s')
  hdlr: FileHandler = FileHandler('xpathlib.log')
  hdlr.setLevel(DEBUG)
  hdlr.setFormatter(fmt)

  getlg().addHandler(hdlr)
  ```
