"""The common base of the AstroMatic tool wrappers: locating the binary, running it, reporting failures."""
from __future__ import annotations

import re
import shutil
import logging
import subprocess

logger = logging.getLogger('sfft.astromatic')


class ToolNotFoundError(RuntimeError):
    pass


class ToolRunError(RuntimeError):
    def __init__(self, message, argv=None, returncode=None, stderr=''):
        self.argv = argv
        self.returncode = returncode
        self.stderr = stderr
        tail = stderr.strip().splitlines()[-8:] if stderr else []
        detail = ('\n  command: %s\n  returncode: %s\n  stderr tail:\n    %s'
                  % (' '.join(argv or []), returncode, '\n    '.join(tail)))
        super().__init__(message + detail)


def find_binary(candidates):
    """First existing binary among candidate names, else ToolNotFoundError."""
    for name in candidates:
        path = shutil.which(name)
        if path is not None:
            return path
    raise ToolNotFoundError(
        'none of %s found on PATH -- install the AstroMatic tool or pass '
        'binary= explicitly' % (candidates,))


class AstromaticTool:
    CANDIDATES: tuple = ()
    NAME = 'astromatic'

    def __init__(self, binary=None, timeout=900):
        self.binary = binary if binary is not None else find_binary(self.CANDIDATES)
        self.timeout = timeout
        self.version = self._query_version()
        logger.info('%s binary: %s (version %s)', self.NAME, self.binary,
                    self.version)

    def _query_version(self):
        try:
            p = subprocess.run([self.binary], capture_output=True, text=True,
                               timeout=30)
            m = re.search(r'[Vv]ersion (\d[\w.]*)', (p.stdout or '') + (p.stderr or ''))
            return m.group(1) if m else 'unknown'
        except Exception:
            return 'unknown'

    def _run(self, argv, cwd=None):
        logger.info('%s: %s', self.NAME, ' '.join(argv))
        try:
            p = subprocess.run(argv, capture_output=True, text=True,
                               timeout=self.timeout, cwd=cwd)
        except subprocess.TimeoutExpired as e:
            raise ToolRunError('%s timed out after %ds' % (self.NAME, self.timeout),
                               argv=argv, stderr=str(e)) from e
        if p.stdout:
            logger.debug('%s stdout:\n%s', self.NAME, p.stdout)
        if p.stderr:
            logger.debug('%s stderr:\n%s', self.NAME, p.stderr)
        if p.returncode != 0:
            raise ToolRunError('%s failed' % self.NAME, argv=argv,
                               returncode=p.returncode, stderr=p.stderr or p.stdout)
        return p
