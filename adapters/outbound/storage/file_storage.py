import json
import logging
import os
import tempfile

from config import StorageConfig
from core.domain.entities import AuthToken
from core.domain.exceptions import DomainException
from core.ports.outbound import TokenStoragePort

logger = logging.getLogger(__name__)

FILE_MODE = 0o600
DIR_MODE = 0o700


class FileTokenStorage(TokenStoragePort):
    """Persists credentials as a 0600 JSON file, written atomically.

    The file holds a long-lived refresh token, so it is created with owner-only
    permissions and replaced via a temp file + rename so a crash mid-write
    cannot leave truncated JSON behind.
    """

    def __init__(self, config: StorageConfig):
        self._path = config.file_path
        self._dir = config.config_dir

    def load(self) -> AuthToken | None:
        if not os.path.exists(self._path):
            return None
        try:
            with open(self._path, encoding="utf-8") as handle:
                return AuthToken.from_dict(json.load(handle))
        except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.warning("Ignoring unreadable credentials at %s: %s", self._path, exc)
            return None

    def save(self, token: AuthToken) -> None:
        payload = json.dumps(token.to_dict(), indent=2)
        try:
            os.makedirs(self._dir, mode=DIR_MODE, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(dir=self._dir, prefix=".credentials-")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(payload)
                os.chmod(tmp_path, FILE_MODE)
                os.replace(tmp_path, self._path)
            except BaseException:
                _silent_unlink(tmp_path)
                raise
        except OSError as exc:
            raise DomainException(f"Could not persist credentials to {self._path}: {exc}") from exc

    def clear(self) -> None:
        try:
            os.remove(self._path)
        except FileNotFoundError:
            return
        except OSError as exc:
            raise DomainException(f"Could not remove {self._path}: {exc}") from exc


def _silent_unlink(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass
