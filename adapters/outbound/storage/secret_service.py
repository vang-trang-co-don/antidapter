import json
import logging
from typing import Any

from config import StorageConfig
from core.domain.entities import AuthToken
from core.ports.outbound import TokenSourcePort

logger = logging.getLogger(__name__)


class SecretServiceCredentialSource(TokenSourcePort):
    """Read-only discovery of credentials held in the OS keyring (libsecret).

    This deliberately implements the read-only TokenSourcePort rather than
    TokenStoragePort: the entries belong to another application (the `agy`
    CLI) and rewriting or deleting them would corrupt that application's
    session. Anything found here is promoted to Antidapter's own storage by the
    chaining adapter, which owns writes.

    The `gi` import is local and optional so the gateway still runs on hosts
    without PyGObject.
    """

    def __init__(self, config: StorageConfig):
        self._service_name = config.keyring_service
        self._username = config.keyring_username

    def load(self) -> AuthToken | None:
        try:
            import gi

            gi.require_version("Secret", "1")
            from gi.repository import Secret
        except (ImportError, ValueError) as exc:
            logger.debug("Secret Service unavailable: %s", exc)
            return None

        try:
            service = Secret.Service.get_sync(Secret.ServiceFlags.LOAD_COLLECTIONS, None)
            for collection in service.get_collections():
                token = self._scan_collection(collection)
                if token is not None:
                    return token
        except Exception as exc:
            logger.debug("Secret Service query failed: %s", exc)
        return None

    def _scan_collection(self, collection: Any) -> AuthToken | None:
        import gi

        gi.require_version("Secret", "1")

        for item in collection.get_items():
            attributes = item.get_attributes()
            if attributes.get("service") != self._service_name:
                continue
            if attributes.get("username") != self._username:
                continue
            item.load_secret_sync(None)
            secret = item.get_secret()
            if secret is None:
                continue
            try:
                raw: Any = json.loads(secret.get().decode("utf-8"))
            except (ValueError, UnicodeDecodeError) as exc:
                logger.debug("Ignoring malformed keyring payload: %s", exc)
                continue
            token = self._to_token(raw)
            if token is not None:
                return token
        return None

    def _to_token(self, raw: Any) -> AuthToken | None:
        # The `agy` CLI nests the token and calls the field `expiry`.
        if not isinstance(raw, dict):
            return None
        payload: Any = raw.get("token")
        if not isinstance(payload, dict):
            payload = raw
        try:
            if "expiry" in payload:
                payload = {**payload, "expiry_time": payload["expiry"]}
            return AuthToken.from_dict(payload)
        except (KeyError, ValueError) as exc:
            logger.debug("Ignoring unusable keyring token: %s", exc)
            return None
