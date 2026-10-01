import logging
from collections.abc import Sequence

from core.domain.entities import AuthToken
from core.ports.outbound import TokenSourcePort, TokenStoragePort

logger = logging.getLogger(__name__)


class ChainedTokenStorage(TokenStoragePort):
    """Reads from an ordered chain of sources, writes to a single sink.

    A token discovered in a read-only source is promoted to the sink so the
    rest of the application only ever depends on the sink.
    """

    def __init__(
        self,
        sink: TokenStoragePort,
        sources: Sequence[TokenSourcePort | TokenStoragePort] = (),
    ):
        self._sink = sink
        # The sink is itself the first place to look, so callers only need to
        # know about this object rather than orchestrating a read order.
        self._sources: tuple[TokenSourcePort | TokenStoragePort, ...] = (sink, *sources)

    def load(self) -> AuthToken | None:
        for position, source in enumerate(self._sources):
            try:
                token = source.load()
            except Exception as exc:
                logger.warning("Token source %s failed: %s", type(source).__name__, exc)
                continue
            if token is None:
                continue
            if position == 0:
                return token
            logger.info("Discovered credentials via %s", type(source).__name__)
            self._promote(token)
            return token
        return None

    def save(self, token: AuthToken) -> None:
        self._sink.save(token)

    def clear(self) -> None:
        errors = []
        for target in dict.fromkeys(self._sources):
            if not isinstance(target, TokenStoragePort):
                continue
            try:
                target.clear()
            except Exception as exc:
                errors.append(f"{type(target).__name__}: {exc}")
        if errors:
            logger.warning("Some token stores could not be cleared: %s", "; ".join(errors))

    def _promote(self, token: AuthToken) -> None:
        try:
            self._sink.save(token)
        except Exception as exc:
            logger.warning("Could not promote discovered token to primary storage: %s", exc)
