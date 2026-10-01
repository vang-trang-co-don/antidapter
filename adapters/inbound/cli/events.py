"""Machine-readable event output for the CLI.

Commands that a supervisor drives (the pi extension) need to learn things
mid-flight: which URL to open, when the server is listening, why it failed.
Human text on stdout cannot carry that, so these commands emit newline-delimited
JSON on stdout instead, one object per line:

    {"event": "auth_url", "url": "https://..."}
    {"event": "serve_ready", "host": "127.0.0.1", "port": 54321}
    {"event": "error", "message": "..."}

Diagnostics still go to stderr, so a supervisor can relay them without having to
parse prose.
"""

import json
import sys
from typing import Any, TextIO

# Event names, kept in one place so the extension and the CLI cannot drift.
AUTH_URL = "auth_url"
PROGRESS = "progress"
SERVE_READY = "serve_ready"
STATUS = "status"
SUCCESS = "success"
ERROR = "error"


class EventSink:
    """Emits NDJSON events, or discards them when not requested."""

    def __init__(self, stream: TextIO | None = None) -> None:
        self._stream = stream

    @property
    def enabled(self) -> bool:
        return self._stream is not None

    def emit(self, event: str, **fields: Any) -> None:
        """Write one event line. Never raises: a supervisor must not see a
        broken pipe here take down the command it is driving."""
        if self._stream is None:
            return
        payload = {"event": event, **fields}
        try:
            self._stream.write(json.dumps(payload) + "\n")
            self._stream.flush()
        except (BrokenPipeError, ValueError):
            self._stream = None


def stdout_sink() -> EventSink:
    return EventSink(sys.stdout)


def null_sink() -> EventSink:
    return EventSink(None)
