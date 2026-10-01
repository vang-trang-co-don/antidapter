#!/usr/bin/env python3
import logging
import sys

from adapters.inbound.cli.events import ERROR, stdout_sink
from config import AppConfig, load_dotenv
from container import Container
from core.domain.exceptions import DomainException

_MISSING_CREDENTIALS_HELP = (
    "\nSet the required variables, for example in a .env file:\n"
    "  ANTIDAPTER_CLIENT_ID=...\n"
    "  ANTIDAPTER_CLIENT_SECRET=...\n"
)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    # Read before the container is built: the event sink decides whether stdout
    # carries NDJSON or human text, which changes how everything downstream
    # writes.
    json_events = "--json-events" in args
    load_dotenv()

    try:
        config = AppConfig.from_env()
    except DomainException as exc:
        message = f"configuration error: {exc.message}"
        if json_events:
            stdout_sink().emit(ERROR, message=exc.message)
        print(message, file=sys.stderr)
        print(_MISSING_CREDENTIALS_HELP, file=sys.stderr)
        return 2

    verbose = "--verbose" in args
    level_name = "DEBUG" if verbose else config.log_level
    logging.basicConfig(
        level=getattr(logging, level_name.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        # stderr always: stdout may be carrying machine-readable events.
        stream=sys.stderr,
    )

    container = Container(config, events=stdout_sink() if json_events else None)
    return container.cli_runner.run(args)


if __name__ == "__main__":
    sys.exit(main())
