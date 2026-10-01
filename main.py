#!/usr/bin/env python3
import logging
import sys

from config import AppConfig, load_dotenv
from container import Container
from core.domain.exceptions import DomainException


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    try:
        config = AppConfig.from_env()
    except DomainException as exc:
        print(f"configuration error: {exc.message}", file=sys.stderr)
        print(
            "\nSet the required variables, for example in a .env file:\n"
            "  ANTIDAPTER_CLIENT_ID=...\n"
            "  ANTIDAPTER_CLIENT_SECRET=...",
            file=sys.stderr,
        )
        return 2

    verbose = "--verbose" in (sys.argv[1:] if argv is None else argv)
    level_name = "DEBUG" if verbose else config.log_level
    logging.basicConfig(
        level=getattr(logging, level_name.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    return Container(config).cli_runner.run(sys.argv[1:] if argv is None else argv)


if __name__ == "__main__":
    sys.exit(main())
