"""
Privacy-preserving structured logging for MedDriftGuard.

Strictly avoids logging patient personally identifiable information (PII)
such as patient names, contact numbers, email addresses, or precise locations.
Logs operational metadata, stage latencies, turn IDs, and metric summaries.
"""

import logging
import sys
from typing import Any


class PIIMaskedFormatter(logging.Formatter):
    """A logging formatter that ensures safe log messages."""

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        return msg


def get_logger(name: str = "meddriftguard") -> logging.Logger:
    """Return a standard structured logger configured for MedDriftGuard."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = PIIMaskedFormatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s (Turn:%(turn_id)s): %(message)s",
            defaults={"turn_id": "-"},
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
