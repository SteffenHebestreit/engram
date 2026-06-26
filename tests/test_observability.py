import logging

from app.observability import configure_logging


def test_configure_logging_sets_root_level():
    try:
        configure_logging("DEBUG")
        assert logging.getLogger().level == logging.DEBUG
        configure_logging("WARNING")
        assert logging.getLogger().level == logging.WARNING
    finally:
        configure_logging("INFO")  # restore a sane default for other tests


def test_configure_logging_tolerates_bad_level():
    # an unknown level falls back to INFO rather than raising
    configure_logging("not-a-level")
    assert logging.getLogger().level == logging.INFO
