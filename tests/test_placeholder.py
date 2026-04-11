"""Smoke tests: verify the src/ package hierarchy imports correctly."""


def test_import_scrapers() -> None:
    import src.scrapers  # noqa: F401


def test_import_processing() -> None:
    import src.processing  # noqa: F401


def test_import_models() -> None:
    import src.models  # noqa: F401


def test_import_evaluation() -> None:
    import src.evaluation  # noqa: F401


def test_import_utils() -> None:
    import src.utils  # noqa: F401
