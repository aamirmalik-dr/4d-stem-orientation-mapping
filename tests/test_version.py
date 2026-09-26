"""Guard that the package version and the installed distribution metadata agree."""

from importlib.metadata import version

import orient4d


def test_version_matches_distribution_metadata() -> None:
    assert orient4d.__version__ == version("orient4d")
