"""
Shared pytest fixtures for mcp_filesystem tests.

All temp directories are created inside tests/tmp/ within the project.
Never use system temp directories (bare tempfile.mkdtemp()) in tests — that
scatters artifacts outside the project and makes reasoning about cleanup harder.
"""

import shutil
import tempfile
from pathlib import Path

import pytest

# Root for all test temp dirs. Must stay inside the project.
# Content is gitignored; the directory itself is kept via tests/tmp/.gitkeep.
TESTS_TMP = Path(__file__).parent / "tmp"


@pytest.fixture()
def tmpdir() -> Path:
    """
    A fresh temporary directory inside tests/tmp/; removed after each test.

    Tests that need a second isolated directory (e.g. an 'outside' dir for
    security rejection tests) should call `make_tmp()` from this module directly.
    """
    TESTS_TMP.mkdir(exist_ok=True)
    d = Path(tempfile.mkdtemp(dir=TESTS_TMP))
    yield d
    shutil.rmtree(d, ignore_errors=True)


def make_tmp() -> Path:
    """
    Create and return a fresh temp dir inside tests/tmp/.

    Callers are responsible for cleanup (shutil.rmtree in a finally block).
    Use this when a test needs more than one isolated directory.
    """
    TESTS_TMP.mkdir(exist_ok=True)
    return Path(tempfile.mkdtemp(dir=TESTS_TMP))
