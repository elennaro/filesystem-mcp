"""
Performance regression checks for mcp_filesystem.operations.

These are NOT benchmarks — they set generous wall-clock budgets that will pass
on any reasonable hardware but catch serious regressions such as:
  - reading a file twice
  - quadratic edit-matching
  - re-walking a directory tree on every call

Each test creates its own data inside tests/tmp/ and tears it down via the
tmpdir fixture (never system temp or real user directories).

Budgets (in seconds) are intentionally loose — 10–50× slower than typical
observed times. Tighten only if a specific regression needs catching.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import pytest

from mcp_filesystem.operations import (
    directory_tree,
    edit_file,
    grep_files,
    read_media_file,
    read_text_file,
    search_files,
    write_file,
)
from mcp_filesystem.security import resolve_allowed_directories
from tests.conftest import make_tmp


def run(coro):
    return asyncio.run(coro)


def _elapsed(coro) -> float:
    """Run a coroutine and return wall-clock seconds."""
    t0 = time.perf_counter()
    run(coro)
    return time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Budgets (seconds) — edit here to tighten/loosen per machine
# ---------------------------------------------------------------------------
BUDGET_READ_1MB   = 2.0   # read_text_file on a 1 MB file
BUDGET_WRITE_1MB  = 2.0   # write_file with 1 MB content
BUDGET_EDIT_LARGE = 2.0   # edit_file: 10 edits on a 1 000-line file
BUDGET_TREE_200   = 3.0   # directory_tree on a 200-entry flat tree
BUDGET_SEARCH_200 = 3.0   # search_files on a 200-entry flat tree
BUDGET_MEDIA_1MB  = 2.0   # read_media_file on a 1 MB binary file
BUDGET_GREP_200   = 5.0   # grep_files on 200 files × 50 lines with context_lines=2


# ---------------------------------------------------------------------------
# read_text_file
# ---------------------------------------------------------------------------

class TestReadTextFilePerf:
    def test_read_1mb_within_budget(self, tmpdir):
        f = tmpdir / "big.txt"
        f.write_bytes(b"x" * 1024 * 1024)
        allowed = resolve_allowed_directories([str(tmpdir)])
        elapsed = _elapsed(read_text_file(str(f), allowed))
        assert elapsed < BUDGET_READ_1MB, (
            f"read_text_file 1 MB took {elapsed:.3f}s — budget {BUDGET_READ_1MB}s"
        )

    def test_read_head_1mb_within_budget(self, tmpdir):
        f = tmpdir / "big.txt"
        f.write_bytes(b"line\n" * 200_000)  # ~1 MB
        allowed = resolve_allowed_directories([str(tmpdir)])
        elapsed = _elapsed(read_text_file(str(f), allowed, head=100))
        assert elapsed < BUDGET_READ_1MB, (
            f"read_text_file head=100 on 1 MB took {elapsed:.3f}s — budget {BUDGET_READ_1MB}s"
        )


# ---------------------------------------------------------------------------
# write_file
# ---------------------------------------------------------------------------

class TestWriteFilePerf:
    def test_write_1mb_within_budget(self, tmpdir):
        f = tmpdir / "out.txt"
        content = "a" * 1024 * 1024
        allowed = resolve_allowed_directories([str(tmpdir)])
        elapsed = _elapsed(write_file(str(f), content, allowed))
        assert elapsed < BUDGET_WRITE_1MB, (
            f"write_file 1 MB took {elapsed:.3f}s — budget {BUDGET_WRITE_1MB}s"
        )

    def test_overwrite_1mb_within_budget(self, tmpdir):
        f = tmpdir / "out.txt"
        f.write_text("old content")
        content = "b" * 1024 * 1024
        allowed = resolve_allowed_directories([str(tmpdir)])
        elapsed = _elapsed(write_file(str(f), content, allowed))
        assert elapsed < BUDGET_WRITE_1MB, (
            f"write_file overwrite 1 MB took {elapsed:.3f}s — budget {BUDGET_WRITE_1MB}s"
        )


# ---------------------------------------------------------------------------
# edit_file
# ---------------------------------------------------------------------------

class TestEditFilePerf:
    def test_10_edits_on_1000_line_file(self, tmpdir):
        lines = [f"line_{i:04d}: some content here" for i in range(1000)]
        content = "\n".join(lines) + "\n"
        f = tmpdir / "target.txt"
        f.write_text(content)
        allowed = resolve_allowed_directories([str(tmpdir)])
        edits = [
            {"oldText": f"line_{i*100:04d}: some content here",
             "newText": f"line_{i*100:04d}: EDITED"}
            for i in range(10)
        ]
        elapsed = _elapsed(edit_file(str(f), edits, allowed))
        assert elapsed < BUDGET_EDIT_LARGE, (
            f"edit_file 10 edits on 1000-line file took {elapsed:.3f}s "
            f"— budget {BUDGET_EDIT_LARGE}s"
        )


# ---------------------------------------------------------------------------
# directory_tree
# ---------------------------------------------------------------------------

class TestDirectoryTreePerf:
    def test_200_entry_flat_tree(self, tmpdir):
        for i in range(200):
            (tmpdir / f"file_{i:03d}.txt").write_text("x")
        allowed = resolve_allowed_directories([str(tmpdir)])
        elapsed = _elapsed(directory_tree(str(tmpdir), allowed))
        assert elapsed < BUDGET_TREE_200, (
            f"directory_tree 200 files took {elapsed:.3f}s — budget {BUDGET_TREE_200}s"
        )

    def test_200_entry_nested_tree(self, tmpdir):
        for i in range(20):
            sub = tmpdir / f"dir_{i:02d}"
            sub.mkdir()
            for j in range(10):
                (sub / f"file_{j:02d}.txt").write_text("x")
        allowed = resolve_allowed_directories([str(tmpdir)])
        elapsed = _elapsed(directory_tree(str(tmpdir), allowed))
        assert elapsed < BUDGET_TREE_200, (
            f"directory_tree 20×10 nested took {elapsed:.3f}s — budget {BUDGET_TREE_200}s"
        )


# ---------------------------------------------------------------------------
# search_files
# ---------------------------------------------------------------------------

class TestSearchFilesPerf:
    def test_search_200_files(self, tmpdir):
        for i in range(200):
            (tmpdir / f"file_{i:03d}.py").write_text("x")
        allowed = resolve_allowed_directories([str(tmpdir)])
        elapsed = _elapsed(search_files(str(tmpdir), "*.py", allowed))
        assert elapsed < BUDGET_SEARCH_200, (
            f"search_files 200 files took {elapsed:.3f}s — budget {BUDGET_SEARCH_200}s"
        )

    def test_search_mixed_200_files(self, tmpdir):
        for i in range(100):
            (tmpdir / f"src_{i:03d}.py").write_text("x")
        for i in range(100):
            (tmpdir / f"data_{i:03d}.csv").write_text("x")
        allowed = resolve_allowed_directories([str(tmpdir)])
        elapsed = _elapsed(search_files(str(tmpdir), "**/*.py", allowed))
        assert elapsed < BUDGET_SEARCH_200, (
            f"search_files mixed 200 files took {elapsed:.3f}s — budget {BUDGET_SEARCH_200}s"
        )


# ---------------------------------------------------------------------------
# read_media_file
# ---------------------------------------------------------------------------

class TestReadMediaFilePerf:
    def test_read_1mb_image_within_budget(self, tmpdir):
        f = tmpdir / "large.png"
        f.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * (1024 * 1024 - 8))
        allowed = resolve_allowed_directories([str(tmpdir)])
        elapsed = _elapsed(read_media_file(str(f), allowed))
        assert elapsed < BUDGET_MEDIA_1MB, (
            f"read_media_file 1 MB PNG took {elapsed:.3f}s — budget {BUDGET_MEDIA_1MB}s"
        )


# ---------------------------------------------------------------------------
# grep_files
# ---------------------------------------------------------------------------

class TestGrepFilesPerf:
    def test_grep_200_files_with_context(self, tmpdir):
        # 200 files × 50 lines, one match per file, context_lines=2
        for i in range(200):
            lines = [f"line {j:04d} content" for j in range(50)]
            lines[25] = f"TARGET_MATCH_{i:03d}"
            (tmpdir / f"file_{i:03d}.py").write_text("\n".join(lines))
        allowed = resolve_allowed_directories([str(tmpdir)])
        elapsed = _elapsed(
            grep_files(str(tmpdir), "TARGET_MATCH", allowed, context_lines=2)
        )
        assert elapsed < BUDGET_GREP_200, (
            f"grep_files 200 files with context took {elapsed:.3f}s "
            f"— budget {BUDGET_GREP_200}s"
        )
