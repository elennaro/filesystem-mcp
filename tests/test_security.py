"""
Tests for mcp_filesystem.security — path normalization, allowed-dir validation,
symlink rejection, and validatePath error messages.

All tests use tempfile.mkdtemp() — no hardcoded paths, no machine-specific data.
Tests are independent of LM Studio and any model.
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path

import pytest

from mcp_filesystem.security import (
    normalize_path,
    resolve_allowed_directories,
    validate_path,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(coro):
    """Run an async coroutine in tests."""
    return asyncio.run(coro)


@pytest.fixture()
def tmpdir():
    """A real temporary directory; cleaned up after each test."""
    d = tempfile.mkdtemp()
    yield Path(d)
    import shutil
    shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# normalize_path — all 4 formats
# ---------------------------------------------------------------------------

class TestNormalizePath:
    """normalize_path must produce an absolute Path from any AI-generated format."""

    def test_standard_forward_slashes(self, tmpdir):
        p = str(tmpdir).replace("\\", "/")
        result = normalize_path(p)
        assert result.is_absolute()
        assert result == tmpdir.resolve()

    def test_standard_backslashes(self, tmpdir):
        if os.name != "nt":
            pytest.skip("backslash paths only meaningful on Windows")
        p = str(tmpdir).replace("/", "\\")
        result = normalize_path(p)
        assert result.is_absolute()

    @pytest.mark.skipif(os.name != "nt", reason="Windows-only path format")
    def test_glm_unix_style_drive(self):
        """
        /F/work/foo.py → F:/work/foo.py  (GLM-generated format)
        normalize_path must convert this before Path() interprets it as root-relative.
        """
        result = normalize_path("/F/work/foo.py")
        assert str(result).startswith("F:") or str(result).startswith("f:")
        assert "work" in str(result)

    @pytest.mark.skipif(os.name != "nt", reason="Windows-only path format")
    def test_bare_drive_letter(self):
        """
        F/work/foo.py → F:/work/foo.py  (missing colon — GLM quirk not in TS server)
        """
        result = normalize_path("F/work/foo.py")
        assert str(result).startswith("F:") or str(result).startswith("f:")

    def test_tilde_expansion(self):
        result = normalize_path("~/testfile.txt")
        assert result.is_absolute()
        assert "~" not in str(result)

    def test_strips_surrounding_quotes(self, tmpdir):
        p = f'"{tmpdir}"'
        result = normalize_path(p)
        assert result.is_absolute()
        assert '"' not in str(result)

    def test_strips_surrounding_whitespace(self, tmpdir):
        p = f"  {tmpdir}  "
        result = normalize_path(p)
        assert result.is_absolute()

    def test_resolves_dotdot(self, tmpdir):
        p = str(tmpdir) + "/subdir/../foo.txt"
        result = normalize_path(p)
        assert ".." not in str(result)


# ---------------------------------------------------------------------------
# resolve_allowed_directories
# ---------------------------------------------------------------------------

class TestResolveAllowedDirectories:
    def test_returns_accessible_dirs(self, tmpdir):
        result = resolve_allowed_directories([str(tmpdir)])
        assert any(str(tmpdir).lower() in str(d).lower() for d in result)

    def test_skips_inaccessible(self, tmpdir, capsys):
        fake = str(tmpdir / "nonexistent_xyz_12345")
        # Should not crash; nonexistent dir gets stored (TS behavior: store normalized)
        # but since it's not a directory, it gets filtered; no exit(1) since tmpdir is also valid
        result = resolve_allowed_directories([str(tmpdir), fake])
        # tmpdir must be present
        assert any(str(tmpdir).lower() in str(d).lower() for d in result)

    def test_deduplicates_same_path(self, tmpdir):
        result = resolve_allowed_directories([str(tmpdir), str(tmpdir)])
        norms = [str(d).lower() for d in result]
        # No duplicates
        assert len(norms) == len(set(norms))

    def test_file_path_excluded(self, tmpdir):
        """A file path must not appear in the resolved allowed dirs list."""
        f = tmpdir / "file.txt"
        f.write_text("x")
        # Pass the file alongside a valid dir so sys.exit(1) is not triggered
        result = resolve_allowed_directories([str(f), str(tmpdir)])
        assert not any(str(f).lower() == str(d).lower() for d in result)


# ---------------------------------------------------------------------------
# validate_path — happy path
# ---------------------------------------------------------------------------

class TestValidatePathHappyPath:
    def test_existing_file_within_allowed(self, tmpdir):
        f = tmpdir / "hello.txt"
        f.write_text("hi")
        allowed = resolve_allowed_directories([str(tmpdir)])
        result = run(validate_path(str(f), allowed))
        assert result.is_absolute()
        assert result.name == "hello.txt"

    def test_existing_dir_within_allowed(self, tmpdir):
        sub = tmpdir / "subdir"
        sub.mkdir()
        allowed = resolve_allowed_directories([str(tmpdir)])
        result = run(validate_path(str(sub), allowed))
        assert result.is_dir()

    def test_new_file_with_existing_parent(self, tmpdir):
        new_file = tmpdir / "new.txt"
        allowed = resolve_allowed_directories([str(tmpdir)])
        result = run(validate_path(str(new_file), allowed))
        assert result.is_absolute()
        assert result.name == "new.txt"

    def test_allowed_dir_itself(self, tmpdir):
        allowed = resolve_allowed_directories([str(tmpdir)])
        result = run(validate_path(str(tmpdir), allowed))
        assert result.is_absolute()


# ---------------------------------------------------------------------------
# validate_path — path format variants (4 formats)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(os.name != "nt", reason="Windows-only path formats")
class TestValidatePathWindowsFormats:
    """All 4 path formats must be accepted for a file inside allowed dirs."""

    def _drive_and_rest(self, tmpdir: Path):
        """Split tmpdir into drive letter and rest for format construction."""
        s = str(tmpdir)
        drive = s[0].upper()  # e.g. 'F'
        rest = s[2:].replace("\\", "/")  # strip 'F:' prefix
        return drive, rest

    def test_forward_slash_format(self, tmpdir):
        f = tmpdir / "t.txt"
        f.write_text("x")
        allowed = resolve_allowed_directories([str(tmpdir)])
        p = str(f).replace("\\", "/")
        result = run(validate_path(p, allowed))
        assert result.is_absolute()

    def test_backslash_format(self, tmpdir):
        f = tmpdir / "t.txt"
        f.write_text("x")
        allowed = resolve_allowed_directories([str(tmpdir)])
        p = str(f).replace("/", "\\")
        result = run(validate_path(p, allowed))
        assert result.is_absolute()

    def test_glm_unix_style(self, tmpdir):
        """
        /F/work/... format — must be accepted without 'path outside allowed' error.
        """
        f = tmpdir / "t.txt"
        f.write_text("x")
        allowed = resolve_allowed_directories([str(tmpdir)])
        drive, rest = self._drive_and_rest(tmpdir)
        glm_path = f"/{drive}/{rest}/t.txt"
        result = run(validate_path(glm_path, allowed))
        assert result.name == "t.txt"

    def test_bare_drive_letter(self, tmpdir):
        """
        F/work/... format — must be accepted (GLM quirk, not in TS server).
        """
        f = tmpdir / "t.txt"
        f.write_text("x")
        allowed = resolve_allowed_directories([str(tmpdir)])
        drive, rest = self._drive_and_rest(tmpdir)
        bare_path = f"{drive}/{rest}/t.txt"
        result = run(validate_path(bare_path, allowed))
        assert result.name == "t.txt"


# ---------------------------------------------------------------------------
# validate_path — security rejections
# ---------------------------------------------------------------------------

class TestValidatePathRejections:
    def test_path_outside_allowed(self, tmpdir):
        other = tempfile.mkdtemp()
        try:
            allowed = resolve_allowed_directories([str(tmpdir)])
            with pytest.raises(PermissionError, match="Access denied - path outside allowed directories"):
                run(validate_path(other, allowed))
        finally:
            import shutil
            shutil.rmtree(other, ignore_errors=True)

    def test_dotdot_escape_attempt(self, tmpdir):
        """/../ traversal must be caught after normalization."""
        allowed = resolve_allowed_directories([str(tmpdir)])
        escape = str(tmpdir) + "/../../etc/passwd"
        with pytest.raises(PermissionError, match="Access denied"):
            run(validate_path(escape, allowed))

    def test_null_byte_in_path(self, tmpdir):
        allowed = resolve_allowed_directories([str(tmpdir)])
        with pytest.raises(ValueError, match="null byte"):
            run(validate_path(str(tmpdir) + "/foo\x00bar.txt", allowed))

    def test_parent_does_not_exist(self, tmpdir):
        allowed = resolve_allowed_directories([str(tmpdir)])
        deep = str(tmpdir) + "/nonexistent_parent_xyz/file.txt"
        with pytest.raises(FileNotFoundError, match="Parent directory does not exist"):
            run(validate_path(deep, allowed))

    @pytest.mark.skipif(
        os.name == "nt" and sys.version_info < (3, 8),
        reason="symlink creation may require elevated perms on older Windows",
    )
    def test_symlink_outside_allowed_rejected(self, tmpdir):
        """A symlink inside allowed that points outside must be rejected."""
        outside = tempfile.mkdtemp()
        try:
            target_file = Path(outside) / "secret.txt"
            target_file.write_text("secret")

            link = tmpdir / "evil_link.txt"
            try:
                link.symlink_to(target_file)
            except (OSError, NotImplementedError):
                pytest.skip("Cannot create symlinks on this platform/configuration")

            allowed = resolve_allowed_directories([str(tmpdir)])
            with pytest.raises(PermissionError, match="symlink target outside allowed directories"):
                run(validate_path(str(link), allowed))
        finally:
            import shutil
            shutil.rmtree(outside, ignore_errors=True)

    def test_symlink_inside_allowed_accepted(self, tmpdir):
        """A symlink inside allowed that also points inside must be accepted."""
        target = tmpdir / "real.txt"
        target.write_text("real content")
        link = tmpdir / "link.txt"
        try:
            link.symlink_to(target)
        except (OSError, NotImplementedError):
            pytest.skip("Cannot create symlinks on this platform/configuration")

        allowed = resolve_allowed_directories([str(tmpdir)])
        result = run(validate_path(str(link), allowed))
        assert result.is_absolute()


# ---------------------------------------------------------------------------
# validate_path — error message strings match TS server exactly
# ---------------------------------------------------------------------------

class TestValidatePathErrorMessages:
    """Error messages must exactly match the TS server strings for model self-correction."""

    def test_outside_allowed_message_format(self, tmpdir):
        other = tempfile.mkdtemp()
        try:
            allowed = resolve_allowed_directories([str(tmpdir)])
            with pytest.raises(PermissionError) as exc_info:
                run(validate_path(other, allowed))
            msg = str(exc_info.value)
            assert "Access denied - path outside allowed directories:" in msg
            assert " not in " in msg
        finally:
            import shutil
            shutil.rmtree(other, ignore_errors=True)

    def test_symlink_message_format(self, tmpdir):
        outside = tempfile.mkdtemp()
        try:
            target_file = Path(outside) / "s.txt"
            target_file.write_text("x")
            link = tmpdir / "link.txt"
            try:
                link.symlink_to(target_file)
            except (OSError, NotImplementedError):
                pytest.skip("Cannot create symlinks")

            allowed = resolve_allowed_directories([str(tmpdir)])
            with pytest.raises(PermissionError) as exc_info:
                run(validate_path(str(link), allowed))
            msg = str(exc_info.value)
            assert "Access denied - symlink target outside allowed directories:" in msg
        finally:
            import shutil
            shutil.rmtree(outside, ignore_errors=True)

    def test_parent_not_exist_message_format(self, tmpdir):
        allowed = resolve_allowed_directories([str(tmpdir)])
        with pytest.raises(FileNotFoundError) as exc_info:
            run(validate_path(str(tmpdir / "nope" / "file.txt"), allowed))
        assert "Parent directory does not exist:" in str(exc_info.value)
