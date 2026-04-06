"""
Tests for mcp_filesystem.security — path normalization, allowed-dir validation,
junction/symlink rejection, and validatePath error messages.

All temp directories are created inside tests/tmp/ within the project via the
`tmpdir` fixture (conftest.py) or `make_tmp()`. No system temp directories,
no hardcoded machine paths, no real user directories referenced anywhere.

Tests are independent of LM Studio and any model.
"""

import asyncio
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from mcp_filesystem.security import (
    normalize_path,
    resolve_allowed_directories,
    validate_path,
)
from tests.conftest import make_tmp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(coro):
    """Run an async coroutine synchronously in tests."""
    return asyncio.run(coro)


def _create_dir_link(link: Path, target: Path) -> bool:
    """
    Create a directory-level reparse point: link → target.

    On Windows: directory junction via 'mklink /J' — requires NO elevated permissions.
    On other platforms: plain directory symlink.

    Directory junctions are followed by Path.resolve() exactly like symlinks,
    exercising the same security code path in validate_path.

    Returns True if the link was created successfully.
    """
    if os.name == "nt":
        result = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            capture_output=True,
        )
        return result.returncode == 0
    else:
        try:
            link.symlink_to(target, target_is_directory=True)
            return True
        except OSError:
            return False


# ---------------------------------------------------------------------------
# normalize_path — all 4 path formats
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
        GLM model produces /DRIVE/path/file.py instead of DRIVE:/path/file.py.
        normalize_path must rewrite this before Path() interprets the leading
        slash as root-relative (which would land on the wrong drive entirely).
        Uses drive letter Q — unlikely to be a real volume on any test machine.
        """
        if (Path.cwd() / "Q").is_dir():
            pytest.skip("Directory 'Q' exists in CWD — skip to avoid ambiguity")
        result = normalize_path("/Q/testdata/sample.py")
        assert str(result).upper().startswith("Q:")
        assert "testdata" in str(result)

    @pytest.mark.skipif(os.name != "nt", reason="Windows-only path format")
    def test_bare_drive_letter_no_real_dir(self):
        """
        GLM model also produces DRIVE/path/file.py (missing colon).
        normalize_path must rewrite this to DRIVE:/path/file.py — but ONLY
        when the leading letter is not a real subdirectory in CWD.
        Uses drive letter Q — unlikely to be a real volume on any test machine.
        """
        if (Path.cwd() / "Q").is_dir():
            pytest.skip("Directory 'Q' exists in CWD — bare-drive rewrite skipped")
        result = normalize_path("Q/testdata/sample.py")
        assert str(result).upper().startswith("Q:")

    @pytest.mark.skipif(os.name != "nt", reason="Windows-only path format")
    def test_bare_drive_letter_real_dir_not_normalized(self, tmpdir):
        """
        SECURITY: if a directory named 'Q' actually exists in CWD, then
        Q/testdata/sample.py must NOT be rewritten to Q:/testdata/sample.py.
        It must be treated as a relative path (cwd/Q/testdata/sample.py).
        """
        single_letter_dir = tmpdir / "Q"
        single_letter_dir.mkdir()
        original_cwd = Path.cwd()
        os.chdir(tmpdir)
        try:
            result = normalize_path("Q/testdata/sample.py")
            assert not str(result).upper().startswith("Q:"), (
                f"normalize_path incorrectly rewrote Q/testdata/sample.py "
                f"to {result} even though directory Q/ exists in CWD"
            )
            assert result.is_absolute()
            assert str(result).lower().startswith(str(tmpdir).lower())
        finally:
            os.chdir(original_cwd)

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

    def test_skips_inaccessible(self, tmpdir):
        fake = str(tmpdir / "nonexistent_xyz_12345")
        result = resolve_allowed_directories([str(tmpdir), fake])
        assert any(str(tmpdir).lower() in str(d).lower() for d in result)

    def test_deduplicates_same_path(self, tmpdir):
        result = resolve_allowed_directories([str(tmpdir), str(tmpdir)])
        norms = [str(d).lower() for d in result]
        assert len(norms) == len(set(norms))

    def test_file_path_excluded(self, tmpdir):
        """A file path must not appear in the resolved allowed dirs list."""
        f = tmpdir / "file.txt"
        f.write_text("x")
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
# validate_path — path format variants (all 4 formats, derived from tmpdir)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(os.name != "nt", reason="Windows-only path formats")
class TestValidatePathWindowsFormats:
    """
    All 4 path formats must work for files inside allowed dirs.

    Drive letter and path components are derived at runtime from tmpdir —
    no real user directory names are hardcoded here.
    """

    def _split(self, tmpdir: Path):
        """Return (drive_letter, rest_as_forward_slashes) from tmpdir."""
        s = str(tmpdir)
        drive = s[0].upper()
        rest = s[2:].replace("\\", "/")  # strip 'X:' prefix
        return drive, rest

    def test_forward_slash_format(self, tmpdir):
        f = tmpdir / "t.txt"
        f.write_text("x")
        allowed = resolve_allowed_directories([str(tmpdir)])
        result = run(validate_path(str(f).replace("\\", "/"), allowed))
        assert result.is_absolute()

    def test_backslash_format(self, tmpdir):
        f = tmpdir / "t.txt"
        f.write_text("x")
        allowed = resolve_allowed_directories([str(tmpdir)])
        result = run(validate_path(str(f).replace("/", "\\"), allowed))
        assert result.is_absolute()

    def test_glm_unix_style_inside_allowed(self, tmpdir):
        """
        /DRIVE/path/... format — must be accepted when it resolves into allowed dir.
        Drive letter is derived from tmpdir, not hardcoded.
        """
        f = tmpdir / "t.txt"
        f.write_text("x")
        allowed = resolve_allowed_directories([str(tmpdir)])
        drive, rest = self._split(tmpdir)
        result = run(validate_path(f"/{drive}/{rest}/t.txt", allowed))
        assert result.name == "t.txt"

    def test_bare_drive_inside_allowed(self, tmpdir):
        """
        DRIVE/path/... format (missing colon) — must be accepted when it resolves
        into allowed dir and the drive letter is not a real CWD subdir.
        Drive letter is derived from tmpdir, not hardcoded.
        """
        f = tmpdir / "t.txt"
        f.write_text("x")
        allowed = resolve_allowed_directories([str(tmpdir)])
        drive, rest = self._split(tmpdir)
        if (Path.cwd() / drive).is_dir():
            pytest.skip(f"Directory '{drive}/' exists in CWD — bare-drive rewrite skipped")
        result = run(validate_path(f"{drive}/{rest}/t.txt", allowed))
        assert result.name == "t.txt"

    def test_bare_drive_outside_allowed_rejected(self, tmpdir):
        """
        SECURITY: DRIVE/forbidden/file.txt — after bare-drive normalization rewrites
        it to DRIVE:/forbidden/file.txt, the result must be checked against allowed
        dirs and rejected if it does not fall inside them.
        """
        allowed = resolve_allowed_directories([str(tmpdir)])
        drive, _ = self._split(tmpdir)
        if (Path.cwd() / drive).is_dir():
            pytest.skip(f"Directory '{drive}/' exists in CWD — bare-drive rewrite skipped")

        with pytest.raises(PermissionError, match="Access denied - path outside allowed directories"):
            run(validate_path(f"{drive}/isolated_forbidden_xyz/secret.txt", allowed))

    def test_glm_unix_style_outside_allowed_rejected(self, tmpdir):
        """
        SECURITY: /DRIVE/forbidden/file.txt — after GLM unix-style normalization
        rewrites it to DRIVE:/forbidden/file.txt, the result must be rejected if
        it does not fall inside allowed dirs.
        """
        allowed = resolve_allowed_directories([str(tmpdir)])
        drive, _ = self._split(tmpdir)

        with pytest.raises(PermissionError, match="Access denied - path outside allowed directories"):
            run(validate_path(f"/{drive}/isolated_forbidden_xyz/secret.txt", allowed))


# ---------------------------------------------------------------------------
# validate_path — security rejections
# ---------------------------------------------------------------------------

class TestValidatePathRejections:
    def test_path_outside_allowed(self, tmpdir):
        other = make_tmp()
        try:
            allowed = resolve_allowed_directories([str(tmpdir)])
            with pytest.raises(PermissionError, match="Access denied - path outside allowed directories"):
                run(validate_path(str(other), allowed))
        finally:
            shutil.rmtree(other, ignore_errors=True)

    def test_dotdot_escape_attempt(self, tmpdir):
        """/../ traversal must be caught after . and .. normalization."""
        allowed = resolve_allowed_directories([str(tmpdir)])
        escape = str(tmpdir) + "/../../isolated_etc/passwd"
        with pytest.raises(PermissionError, match="Access denied"):
            run(validate_path(escape, allowed))

    def test_null_byte_in_path(self, tmpdir):
        allowed = resolve_allowed_directories([str(tmpdir)])
        with pytest.raises(ValueError, match="null byte"):
            run(validate_path(str(tmpdir) + "/foo\x00bar.txt", allowed))

    def test_parent_does_not_exist(self, tmpdir):
        allowed = resolve_allowed_directories([str(tmpdir)])
        with pytest.raises(FileNotFoundError, match="Parent directory does not exist"):
            run(validate_path(str(tmpdir / "nonexistent_parent_xyz" / "file.txt"), allowed))

    def test_escape_via_junction_rejected(self, tmpdir):
        """
        A junction inside allowed pointing to a directory OUTSIDE allowed must
        be rejected. validate_path follows the junction via Path.resolve() and
        checks the real destination against allowed dirs.

        Uses directory junctions on Windows (no elevated permissions needed),
        directory symlinks on other platforms.
        """
        outside = make_tmp()
        try:
            (outside / "secret.txt").write_text("secret")

            junction = tmpdir / "escape"
            if not _create_dir_link(junction, outside):
                pytest.fail(
                    "Could not create directory junction — required for security test. "
                    "On Windows, ensure 'cmd' is available."
                )

            allowed = resolve_allowed_directories([str(tmpdir)])
            with pytest.raises(PermissionError, match="symlink target outside allowed directories"):
                run(validate_path(str(junction / "secret.txt"), allowed))
        finally:
            shutil.rmtree(outside, ignore_errors=True)

    def test_junction_inside_allowed_accepted(self, tmpdir):
        """
        A junction inside allowed pointing to ANOTHER location also inside
        allowed must be accepted.
        """
        real_subdir = tmpdir / "real_subdir"
        real_subdir.mkdir()
        (real_subdir / "file.txt").write_text("content")

        junction = tmpdir / "link_subdir"
        if not _create_dir_link(junction, real_subdir):
            pytest.fail("Could not create directory junction — required for security test.")

        allowed = resolve_allowed_directories([str(tmpdir)])
        result = run(validate_path(str(junction / "file.txt"), allowed))
        assert result.is_absolute()


# ---------------------------------------------------------------------------
# validate_path — error message strings match TS server exactly
# ---------------------------------------------------------------------------

class TestValidatePathErrorMessages:
    """Error messages must exactly match the TS server strings for model self-correction."""

    def test_outside_allowed_message_format(self, tmpdir):
        other = make_tmp()
        try:
            allowed = resolve_allowed_directories([str(tmpdir)])
            with pytest.raises(PermissionError) as exc_info:
                run(validate_path(str(other), allowed))
            msg = str(exc_info.value)
            assert "Access denied - path outside allowed directories:" in msg
            assert " not in " in msg
        finally:
            shutil.rmtree(other, ignore_errors=True)

    def test_escape_message_format(self, tmpdir):
        """Error message for junction escape must match TS server string exactly."""
        outside = make_tmp()
        try:
            (outside / "s.txt").write_text("x")
            junction = tmpdir / "escape"
            if not _create_dir_link(junction, outside):
                pytest.fail("Could not create directory junction — required for this test.")

            allowed = resolve_allowed_directories([str(tmpdir)])
            with pytest.raises(PermissionError) as exc_info:
                run(validate_path(str(junction / "s.txt"), allowed))
            msg = str(exc_info.value)
            assert "Access denied - symlink target outside allowed directories:" in msg
            assert " not in " in msg
        finally:
            shutil.rmtree(outside, ignore_errors=True)

    def test_parent_not_exist_message_format(self, tmpdir):
        allowed = resolve_allowed_directories([str(tmpdir)])
        with pytest.raises(FileNotFoundError) as exc_info:
            run(validate_path(str(tmpdir / "nope" / "file.txt"), allowed))
        assert "Parent directory does not exist:" in str(exc_info.value)
