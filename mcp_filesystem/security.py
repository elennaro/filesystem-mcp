"""
Path validation and security module.

Mirrors the security model of the Official MCP TypeScript filesystem server
(modelcontextprotocol/servers/src/filesystem), adapted for Python and Windows.

Key behaviours that match the TS server exactly:
- validatePath raises the same error message strings
- Symlinks are resolved; target must be within allowed directories
- Non-existent files: parent directory must exist and be within allowed dirs
- Allowed directories resolved to real paths at startup (both original + realpath stored)

Python-only additions (not in TS server):
- Handles "F/work/..." bare-drive-letter paths (GLM model quirk) — only when
  the leading single letter is NOT a real directory in the current working directory
- Null-byte rejection in paths
- File size limit check (enforced by callers, not here)

INTENTIONALLY NOT IMPLEMENTED (security policy):
- No recursive directory deletion (shutil.rmtree or equivalent) anywhere in
  server code. The MCP tools exposed to models do not include any delete
  operation. This matches the TS reference server which also has no delete tools.
  Callers that request deletion will receive a PermissionError or a tool-not-found
  response — never silent data loss.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Sequence


def normalize_path(path_str: str) -> Path:
    """
    Normalize a path string to an absolute pathlib.Path WITHOUT following symlinks.

    Makes the path absolute and resolves . and .. components, but deliberately
    does NOT follow symlinks or directory junctions. Symlink resolution is
    deferred to validate_path (via Path.resolve(strict=True)) so that junctions
    pointing outside allowed directories are caught at the correct step with the
    correct error message — matching the TS server's two-phase check:
      phase 1: is the path string within allowed dirs?  (normalize_path result)
      phase 2: is the symlink TARGET within allowed dirs? (Path.resolve() result)

    Handles four formats that AI models produce on Windows:
      - Standard:       C:/projects/file.py  or  C:\\projects\\file.py
      - GLM Unix-style: /C/projects/file.py    (drive letter in first component)
      - Bare drive:     C/projects/file.py     (missing colon — GLM quirk, see below)
      - Tilde:          ~/file.py              (home dir expansion)

    SECURITY NOTE — bare drive normalization (LETTER/path/...):
    This is ambiguous on Windows: "C/projects/file.py" could mean:
      (a) a relative path, i.e. cwd/C/projects/file.py, if a directory named "C" exists
      (b) a GLM-generated shorthand for the Windows path C:/projects/file.py

    We resolve the ambiguity by checking the filesystem: if a directory named
    "C" (the single letter) exists in the current working directory, we treat the
    path as relative (interpretation a). Only if it does NOT exist do we apply
    the drive-letter rewrite (interpretation b).

    The allowed-directory check in validate_path is the final security gate —
    any path that reaches the server, whether normalized or not, is rejected
    unless it falls within an explicitly configured allowed directory.

    Returns an absolute Path (symlinks not yet resolved).
    """
    p = path_str.strip().strip("\"'")

    if os.name == "nt":
        # /C/projects/... → C:/projects/...  (matches TS convertToWindowsPath for /c/ paths)
        # This is unambiguous: an absolute Unix-style path starting with /LETTER/
        # cannot be a real path in the server's CWD on Windows.
        p = re.sub(
            r"^/([A-Za-z])(/|$)",
            lambda m: m.group(1).upper() + ":/" + (m.group(2) if m.group(2) else ""),
            p,
        )
        # C/projects/... → C:/projects/...  ONLY if "C" is not a real directory in CWD.
        # If cwd/C/ exists, leave the path as-is (it's a legitimate relative path).
        bare_match = re.match(r"^([A-Za-z])/", p)
        if bare_match:
            letter = bare_match.group(1)
            if not (Path.cwd() / letter).is_dir():
                p = letter.upper() + ":/" + p[2:]

    path = Path(p).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    # os.path.normpath resolves . and .. without following symlinks or junctions
    return Path(os.path.normpath(path))


def _normalize_for_comparison(path: Path) -> str:
    """
    Return a string suitable for allowed-directory containment checks.

    On Windows: lowercased (case-insensitive FS).
    On other platforms: as-is.
    """
    s = str(path)
    if os.name == "nt":
        return s.lower()
    return s


def _is_within_allowed(path: Path, allowed: Sequence[Path]) -> bool:
    """
    Return True if `path` is equal to or a descendant of any allowed directory.

    Rejects null bytes. Both path and allowed dirs are compared after
    case-normalisation on Windows.
    """
    s = str(path)
    if "\x00" in s:
        return False

    normalized = _normalize_for_comparison(path)

    for allowed_dir in allowed:
        a = _normalize_for_comparison(allowed_dir)
        if normalized == a or normalized.startswith(a + os.sep):
            return True
    return False


def resolve_allowed_directories(raw_dirs: Sequence[str]) -> list[Path]:
    """
    Resolve CLI-supplied directory strings to real absolute Paths.

    Mirrors TS startup logic:
    - expandHome + resolve
    - try realpath; store both original-normalized and realpath if they differ
    - skip entries that are not accessible directories (warn to stderr)
    - exit(1) if all specified dirs are inaccessible

    Returns a deduplicated list of Path objects representing the allowed dirs.
    """
    candidates: list[Path] = []

    for raw in raw_dirs:
        p = normalize_path(raw)
        try:
            real = p.resolve(strict=True)  # raises if path doesn't exist
            if p != real:
                # Store both (mirrors TS: normalizedOriginal and normalizedResolved)
                candidates.append(p)
                candidates.append(real)
            else:
                candidates.append(real)
        except (OSError, RuntimeError):
            # Can't resolve — use normalized absolute (TS: store normalizedOriginal)
            candidates.append(p)

    # Filter to accessible directories
    accessible: list[Path] = []
    seen: set[str] = set()
    for p in candidates:
        key = _normalize_for_comparison(p)
        if key in seen:
            continue
        try:
            if p.is_dir():
                seen.add(key)
                accessible.append(p)
            else:
                print(f"Warning: {p} is not a directory, skipping", file=sys.stderr)
        except OSError:
            print(f"Warning: Cannot access directory {p}, skipping", file=sys.stderr)

    if not accessible and raw_dirs:
        print("Error: None of the specified directories are accessible", file=sys.stderr)
        sys.exit(1)

    return accessible


async def validate_path(requested_path: str, allowed_directories: Sequence[Path]) -> Path:
    """
    Validate that `requested_path` is within allowed directories and resolve it.

    Mirrors the TS `validatePath` function from lib.ts exactly, including
    error message strings.

    Steps:
    1. Normalize and resolve the requested path (handles all 4 AI path formats)
    2. Check it is within allowed directories (pre-symlink check, fast reject)
    3. Resolve symlinks (realpath); verify real target is also within allowed dirs
    4. For non-existent paths: verify parent is within allowed dirs and exists

    Args:
        requested_path:      Raw path string from the model.
        allowed_directories: Resolved allowed dir Paths (from resolve_allowed_directories).

    Returns:
        Resolved absolute Path (realpath for existing files; absolute for new files).

    Raises:
        PermissionError: path or symlink target outside allowed directories.
        FileNotFoundError: parent directory does not exist.
        ValueError: null byte in path.
    """
    if "\x00" in requested_path:
        raise ValueError(f"Invalid path: null byte in {requested_path!r}")

    # Step 1: normalize (handles /F/work/..., F/work/..., ~, etc.)
    absolute = normalize_path(requested_path)

    # Step 2: pre-symlink allowed-dir check (mirrors TS step before realpath)
    if not _is_within_allowed(absolute, allowed_directories):
        dirs_str = ", ".join(str(d) for d in allowed_directories)
        raise PermissionError(
            f"Access denied - path outside allowed directories: {absolute} not in {dirs_str}"
        )

    # Step 3: resolve symlinks, verify real target
    try:
        real = absolute.resolve(strict=True)  # raises OSError if not found
        if not _is_within_allowed(real, allowed_directories):
            dirs_str = ", ".join(str(d) for d in allowed_directories)
            raise PermissionError(
                f"Access denied - symlink target outside allowed directories: {real} not in {dirs_str}"
            )
        return real

    except OSError as exc:
        # File doesn't exist yet — check parent (mirrors TS ENOENT handling)
        if not absolute.exists():
            parent = absolute.parent
            try:
                real_parent = parent.resolve(strict=True)
                if not _is_within_allowed(real_parent, allowed_directories):
                    dirs_str = ", ".join(str(d) for d in allowed_directories)
                    raise PermissionError(
                        f"Access denied - parent directory outside allowed directories: "
                        f"{real_parent} not in {dirs_str}"
                    ) from exc
                return absolute  # new file — return absolute (not realpath)
            except OSError:
                raise FileNotFoundError(
                    f"Parent directory does not exist: {parent}"
                ) from exc
        raise


async def validate_path_for_creation(
    requested_path: str, allowed_directories: Sequence[Path]
) -> Path:
    """
    Validate a path for write_file / create_directory operations that auto-create
    parent directories.

    Unlike validate_path (which requires the immediate parent to exist), this
    function walks up the ancestor chain to find the nearest existing directory
    and validates THAT against allowed dirs.  The path itself need not exist, and
    neither do any of its parents — as long as the nearest existing ancestor is
    within allowed directories and is not a symlink pointing out of them.

    Python-only deviation from the TS server (which throws "Parent directory does
    not exist" in this situation; the TS server never auto-creates parents).

    Returns the absolute (non-realpath) path — callers create it themselves.
    """
    if "\x00" in requested_path:
        raise ValueError(f"Invalid path: null byte in {requested_path!r}")

    absolute = normalize_path(requested_path)

    # Pre-symlink allowed-dir check on the path string itself
    if not _is_within_allowed(absolute, allowed_directories):
        dirs_str = ", ".join(str(d) for d in allowed_directories)
        raise PermissionError(
            f"Access denied - path outside allowed directories: {absolute} not in {dirs_str}"
        )

    # Walk up to find the nearest existing ancestor
    ancestor = absolute
    while not ancestor.exists() and ancestor != ancestor.parent:
        ancestor = ancestor.parent

    if not ancestor.exists():
        dirs_str = ", ".join(str(d) for d in allowed_directories)
        raise FileNotFoundError(
            f"No accessible ancestor directory found for: {absolute}"
        )

    # Resolve ancestor (follows any symlinks) and verify it is within allowed dirs
    try:
        real_ancestor = ancestor.resolve(strict=True)
    except OSError as exc:
        raise FileNotFoundError(
            f"Cannot resolve ancestor directory: {ancestor}"
        ) from exc

    if not _is_within_allowed(real_ancestor, allowed_directories):
        dirs_str = ", ".join(str(d) for d in allowed_directories)
        raise PermissionError(
            f"Access denied - path outside allowed directories: {absolute} not in {dirs_str}"
        )

    return absolute
