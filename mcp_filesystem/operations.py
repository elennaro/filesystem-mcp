"""
File operations for the MCP filesystem server.

Implements all filesystem tools except list_allowed_directories (which lives in
server.py because it needs the runtime allowed_directories list).

All public functions accept raw path strings and the allowed_directories list.
Path validation via security.validate_path is always the first step — no I/O
before the path is confirmed safe.

Design notes:
- Atomic writes: tempfile.mkstemp + os.replace (Windows-safe; avoids EPERM
  that occurs with os.rename when source is on a different volume)
- File size limit: 50 MB (read + write) — protects against OOM; TS has no limit
- GLM coercion: edit_file silently accepts edits as a JSON string (model bug)
- Auto-create parent dirs on write: mkdir(parents=True) before atomic write;
  TS server throws "Parent directory does not exist" instead
- CRLF normalization: edit_file normalizes to LF for matching; writes LF
- No shutil.rmtree anywhere in this module (security policy)
"""

from __future__ import annotations

import difflib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from mcp_filesystem.security import validate_path, validate_path_for_creation

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_size(path: Path) -> None:
    """Raise ValueError if file exceeds the size limit."""
    size = path.stat().st_size
    if size > MAX_FILE_SIZE:
        limit_mb = MAX_FILE_SIZE // (1024 * 1024)
        raise ValueError(f"File too large: {size} bytes (limit {limit_mb} MB)")


def _atomic_write(path: Path, content: str) -> None:
    """Write content to path atomically: temp file in same dir + os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(content)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _format_size(size_bytes: int) -> str:
    """Format byte count as human-readable string (matches TS formatSize)."""
    if size_bytes == 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size_bytes)
    idx = 0
    while value >= 1024 and idx < len(units) - 1:
        value /= 1024
        idx += 1
    return f"{value:.2f} {units[idx]}"


def _glob_to_regex(pattern: str) -> re.Pattern:
    """
    Convert a glob pattern (with **, *, ?) to a compiled regex.

    Rules:
    - ** matches any number of path segments (including zero)
    - *  matches any characters except /
    - ?  matches any single character except /
    - All other characters are escaped
    """
    pat = pattern.replace("\\", "/")
    parts: list[str] = []
    i = 0
    while i < len(pat):
        c = pat[i]
        if c == "*" and i + 1 < len(pat) and pat[i + 1] == "*":
            i += 2
            if i < len(pat) and pat[i] == "/":
                i += 1
                # **/ matches zero or more directory segments
                parts.append("(?:.*/)?")
            else:
                # ** at end (or before non-/) matches everything
                parts.append(".*")
        elif c == "*":
            parts.append("[^/]*")
            i += 1
        elif c == "?":
            parts.append("[^/]")
            i += 1
        else:
            parts.append(re.escape(c))
            i += 1
    return re.compile("^" + "".join(parts) + "$")


def _glob_match(rel_str: str, pattern: str) -> bool:
    """Return True if rel_str (forward-slash path) matches glob pattern."""
    return bool(_glob_to_regex(pattern).match(rel_str.replace("\\", "/")))


def _matches_exclude(rel_str: str, patterns: list[str]) -> bool:
    """
    Return True if rel_str matches any exclusion pattern.

    Mirrors TS minimatch behaviour:
    - Pattern with *: test as-is against relative path.
    - Pattern without * (bare name like 'node_modules'): also test
      **/{pattern} and **/{pattern}/** to exclude anywhere in the tree.
    """
    rel = rel_str.replace("\\", "/")
    for pat in patterns:
        if _glob_match(rel, pat):
            return True
        if "*" not in pat:
            if _glob_match(rel, f"**/{pat}"):
                return True
            if _glob_match(rel, f"**/{pat}/**"):
                return True
    return False


def _make_diff(original: str, modified: str, path: Path) -> str:
    """Produce a git-style unified diff in a dynamic-backtick fence."""
    diff_lines = list(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            modified.splitlines(keepends=True),
            fromfile=f"{path}\toriginal",
            tofile=f"{path}\tmodified",
        )
    )
    diff_str = "".join(diff_lines)
    max_seq = max(
        (len(m.group()) for m in re.finditer(r"`+", diff_str)),
        default=2,
    )
    fence = "`" * max(3, max_seq + 1)
    return f"{fence}diff\n{diff_str}\n{fence}"


def _apply_edits(content: str, edits: list[dict]) -> str:
    """
    Apply a sequence of edits to content (normalized to LF).

    For each edit:
    1. Exact substring match (first occurrence) — fast path.
    2. Line-by-line flexible match: lines compared with .strip(); original
       indentation of the first matched line is preserved; subsequent new
       lines are adjusted by the same indent delta.
    3. No match → ValueError with TS-matching message.
    """
    # Normalize CRLF → LF for the whole content
    result = content.replace("\r\n", "\n").replace("\r", "\n")

    for edit in edits:
        old = edit["oldText"].replace("\r\n", "\n").replace("\r", "\n")
        new = edit["newText"].replace("\r\n", "\n").replace("\r", "\n")

        # --- 1. Exact match ---
        if old in result:
            result = result.replace(old, new, 1)
            continue

        # --- 2. Flexible line-by-line match ---
        had_trailing_nl = result.endswith("\n")
        result_lines = result.splitlines()  # trailing newline absorbed by splitlines
        old_lines = old.splitlines()

        match_start: int | None = None
        for i in range(len(result_lines) - len(old_lines) + 1):
            if all(
                result_lines[i + j].strip() == old_lines[j].strip()
                for j in range(len(old_lines))
            ):
                match_start = i
                break

        if match_start is None:
            raise ValueError(
                f"Could not find exact match for edit:\n{edit['oldText']}"
            )

        # Compute indentation delta from first matched line
        orig_first = result_lines[match_start]
        orig_indent = len(orig_first) - len(orig_first.lstrip())
        old_first_indent = (
            len(old_lines[0]) - len(old_lines[0].lstrip()) if old_lines else 0
        )
        indent_delta = orig_indent - old_first_indent

        new_lines = new.splitlines()
        adjusted: list[str] = []
        for line in new_lines:
            if not line.strip():
                adjusted.append(line)  # blank lines: preserve as-is
                continue
            line_indent = len(line) - len(line.lstrip())
            new_indent = max(0, line_indent + indent_delta)
            adjusted.append(" " * new_indent + line.lstrip())

        result_parts = (
            result_lines[:match_start]
            + adjusted
            + result_lines[match_start + len(old_lines):]
        )
        result = "\n".join(result_parts)
        if had_trailing_nl:
            result += "\n"

    return result


# ---------------------------------------------------------------------------
# Public operations
# ---------------------------------------------------------------------------

async def read_text_file(
    path: str,
    allowed: Sequence[Path],
    *,
    head: int | None = None,
    tail: int | None = None,
) -> str:
    """
    Read a text file, optionally returning only the first/last N lines.

    Shared handler for read_file (deprecated) and read_text_file.
    """
    if head is not None and tail is not None:
        raise ValueError(
            "Cannot specify both head and tail parameters simultaneously"
        )
    valid = await validate_path(path, allowed)
    _check_size(valid)
    text = valid.read_text(encoding="utf-8", errors="replace")
    if head is not None:
        return "\n".join(text.splitlines()[:head])
    if tail is not None:
        return "\n".join(text.splitlines()[-tail:])
    return text


async def read_multiple_files(paths: list[str], allowed: Sequence[Path]) -> str:
    """
    Read multiple files; individual failures are non-fatal.

    Returns results joined by '\\n---\\n'. Each entry is either
    '{path}:\\n{content}\\n' or '{path}: Error - {message}'.
    """
    results: list[str] = []
    for p in paths:
        try:
            valid = await validate_path(p, allowed)
            _check_size(valid)
            text = valid.read_text(encoding="utf-8", errors="replace")
            results.append(f"{p}:\n{text}\n")
        except Exception as exc:
            results.append(f"{p}: Error - {exc}")
    return "\n---\n".join(results)


async def write_file(path: str, content: str, allowed: Sequence[Path]) -> str:
    """Create or overwrite a file with content (atomic write)."""
    encoded_len = len(content.encode("utf-8"))
    if encoded_len > MAX_FILE_SIZE:
        limit_mb = MAX_FILE_SIZE // (1024 * 1024)
        raise ValueError(
            f"Content too large: {encoded_len} bytes (limit {limit_mb} MB)"
        )
    # Use creation-aware validator: auto-creates missing parents, so immediate
    # parent need not exist yet (Python deviation from TS server behaviour).
    valid = await validate_path_for_creation(path, allowed)
    _atomic_write(valid, content)
    return f"Successfully wrote to {path}"


async def edit_file(
    path: str,
    edits: list[dict] | str,
    allowed: Sequence[Path],
    *,
    dry_run: bool = False,
) -> str:
    """
    Apply line-based edits and return a unified diff.

    GLM model bug: edits may arrive as a JSON string instead of a list.
    Silently parsed if so.
    """
    # GLM coercion: edits sent as JSON string
    if isinstance(edits, str):
        try:
            edits = json.loads(edits)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"edits must be a list of {{oldText, newText}} objects; "
                f"got JSON parse error: {exc}"
            ) from exc

    valid = await validate_path(path, allowed)
    _check_size(valid)
    original = valid.read_text(encoding="utf-8", errors="replace")
    modified = _apply_edits(original, edits)
    diff = _make_diff(original, modified, valid)

    if not dry_run:
        _atomic_write(valid, modified)

    return diff


async def create_directory(path: str, allowed: Sequence[Path]) -> str:
    """Create a directory (and all parents). Idempotent — succeeds if exists."""
    valid = await validate_path_for_creation(path, allowed)
    valid.mkdir(parents=True, exist_ok=True)
    return f"Successfully created directory {path}"


async def list_directory(path: str, allowed: Sequence[Path]) -> str:
    """
    List directory contents with [FILE] / [DIR] prefixes.

    Entries are in filesystem order (not sorted), matching TS behaviour.
    """
    valid = await validate_path(path, allowed)
    entries: list[str] = []
    for entry in valid.iterdir():
        prefix = "[DIR]" if entry.is_dir() else "[FILE]"
        entries.append(f"{prefix} {entry.name}")
    return "\n".join(entries)


async def list_directory_with_sizes(
    path: str,
    allowed: Sequence[Path],
    *,
    sort_by: str = "name",
) -> str:
    """
    List directory with file sizes, totals, and optional sorting.

    Format: name column 30 chars wide, size right-aligned in 10 chars.
    sortBy 'size' sorts descending; 'name' sorts case-insensitively.
    """
    valid = await validate_path(path, allowed)

    # Gather entries: (name, is_dir, size_bytes)
    entries: list[tuple[str, bool, int]] = []
    for entry in valid.iterdir():
        is_dir = entry.is_dir()
        size = 0 if is_dir else entry.stat().st_size
        entries.append((entry.name, is_dir, size))

    if sort_by == "size":
        entries.sort(key=lambda e: e[2], reverse=True)
    else:
        entries.sort(key=lambda e: e[0].lower())

    lines: list[str] = []
    file_count = 0
    dir_count = 0
    total_size = 0

    for name, is_dir, size in entries:
        if is_dir:
            dir_count += 1
            tag = "[DIR] "   # extra space to align with [FILE]
            size_str = ""
        else:
            file_count += 1
            tag = "[FILE]"
            size_str = _format_size(size)
            total_size += size
        # Tag (6) + space (1) + name padded to 30 + size right-aligned to 10
        lines.append(f"{tag} {name:<30}{size_str:>10}")

    lines.append("")
    lines.append(f"Total: {file_count} files, {dir_count} directories")
    if file_count > 0:
        lines.append(f"Combined size: {_format_size(total_size)}")
    return "\n".join(lines)


def _build_tree(
    root: Path,
    base: Path,
    exclude_patterns: list[str],
) -> list[dict]:
    """Recursively build directory tree, skipping excluded entries."""
    entries: list[dict] = []
    try:
        children = sorted(root.iterdir(), key=lambda e: e.name)
    except PermissionError:
        return entries

    for entry in children:
        rel = str(entry.relative_to(base)).replace("\\", "/")
        if _matches_exclude(rel, exclude_patterns):
            continue
        if entry.is_dir():
            sub = _build_tree(entry, base, exclude_patterns)
            entries.append({"name": entry.name, "type": "directory", "children": sub})
        else:
            entries.append({"name": entry.name, "type": "file"})
    return entries


async def directory_tree(
    path: str,
    allowed: Sequence[Path],
    *,
    exclude_patterns: list[str] | None = None,
) -> str:
    """Return recursive directory tree as a JSON string (2-space indent)."""
    if exclude_patterns is None:
        exclude_patterns = []
    valid = await validate_path(path, allowed)
    tree = _build_tree(valid, valid, exclude_patterns)
    return json.dumps(tree, indent=2)


async def move_file(source: str, destination: str, allowed: Sequence[Path]) -> str:
    """
    Move or rename source to destination.

    Fails if destination already exists (TS: fs.rename raises EEXIST).
    Both paths must be within allowed directories.
    """
    valid_src = await validate_path(source, allowed)
    valid_dst = await validate_path(destination, allowed)
    if valid_dst.exists():
        raise FileExistsError(f"Destination already exists: {destination}")
    valid_src.rename(valid_dst)
    return f"Successfully moved {source} to {destination}"


async def search_files(
    path: str,
    pattern: str,
    allowed: Sequence[Path],
    *,
    exclude_patterns: list[str] | None = None,
) -> str:
    """
    Recursively search for files and directories matching a glob pattern.

    Pattern is matched against the path relative to the search root.
    Returns full absolute paths joined by newlines, or 'No matches found'.
    """
    if exclude_patterns is None:
        exclude_patterns = []
    valid = await validate_path(path, allowed)
    matches: list[str] = []

    for dirpath_str, dirnames, filenames in os.walk(valid):
        current = Path(dirpath_str)

        # Prune excluded directories in-place (modifying dirnames stops os.walk descent)
        dirnames[:] = [
            d for d in dirnames
            if not _matches_exclude(
                str(current.relative_to(valid) / d).replace("\\", "/"),
                exclude_patterns,
            )
        ]

        all_names = [(n, True) for n in dirnames] + [(n, False) for n in filenames]
        for name, _ in all_names:
            entry = current / name
            rel = str(entry.relative_to(valid)).replace("\\", "/")
            if _matches_exclude(rel, exclude_patterns):
                continue
            if _glob_match(rel, pattern):
                matches.append(str(entry))

    if not matches:
        return "No matches found"
    return "\n".join(matches)


_MEDIA_MIME_TYPES: dict[str, str] = {
    ".png":  "image/png",
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif":  "image/gif",
    ".webp": "image/webp",
    ".bmp":  "image/bmp",
    ".svg":  "image/svg+xml",
    ".mp3":  "audio/mpeg",
    ".wav":  "audio/wav",
    ".ogg":  "audio/ogg",
    ".flac": "audio/flac",
}


async def read_media_file(
    path: str,
    allowed: Sequence[Path],
) -> tuple[str, str, bytes]:
    """
    Read a media file and return (content_type, mime_type, data).

    content_type is 'image', 'audio', or 'blob' — mirrors the TS server's
    content-type selection logic.  data is the raw file bytes; callers are
    responsible for base64-encoding if needed.
    """
    valid = await validate_path(path, allowed)
    mime_type = _MEDIA_MIME_TYPES.get(valid.suffix.lower(), "application/octet-stream")

    if mime_type.startswith("image/"):
        content_type = "image"
    elif mime_type.startswith("audio/"):
        content_type = "audio"
    else:
        content_type = "blob"

    _check_size(valid)
    return content_type, mime_type, valid.read_bytes()


async def get_file_info(path: str, allowed: Sequence[Path]) -> str:
    """
    Return file/directory metadata as key-value text.

    Matches TS FileInfo struct: size, created, modified, accessed,
    isDirectory, isFile, permissions (octal last 3 digits).
    """
    valid = await validate_path(path, allowed)
    s = valid.stat()

    def _iso(ts: float) -> str:
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        ms = int(ts % 1 * 1000)
        return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{ms:03d}Z"

    perm = oct(s.st_mode)[-3:]
    return "\n".join([
        f"size: {s.st_size}",
        f"created: {_iso(s.st_ctime)}",
        f"modified: {_iso(s.st_mtime)}",
        f"accessed: {_iso(s.st_atime)}",
        f"isDirectory: {str(valid.is_dir()).lower()}",
        f"isFile: {str(valid.is_file()).lower()}",
        f"permissions: {perm}",
    ])
