"""
Tests for mcp_filesystem.operations — read, write, edit, list, search, etc.

All temp directories are created inside tests/tmp/ via the tmpdir fixture or
make_tmp().  No system temp directories, no hardcoded machine paths.

Tests are independent of LM Studio and any model.
"""

import asyncio
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from mcp_filesystem.operations import (
    MAX_FILE_SIZE,
    _apply_edits,
    _format_size,
    _glob_match,
    create_directory,
    directory_tree,
    edit_file,
    get_file_info,
    list_directory,
    list_directory_with_sizes,
    move_file,
    read_media_file,
    read_multiple_files,
    read_text_file,
    search_files,
    write_file,
)
from mcp_filesystem.security import resolve_allowed_directories
from tests.conftest import make_tmp


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# _format_size
# ---------------------------------------------------------------------------

class TestFormatSize:
    def test_zero(self):
        assert _format_size(0) == "0 B"

    def test_bytes(self):
        assert _format_size(500) == "500.00 B"

    def test_kilobytes(self):
        assert _format_size(1024) == "1.00 KB"

    def test_megabytes(self):
        assert _format_size(1024 * 1024) == "1.00 MB"

    def test_gigabytes(self):
        assert _format_size(1024 ** 3) == "1.00 GB"


# ---------------------------------------------------------------------------
# _glob_match
# ---------------------------------------------------------------------------

class TestGlobMatch:
    def test_star_matches_name(self):
        assert _glob_match("foo.txt", "*.txt")

    def test_star_does_not_cross_slash(self):
        assert not _glob_match("sub/foo.txt", "*.txt")

    def test_double_star_slash_matches_root(self):
        assert _glob_match("foo.txt", "**/*.txt")

    def test_double_star_slash_matches_nested(self):
        assert _glob_match("a/b/foo.txt", "**/*.txt")

    def test_double_star_matches_everything(self):
        assert _glob_match("a/b/c", "**")

    def test_literal_match(self):
        assert _glob_match("README.md", "README.md")

    def test_literal_no_match(self):
        assert not _glob_match("readme.md", "README.md")

    def test_question_mark(self):
        assert _glob_match("f.txt", "?.txt")
        assert not _glob_match("fo.txt", "?.txt")


# ---------------------------------------------------------------------------
# _apply_edits
# ---------------------------------------------------------------------------

class TestApplyEdits:
    def test_exact_match(self):
        content = "hello world\n"
        result = _apply_edits(content, [{"oldText": "hello", "newText": "hi"}])
        assert result == "hi world\n"

    def test_exact_match_multiline(self):
        content = "line1\nline2\nline3\n"
        result = _apply_edits(
            content,
            [{"oldText": "line1\nline2", "newText": "replaced"}],
        )
        assert result == "replaced\nline3\n"

    def test_flexible_match_trims_whitespace(self):
        content = "    def foo():\n        pass\n"
        result = _apply_edits(
            content,
            [{"oldText": "def foo():\n    pass", "newText": "def bar():\n    return 1"}],
        )
        assert "def bar():" in result
        assert "return 1" in result

    def test_flexible_match_preserves_indent(self):
        content = "    def foo():\n        pass\n"
        # oldText has no indent, content has 4-space indent → delta = +4
        result = _apply_edits(
            content,
            [{"oldText": "def foo():\n    pass", "newText": "def bar():\n    return 1"}],
        )
        # First line should be indented 4 spaces (original indent)
        first_line = result.splitlines()[0]
        assert first_line.startswith("    def bar():")

    def test_no_match_raises(self):
        with pytest.raises(ValueError, match="Could not find exact match"):
            _apply_edits("hello", [{"oldText": "xyz", "newText": "abc"}])

    def test_preserves_trailing_newline(self):
        content = "a\nb\n"
        result = _apply_edits(content, [{"oldText": "a", "newText": "A"}])
        assert result.endswith("\n")

    def test_crlf_normalized(self):
        content = "a\r\nb\r\n"
        result = _apply_edits(content, [{"oldText": "a", "newText": "A"}])
        assert "\r" not in result
        assert result == "A\nb\n"

    def test_multiple_edits_applied_in_sequence(self):
        content = "foo bar baz"
        result = _apply_edits(
            content,
            [
                {"oldText": "foo", "newText": "one"},
                {"oldText": "bar", "newText": "two"},
            ],
        )
        assert result == "one two baz"


# ---------------------------------------------------------------------------
# read_text_file
# ---------------------------------------------------------------------------

class TestReadTextFile:
    def test_reads_full_file(self, tmpdir):
        f = tmpdir / "hello.txt"
        f.write_text("hello world")
        allowed = resolve_allowed_directories([str(tmpdir)])
        result = run(read_text_file(str(f), allowed))
        assert result == "hello world"

    def test_head_parameter(self, tmpdir):
        f = tmpdir / "lines.txt"
        f.write_text("a\nb\nc\nd\n")
        allowed = resolve_allowed_directories([str(tmpdir)])
        result = run(read_text_file(str(f), allowed, head=2))
        assert result == "a\nb"

    def test_tail_parameter(self, tmpdir):
        f = tmpdir / "lines.txt"
        f.write_text("a\nb\nc\nd\n")
        allowed = resolve_allowed_directories([str(tmpdir)])
        result = run(read_text_file(str(f), allowed, tail=2))
        assert result == "c\nd"

    def test_head_and_tail_raises(self, tmpdir):
        f = tmpdir / "x.txt"
        f.write_text("x")
        allowed = resolve_allowed_directories([str(tmpdir)])
        with pytest.raises(ValueError, match="Cannot specify both head and tail"):
            run(read_text_file(str(f), allowed, head=1, tail=1))

    def test_path_outside_allowed_rejected(self, tmpdir):
        other = make_tmp()
        try:
            f = other / "secret.txt"
            f.write_text("secret")
            allowed = resolve_allowed_directories([str(tmpdir)])
            with pytest.raises(PermissionError):
                run(read_text_file(str(f), allowed))
        finally:
            import shutil
            shutil.rmtree(other, ignore_errors=True)

    def test_size_limit_enforced(self, tmpdir):
        f = tmpdir / "big.txt"
        f.write_text("x")
        allowed = resolve_allowed_directories([str(tmpdir)])
        with patch("mcp_filesystem.operations.MAX_FILE_SIZE", 0):
            with pytest.raises(ValueError, match="File too large"):
                run(read_text_file(str(f), allowed))


# ---------------------------------------------------------------------------
# read_multiple_files
# ---------------------------------------------------------------------------

class TestReadMultipleFiles:
    def test_reads_multiple(self, tmpdir):
        a = tmpdir / "a.txt"
        b = tmpdir / "b.txt"
        a.write_text("AAA")
        b.write_text("BBB")
        allowed = resolve_allowed_directories([str(tmpdir)])
        result = run(read_multiple_files([str(a), str(b)], allowed))
        assert "AAA" in result
        assert "BBB" in result

    def test_separator_between_files(self, tmpdir):
        a = tmpdir / "a.txt"
        b = tmpdir / "b.txt"
        a.write_text("A")
        b.write_text("B")
        allowed = resolve_allowed_directories([str(tmpdir)])
        result = run(read_multiple_files([str(a), str(b)], allowed))
        assert "\n---\n" in result

    def test_individual_failure_non_fatal(self, tmpdir):
        good = tmpdir / "good.txt"
        good.write_text("good")
        allowed = resolve_allowed_directories([str(tmpdir)])
        bad_path = str(tmpdir / "nonexistent.txt")
        result = run(read_multiple_files([str(good), bad_path], allowed))
        assert "good" in result
        assert "Error" in result


# ---------------------------------------------------------------------------
# write_file
# ---------------------------------------------------------------------------

class TestWriteFile:
    def test_creates_new_file(self, tmpdir):
        target = tmpdir / "new.txt"
        allowed = resolve_allowed_directories([str(tmpdir)])
        result = run(write_file(str(target), "hello", allowed))
        assert "Successfully wrote" in result
        assert target.read_text() == "hello"

    def test_overwrites_existing(self, tmpdir):
        target = tmpdir / "existing.txt"
        target.write_text("old")
        allowed = resolve_allowed_directories([str(tmpdir)])
        run(write_file(str(target), "new content", allowed))
        assert target.read_text() == "new content"

    def test_creates_parent_dirs(self, tmpdir):
        target = tmpdir / "sub" / "nested" / "file.txt"
        allowed = resolve_allowed_directories([str(tmpdir)])
        run(write_file(str(target), "data", allowed))
        assert target.read_text() == "data"

    def test_size_limit_enforced(self, tmpdir):
        target = tmpdir / "big.txt"
        allowed = resolve_allowed_directories([str(tmpdir)])
        with patch("mcp_filesystem.operations.MAX_FILE_SIZE", 3):
            with pytest.raises(ValueError, match="Content too large"):
                run(write_file(str(target), "hello", allowed))

    def test_path_outside_allowed_rejected(self, tmpdir):
        other = make_tmp()
        try:
            target = other / "file.txt"
            allowed = resolve_allowed_directories([str(tmpdir)])
            with pytest.raises(PermissionError):
                run(write_file(str(target), "x", allowed))
        finally:
            import shutil
            shutil.rmtree(other, ignore_errors=True)

    def test_no_leftover_temp_file_on_success(self, tmpdir):
        target = tmpdir / "file.txt"
        allowed = resolve_allowed_directories([str(tmpdir)])
        run(write_file(str(target), "data", allowed))
        tmp_files = list(tmpdir.glob("*.tmp"))
        assert tmp_files == []


# ---------------------------------------------------------------------------
# edit_file
# ---------------------------------------------------------------------------

class TestEditFile:
    def test_exact_match_edit(self, tmpdir):
        f = tmpdir / "code.py"
        f.write_text("x = 1\ny = 2\n")
        allowed = resolve_allowed_directories([str(tmpdir)])
        result = run(edit_file(
            str(f), [{"oldText": "x = 1", "newText": "x = 99"}], allowed
        ))
        assert "diff" in result
        assert f.read_text() == "x = 99\ny = 2\n"

    def test_dry_run_does_not_write(self, tmpdir):
        f = tmpdir / "code.py"
        original = "x = 1\n"
        f.write_text(original)
        allowed = resolve_allowed_directories([str(tmpdir)])
        run(edit_file(
            str(f), [{"oldText": "x = 1", "newText": "x = 99"}], allowed,
            dry_run=True,
        ))
        assert f.read_text() == original

    def test_glm_coercion_json_string(self, tmpdir):
        f = tmpdir / "code.py"
        f.write_text("hello world\n")
        allowed = resolve_allowed_directories([str(tmpdir)])
        edits_str = json.dumps([{"oldText": "hello", "newText": "hi"}])
        run(edit_file(str(f), edits_str, allowed))
        assert f.read_text() == "hi world\n"

    def test_no_match_raises(self, tmpdir):
        f = tmpdir / "code.py"
        f.write_text("hello\n")
        allowed = resolve_allowed_directories([str(tmpdir)])
        with pytest.raises(ValueError, match="Could not find exact match"):
            run(edit_file(
                str(f), [{"oldText": "xyz", "newText": "abc"}], allowed
            ))

    def test_diff_output_format(self, tmpdir):
        f = tmpdir / "code.py"
        f.write_text("a\nb\n")
        allowed = resolve_allowed_directories([str(tmpdir)])
        result = run(edit_file(
            str(f), [{"oldText": "a", "newText": "A"}], allowed
        ))
        # Should be wrapped in a backtick fence
        assert result.startswith("```")
        assert "diff" in result.split("\n")[0]


# ---------------------------------------------------------------------------
# create_directory
# ---------------------------------------------------------------------------

class TestCreateDirectory:
    def test_creates_new_dir(self, tmpdir):
        new_dir = tmpdir / "newsubdir"
        allowed = resolve_allowed_directories([str(tmpdir)])
        result = run(create_directory(str(new_dir), allowed))
        assert "Successfully created directory" in result
        assert new_dir.is_dir()

    def test_idempotent_on_existing(self, tmpdir):
        existing = tmpdir / "exists"
        existing.mkdir()
        allowed = resolve_allowed_directories([str(tmpdir)])
        # Should not raise
        run(create_directory(str(existing), allowed))
        assert existing.is_dir()

    def test_creates_nested(self, tmpdir):
        nested = tmpdir / "a" / "b" / "c"
        allowed = resolve_allowed_directories([str(tmpdir)])
        run(create_directory(str(nested), allowed))
        assert nested.is_dir()


# ---------------------------------------------------------------------------
# list_directory
# ---------------------------------------------------------------------------

class TestListDirectory:
    def test_lists_files_and_dirs(self, tmpdir):
        (tmpdir / "file.txt").write_text("x")
        (tmpdir / "subdir").mkdir()
        allowed = resolve_allowed_directories([str(tmpdir)])
        result = run(list_directory(str(tmpdir), allowed))
        assert "[FILE] file.txt" in result
        assert "[DIR] subdir" in result

    def test_empty_dir(self, tmpdir):
        allowed = resolve_allowed_directories([str(tmpdir)])
        result = run(list_directory(str(tmpdir), allowed))
        assert result == ""


# ---------------------------------------------------------------------------
# list_directory_with_sizes
# ---------------------------------------------------------------------------

class TestListDirectoryWithSizes:
    def test_includes_sizes(self, tmpdir):
        f = tmpdir / "data.txt"
        f.write_text("hello")
        allowed = resolve_allowed_directories([str(tmpdir)])
        result = run(list_directory_with_sizes(str(tmpdir), allowed))
        assert "[FILE]" in result
        assert "data.txt" in result
        assert "B" in result  # some size unit

    def test_includes_totals(self, tmpdir):
        (tmpdir / "a.txt").write_text("hi")
        (tmpdir / "sub").mkdir()
        allowed = resolve_allowed_directories([str(tmpdir)])
        result = run(list_directory_with_sizes(str(tmpdir), allowed))
        assert "Total:" in result
        assert "1 files" in result
        assert "1 directories" in result

    def test_sort_by_name(self, tmpdir):
        (tmpdir / "z.txt").write_text("z")
        (tmpdir / "a.txt").write_text("a")
        allowed = resolve_allowed_directories([str(tmpdir)])
        result = run(list_directory_with_sizes(str(tmpdir), allowed, sort_by="name"))
        lines = [l for l in result.splitlines() if l.strip()]
        names = [l.split()[1] for l in lines if "[FILE]" in l]
        assert names == sorted(names, key=str.lower)

    def test_sort_by_size(self, tmpdir):
        (tmpdir / "big.txt").write_text("x" * 1000)
        (tmpdir / "small.txt").write_text("x")
        allowed = resolve_allowed_directories([str(tmpdir)])
        result = run(list_directory_with_sizes(str(tmpdir), allowed, sort_by="size"))
        lines = [l for l in result.splitlines() if "[FILE]" in l]
        # big.txt should appear before small.txt (descending size)
        assert lines[0].split()[1] == "big.txt"


# ---------------------------------------------------------------------------
# directory_tree
# ---------------------------------------------------------------------------

class TestDirectoryTree:
    def test_basic_tree(self, tmpdir):
        (tmpdir / "a.txt").write_text("x")
        sub = tmpdir / "subdir"
        sub.mkdir()
        (sub / "b.txt").write_text("y")
        allowed = resolve_allowed_directories([str(tmpdir)])
        result = run(directory_tree(str(tmpdir), allowed))
        tree = json.loads(result)
        assert any(e["name"] == "a.txt" and e["type"] == "file" for e in tree)
        sub_entry = next(e for e in tree if e["name"] == "subdir")
        assert sub_entry["type"] == "directory"
        assert "children" in sub_entry
        assert any(c["name"] == "b.txt" for c in sub_entry["children"])

    def test_exclude_pattern(self, tmpdir):
        (tmpdir / "keep.txt").write_text("x")
        (tmpdir / "node_modules").mkdir()
        (tmpdir / "node_modules" / "pkg.js").write_text("y")
        allowed = resolve_allowed_directories([str(tmpdir)])
        result = run(directory_tree(
            str(tmpdir), allowed, exclude_patterns=["node_modules"]
        ))
        tree = json.loads(result)
        names = [e["name"] for e in tree]
        assert "node_modules" not in names
        assert "keep.txt" in names

    def test_files_have_no_children(self, tmpdir):
        (tmpdir / "f.txt").write_text("x")
        allowed = resolve_allowed_directories([str(tmpdir)])
        result = run(directory_tree(str(tmpdir), allowed))
        tree = json.loads(result)
        file_entry = next(e for e in tree if e["name"] == "f.txt")
        assert "children" not in file_entry


# ---------------------------------------------------------------------------
# move_file
# ---------------------------------------------------------------------------

class TestMoveFile:
    def test_rename_in_place(self, tmpdir):
        src = tmpdir / "old.txt"
        dst = tmpdir / "new.txt"
        src.write_text("data")
        allowed = resolve_allowed_directories([str(tmpdir)])
        result = run(move_file(str(src), str(dst), allowed))
        assert "Successfully moved" in result
        assert not src.exists()
        assert dst.read_text() == "data"

    def test_move_to_subdir(self, tmpdir):
        src = tmpdir / "file.txt"
        sub = tmpdir / "subdir"
        sub.mkdir()
        dst = sub / "file.txt"
        src.write_text("content")
        allowed = resolve_allowed_directories([str(tmpdir)])
        run(move_file(str(src), str(dst), allowed))
        assert dst.read_text() == "content"

    def test_fails_if_destination_exists(self, tmpdir):
        src = tmpdir / "a.txt"
        dst = tmpdir / "b.txt"
        src.write_text("a")
        dst.write_text("b")
        allowed = resolve_allowed_directories([str(tmpdir)])
        with pytest.raises(FileExistsError):
            run(move_file(str(src), str(dst), allowed))

    def test_source_outside_allowed_rejected(self, tmpdir):
        other = make_tmp()
        try:
            src = other / "file.txt"
            src.write_text("x")
            dst = tmpdir / "file.txt"
            allowed = resolve_allowed_directories([str(tmpdir)])
            with pytest.raises(PermissionError):
                run(move_file(str(src), str(dst), allowed))
        finally:
            import shutil
            shutil.rmtree(other, ignore_errors=True)


# ---------------------------------------------------------------------------
# search_files
# ---------------------------------------------------------------------------

class TestSearchFiles:
    def test_finds_matching_files_by_extension(self, tmpdir):
        (tmpdir / "a.txt").write_text("x")
        (tmpdir / "b.txt").write_text("y")
        (tmpdir / "c.py").write_text("z")
        allowed = resolve_allowed_directories([str(tmpdir)])
        result = run(search_files(str(tmpdir), "*.txt", allowed))
        assert "a.txt" in result
        assert "b.txt" in result
        assert "c.py" not in result

    def test_recursive_with_double_star(self, tmpdir):
        sub = tmpdir / "sub"
        sub.mkdir()
        (sub / "deep.txt").write_text("x")
        (tmpdir / "root.txt").write_text("y")
        allowed = resolve_allowed_directories([str(tmpdir)])
        result = run(search_files(str(tmpdir), "**/*.txt", allowed))
        assert "deep.txt" in result
        assert "root.txt" in result

    def test_no_matches(self, tmpdir):
        (tmpdir / "a.py").write_text("x")
        allowed = resolve_allowed_directories([str(tmpdir)])
        result = run(search_files(str(tmpdir), "*.txt", allowed))
        assert result == "No matches found"

    def test_exclude_patterns(self, tmpdir):
        sub = tmpdir / "node_modules"
        sub.mkdir()
        (sub / "pkg.txt").write_text("x")
        (tmpdir / "keep.txt").write_text("y")
        allowed = resolve_allowed_directories([str(tmpdir)])
        result = run(search_files(
            str(tmpdir), "**/*.txt", allowed,
            exclude_patterns=["node_modules"],
        ))
        assert "keep.txt" in result
        assert "pkg.txt" not in result

    def test_star_does_not_cross_slash(self, tmpdir):
        sub = tmpdir / "sub"
        sub.mkdir()
        (sub / "file.txt").write_text("x")
        (tmpdir / "root.txt").write_text("y")
        allowed = resolve_allowed_directories([str(tmpdir)])
        result = run(search_files(str(tmpdir), "*.txt", allowed))
        # *.txt should match root.txt (relative = "root.txt") but not sub/file.txt
        assert "root.txt" in result
        assert "file.txt" not in result


# ---------------------------------------------------------------------------
# get_file_info
# ---------------------------------------------------------------------------

class TestGetFileInfo:
    def test_file_info_fields(self, tmpdir):
        f = tmpdir / "data.txt"
        f.write_text("hello")
        allowed = resolve_allowed_directories([str(tmpdir)])
        result = run(get_file_info(str(f), allowed))
        assert "size: 5" in result
        assert "isFile: true" in result
        assert "isDirectory: false" in result
        assert "modified:" in result
        assert "created:" in result
        assert "permissions:" in result

    def test_dir_info(self, tmpdir):
        sub = tmpdir / "subdir"
        sub.mkdir()
        allowed = resolve_allowed_directories([str(tmpdir)])
        result = run(get_file_info(str(sub), allowed))
        assert "isDirectory: true" in result
        assert "isFile: false" in result

    def test_timestamp_format(self, tmpdir):
        f = tmpdir / "f.txt"
        f.write_text("x")
        allowed = resolve_allowed_directories([str(tmpdir)])
        result = run(get_file_info(str(f), allowed))
        # Timestamps should match ISO 8601 with Z suffix
        import re
        ts_pattern = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z"
        modified_line = next(l for l in result.splitlines() if l.startswith("modified:"))
        assert re.search(ts_pattern, modified_line)


# ---------------------------------------------------------------------------
# read_media_file
# ---------------------------------------------------------------------------

class TestReadMediaFile:
    def test_image_content_type_and_mime(self, tmpdir):
        p = tmpdir / "photo.png"
        p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
        allowed = resolve_allowed_directories([str(tmpdir)])
        ct, mime, data = run(read_media_file(str(p), allowed))
        assert ct == "image"
        assert mime == "image/png"
        assert data == b"\x89PNG\r\n\x1a\n" + b"\x00" * 16

    def test_jpeg_extension(self, tmpdir):
        p = tmpdir / "photo.jpg"
        p.write_bytes(b"\xff\xd8\xff")
        allowed = resolve_allowed_directories([str(tmpdir)])
        ct, mime, _ = run(read_media_file(str(p), allowed))
        assert ct == "image"
        assert mime == "image/jpeg"

    def test_jpeg_long_extension(self, tmpdir):
        p = tmpdir / "photo.jpeg"
        p.write_bytes(b"\xff\xd8\xff")
        allowed = resolve_allowed_directories([str(tmpdir)])
        _, mime, _ = run(read_media_file(str(p), allowed))
        assert mime == "image/jpeg"

    def test_svg_mime(self, tmpdir):
        p = tmpdir / "icon.svg"
        p.write_bytes(b"<svg/>")
        allowed = resolve_allowed_directories([str(tmpdir)])
        ct, mime, _ = run(read_media_file(str(p), allowed))
        assert ct == "image"
        assert mime == "image/svg+xml"

    def test_audio_mp3(self, tmpdir):
        p = tmpdir / "track.mp3"
        p.write_bytes(b"\xff\xfb" + b"\x00" * 16)
        allowed = resolve_allowed_directories([str(tmpdir)])
        ct, mime, _ = run(read_media_file(str(p), allowed))
        assert ct == "audio"
        assert mime == "audio/mpeg"

    def test_audio_wav(self, tmpdir):
        p = tmpdir / "sound.wav"
        p.write_bytes(b"RIFF" + b"\x00" * 36)
        allowed = resolve_allowed_directories([str(tmpdir)])
        ct, mime, _ = run(read_media_file(str(p), allowed))
        assert ct == "audio"
        assert mime == "audio/wav"

    def test_blob_for_unknown_extension(self, tmpdir):
        p = tmpdir / "data.bin"
        p.write_bytes(b"\x00\x01\x02")
        allowed = resolve_allowed_directories([str(tmpdir)])
        ct, mime, _ = run(read_media_file(str(p), allowed))
        assert ct == "blob"
        assert mime == "application/octet-stream"

    def test_data_roundtrip(self, tmpdir):
        payload = b"\xde\xad\xbe\xef" * 8
        p = tmpdir / "img.gif"
        p.write_bytes(payload)
        allowed = resolve_allowed_directories([str(tmpdir)])
        _, _, data = run(read_media_file(str(p), allowed))
        assert data == payload

    def test_path_outside_allowed_rejected(self, tmpdir):
        other = make_tmp()
        p = other / "img.png"
        p.write_bytes(b"")
        allowed = resolve_allowed_directories([str(tmpdir)])
        with pytest.raises(PermissionError, match="Access denied"):
            run(read_media_file(str(p), allowed))
