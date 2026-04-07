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
    grep_files,
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

    def test_head_stops_early_does_not_read_whole_file(self, tmpdir):
        # 1000 lines; head=3 must return exactly the first 3 regardless of file length
        f = tmpdir / "big.txt"
        f.write_text("\n".join(str(i) for i in range(1000)))
        allowed = resolve_allowed_directories([str(tmpdir)])
        result = run(read_text_file(str(f), allowed, head=3))
        assert result == "0\n1\n2"

    def test_head_crlf_normalized(self, tmpdir):
        f = tmpdir / "crlf.txt"
        f.write_bytes(b"a\r\nb\r\nc\r\n")
        allowed = resolve_allowed_directories([str(tmpdir)])
        result = run(read_text_file(str(f), allowed, head=2))
        assert result == "a\nb"

    def test_head_larger_than_file(self, tmpdir):
        f = tmpdir / "small.txt"
        f.write_text("x\ny\n")
        allowed = resolve_allowed_directories([str(tmpdir)])
        result = run(read_text_file(str(f), allowed, head=100))
        assert result == "x\ny"

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

    def test_excluded_dir_itself_not_in_results(self, tmpdir):
        # Excluded dir should not appear even when pattern would match it
        sub = tmpdir / "dist"
        sub.mkdir()
        (sub / "out.js").write_text("x")
        allowed = resolve_allowed_directories([str(tmpdir)])
        result = run(search_files(
            str(tmpdir), "**/*", allowed,
            exclude_patterns=["dist"],
        ))
        assert "dist" not in result.split("\n")[0] if result != "No matches found" else True
        assert "out.js" not in result

    def test_non_excluded_dir_matches_pattern(self, tmpdir):
        # Dirs that are NOT excluded should still appear in results when matched
        sub = tmpdir / "src"
        sub.mkdir()
        (sub / "main.py").write_text("x")
        allowed = resolve_allowed_directories([str(tmpdir)])
        result = run(search_files(str(tmpdir), "src", allowed))
        assert "src" in result

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


# ---------------------------------------------------------------------------
# grep_files
# ---------------------------------------------------------------------------

class TestGrepFiles:

    # --- Basic matching ---

    def test_basic_match_single_file(self, tmpdir):
        f = tmpdir / "a.txt"
        f.write_text("hello world\ngoodbye\n")
        allowed = resolve_allowed_directories([str(tmpdir)])
        result = run(grep_files(str(tmpdir), "hello", allowed))
        assert "hello world" in result

    def test_basic_match_directory_multi_file(self, tmpdir):
        (tmpdir / "a.txt").write_text("match here\n")
        (tmpdir / "b.txt").write_text("no hit\n")
        (tmpdir / "c.txt").write_text("another match\n")
        allowed = resolve_allowed_directories([str(tmpdir)])
        result = run(grep_files(str(tmpdir), "match", allowed))
        assert "a.txt" in result
        assert "c.txt" in result
        assert "b.txt" not in result

    def test_no_matches_returns_sentinel(self, tmpdir):
        (tmpdir / "f.txt").write_text("nothing here\n")
        allowed = resolve_allowed_directories([str(tmpdir)])
        result = run(grep_files(str(tmpdir), "XYZZY_NOMATCH", allowed))
        assert result == "No matches found"

    def test_output_format_path_colon_lineno_colon_content(self, tmpdir):
        f = tmpdir / "src.py"
        f.write_text("line one\ntarget line\nline three\n")
        allowed = resolve_allowed_directories([str(tmpdir)])
        result = run(grep_files(str(tmpdir), "target", allowed))
        # Format: absolute_path:2:target line
        assert ":2:target line" in result
        assert str(f) in result

    def test_line_numbers_are_one_based(self, tmpdir):
        f = tmpdir / "f.txt"
        f.write_text("alpha\nbeta\ngamma\n")
        allowed = resolve_allowed_directories([str(tmpdir)])
        result = run(grep_files(str(tmpdir), "gamma", allowed))
        assert ":3:gamma" in result

    # --- Regex patterns ---

    def test_regex_digit_class(self, tmpdir):
        (tmpdir / "f.txt").write_text("price: 42 dollars\n")
        allowed = resolve_allowed_directories([str(tmpdir)])
        result = run(grep_files(str(tmpdir), r"\d+", allowed))
        assert "price: 42 dollars" in result

    def test_regex_word_boundary(self, tmpdir):
        (tmpdir / "f.txt").write_text("foo foobar bar\n")
        allowed = resolve_allowed_directories([str(tmpdir)])
        # \bfoo\b matches 'foo' but not 'foobar'
        result = run(grep_files(str(tmpdir), r"\bfoo\b", allowed))
        assert "foo foobar bar" in result  # line matched because 'foo' is in it

    def test_regex_alternation(self, tmpdir):
        (tmpdir / "f.txt").write_text("apple\nbanana\ncherry\n")
        allowed = resolve_allowed_directories([str(tmpdir)])
        result = run(grep_files(str(tmpdir), r"apple|cherry", allowed))
        assert "apple" in result
        assert "cherry" in result
        assert "banana" not in result

    def test_regex_capture_groups_work(self, tmpdir):
        (tmpdir / "f.txt").write_text('{"type": "message"}\n')
        allowed = resolve_allowed_directories([str(tmpdir)])
        result = run(grep_files(str(tmpdir), r'"type":\s*"(\w+)"', allowed))
        assert '"type"' in result

    def test_regex_lookahead(self, tmpdir):
        (tmpdir / "f.txt").write_text("foo123\nfoobar\n")
        allowed = resolve_allowed_directories([str(tmpdir)])
        # foo followed by digit
        result = run(grep_files(str(tmpdir), r"foo(?=\d)", allowed))
        assert "foo123" in result
        assert "foobar" not in result

    def test_regex_anchors_start_and_end(self, tmpdir):
        (tmpdir / "f.txt").write_text("start here\nnot at start\nend\n")
        allowed = resolve_allowed_directories([str(tmpdir)])
        result = run(grep_files(str(tmpdir), r"^start", allowed))
        assert "start here" in result
        assert "not at start" not in result

    def test_regex_inside_json_object(self, tmpdir):
        """Realistic LLM use case: search JSON for a type field."""
        (tmpdir / "data.json").write_text(
            '{"type": "message", "content": "hello"}\n'
            '{"type": "tool_result", "data": {}}\n'
            '{"event": "ping"}\n'
        )
        allowed = resolve_allowed_directories([str(tmpdir)])
        result = run(grep_files(str(tmpdir), r'"type":\s*"\w+"', allowed))
        assert '"type": "message"' in result
        assert '"type": "tool_result"' in result
        assert '"event": "ping"' not in result

    # --- Escaping correctness ---

    def test_double_escaping_fix_regex_metachar_matches_digit(self, tmpdir):
        """\\d must be interpreted as regex 'any digit', not the literal chars \\d.
        Verifies the impl does NOT call re.escape() on regex patterns."""
        (tmpdir / "has_digit.txt").write_text("price: 99 dollars\n")
        # Literal two chars backslash+d — contains no digit characters
        (tmpdir / "has_backslash_d.txt").write_text("pattern: \\d+\n")
        allowed = resolve_allowed_directories([str(tmpdir)])
        result = run(grep_files(str(tmpdir), r"\d", allowed))
        assert "has_digit.txt" in result       # digit found
        assert "has_backslash_d.txt" not in result  # \\d is not a digit

    def test_triple_escaping_fix_escaped_dot_is_literal(self, tmpdir):
        """\\. must match a literal dot (not any char).
        Verifies no extra escaping layer is applied to the pattern."""
        (tmpdir / "with_dot.txt").write_text("file.py\n")
        (tmpdir / "no_dot.txt").write_text("filepy\n")  # no dot
        allowed = resolve_allowed_directories([str(tmpdir)])
        result = run(grep_files(str(tmpdir), r"\.", allowed))
        assert "with_dot.txt" in result   # dot found
        assert "no_dot.txt" not in result  # no dot present

    def test_triple_escaping_literal_backslash_dot_sequence(self, tmpdir):
        r"""Pattern \\\.py (regex: literal backslash then literal dot then py)
        should only match the exact chars \.py in the file."""
        (tmpdir / "regex_file.txt").write_text(r"match \.py here" + "\n")
        (tmpdir / "normal.txt").write_text("match .py here\n")  # dot but no backslash
        allowed = resolve_allowed_directories([str(tmpdir)])
        # r"\\\.py": regex \\\.py = escaped-backslash + escaped-dot + py
        # matches the literal string "\.py"
        result = run(grep_files(str(tmpdir), r"\\\.py", allowed))
        assert "regex_file.txt" in result
        assert "normal.txt" not in result

    def test_fixed_strings_regex_special_chars_treated_literally(self, tmpdir):
        """fixed_strings=True: pattern [a-z]+ is searched literally, not as char class."""
        (tmpdir / "f.txt").write_text("[a-z]+ in file\nno brackets here\n")
        allowed = resolve_allowed_directories([str(tmpdir)])
        result = run(grep_files(str(tmpdir), "[a-z]+", allowed, fixed_strings=True))
        assert "[a-z]+ in file" in result
        assert "no brackets here" not in result

    def test_fixed_strings_backslash_d_is_literal(self, tmpdir):
        r"""fixed_strings=True: pattern \d must match the literal two chars \d,
        not any digit."""
        (tmpdir / "has_digit.txt").write_text("value: 99\n")
        (tmpdir / "has_backslash_d.txt").write_text("pattern: \\d+\n")
        allowed = resolve_allowed_directories([str(tmpdir)])
        result = run(grep_files(str(tmpdir), r"\d", allowed, fixed_strings=True))
        assert "has_backslash_d.txt" in result   # literal \\d matched
        assert "has_digit.txt" not in result     # digit 99 is not the chars \\d

    def test_fixed_strings_dot_does_not_match_any_char(self, tmpdir):
        """fixed_strings=True: 'a.b' matches 'a.b' exactly, not 'axb'."""
        (tmpdir / "exact.txt").write_text("a.b\n")
        (tmpdir / "wild.txt").write_text("axb\n")
        allowed = resolve_allowed_directories([str(tmpdir)])
        result = run(grep_files(str(tmpdir), "a.b", allowed, fixed_strings=True))
        assert "exact.txt" in result
        assert "wild.txt" not in result

    def test_fixed_strings_parens_and_plus_literal(self, tmpdir):
        """fixed_strings=True: '(foo)+' matches the literal string '(foo)+'."""
        (tmpdir / "f.txt").write_text("regex: (foo)+\n")
        (tmpdir / "g.txt").write_text("foofoo\n")
        allowed = resolve_allowed_directories([str(tmpdir)])
        result = run(grep_files(str(tmpdir), "(foo)+", allowed, fixed_strings=True))
        assert "f.txt" in result
        assert "g.txt" not in result

    # --- Case sensitivity ---

    def test_case_sensitive_default(self, tmpdir):
        (tmpdir / "f.txt").write_text("Hello World\nhello world\n")
        allowed = resolve_allowed_directories([str(tmpdir)])
        result = run(grep_files(str(tmpdir), "Hello", allowed))
        assert "Hello World" in result
        assert "hello world" not in result

    def test_case_insensitive(self, tmpdir):
        (tmpdir / "f.txt").write_text("Hello World\nhello world\n")
        allowed = resolve_allowed_directories([str(tmpdir)])
        result = run(grep_files(str(tmpdir), "hello", allowed, case_sensitive=False))
        assert "Hello World" in result
        assert "hello world" in result

    # --- include glob filter ---

    def test_include_filters_by_extension(self, tmpdir):
        (tmpdir / "code.py").write_text("TARGET\n")
        (tmpdir / "data.txt").write_text("TARGET\n")
        allowed = resolve_allowed_directories([str(tmpdir)])
        result = run(grep_files(str(tmpdir), "TARGET", allowed, include="*.py"))
        assert "code.py" in result
        assert "data.txt" not in result

    def test_include_double_star_searches_recursively(self, tmpdir):
        sub = tmpdir / "src"
        sub.mkdir()
        (sub / "deep.py").write_text("TARGET\n")
        (tmpdir / "root.py").write_text("TARGET\n")
        (tmpdir / "root.txt").write_text("TARGET\n")
        allowed = resolve_allowed_directories([str(tmpdir)])
        result = run(grep_files(str(tmpdir), "TARGET", allowed, include="**/*.py"))
        assert "deep.py" in result
        assert "root.py" in result
        assert "root.txt" not in result

    def test_include_no_file_match_returns_no_matches(self, tmpdir):
        (tmpdir / "f.js").write_text("TARGET\n")
        allowed = resolve_allowed_directories([str(tmpdir)])
        result = run(grep_files(str(tmpdir), "TARGET", allowed, include="*.py"))
        assert result == "No matches found"

    # --- Context lines ---

    def test_context_lines_before_and_after(self, tmpdir):
        f = tmpdir / "f.txt"
        f.write_text("before\nmatch line\nafter\n")
        allowed = resolve_allowed_directories([str(tmpdir)])
        result = run(grep_files(str(tmpdir), "match line", allowed, context_lines=1))
        assert "before" in result
        assert "match line" in result
        assert "after" in result

    def test_context_match_line_format_colon(self, tmpdir):
        f = tmpdir / "f.txt"
        f.write_text("before\nmatch\nafter\n")
        allowed = resolve_allowed_directories([str(tmpdir)])
        result = run(grep_files(str(tmpdir), "match", allowed, context_lines=1))
        assert ":2:match" in result  # match line uses :

    def test_context_line_format_dash(self, tmpdir):
        f = tmpdir / "f.txt"
        f.write_text("context\nmatch\ncontext\n")
        allowed = resolve_allowed_directories([str(tmpdir)])
        result = run(grep_files(str(tmpdir), "match", allowed, context_lines=1))
        assert "-1-context" in result  # context line uses -

    def test_context_at_start_of_file_no_lines_before(self, tmpdir):
        f = tmpdir / "f.txt"
        f.write_text("match\nsecond\nthird\n")
        allowed = resolve_allowed_directories([str(tmpdir)])
        result = run(grep_files(str(tmpdir), "match", allowed, context_lines=2))
        lines = result.splitlines()
        # First output line must be the match (no context before line 1)
        assert lines[0].endswith(":1:match")

    def test_context_at_end_of_file_no_lines_after(self, tmpdir):
        f = tmpdir / "f.txt"
        f.write_text("first\nsecond\nmatch\n")
        allowed = resolve_allowed_directories([str(tmpdir)])
        result = run(grep_files(str(tmpdir), "match", allowed, context_lines=2))
        lines = result.splitlines()
        assert lines[-1].endswith(":3:match")

    def test_context_non_adjacent_groups_separated_by_dashes(self, tmpdir):
        f = tmpdir / "f.txt"
        # matches at line 1 and line 10 (far apart with context_lines=1)
        lines = ["line"] * 12
        lines[0] = "match_a"
        lines[9] = "match_b"
        f.write_text("\n".join(lines))
        allowed = resolve_allowed_directories([str(tmpdir)])
        result = run(grep_files(str(tmpdir), "match_", allowed, context_lines=1))
        assert "--" in result

    def test_context_overlapping_windows_merged_no_separator(self, tmpdir):
        f = tmpdir / "f.txt"
        # matches at lines 2 and 4 with context_lines=2 — windows overlap
        lines = ["x", "match_a", "x", "match_b", "x", "x", "x"]
        f.write_text("\n".join(lines))
        allowed = resolve_allowed_directories([str(tmpdir)])
        result = run(grep_files(str(tmpdir), "match_", allowed, context_lines=2))
        assert "--" not in result  # single merged group

    def test_context_zero_no_separator_between_matches(self, tmpdir):
        f = tmpdir / "f.txt"
        f.write_text("match\nother\nmatch\n")
        allowed = resolve_allowed_directories([str(tmpdir)])
        result = run(grep_files(str(tmpdir), "match", allowed, context_lines=0))
        assert "--" not in result

    # --- max_results ---

    def test_max_results_limits_total_match_count(self, tmpdir):
        f = tmpdir / "f.txt"
        f.write_text("\n".join(f"match {i}" for i in range(20)))
        allowed = resolve_allowed_directories([str(tmpdir)])
        result = run(grep_files(str(tmpdir), "match", allowed, max_results=5))
        assert result.count("match") == 5

    def test_max_results_zero_returns_no_matches_found(self, tmpdir):
        (tmpdir / "f.txt").write_text("match\n")
        allowed = resolve_allowed_directories([str(tmpdir)])
        result = run(grep_files(str(tmpdir), "match", allowed, max_results=0))
        assert result == "No matches found"

    def test_max_results_larger_than_total_returns_all(self, tmpdir):
        (tmpdir / "f.txt").write_text("match\nmatch\n")
        allowed = resolve_allowed_directories([str(tmpdir)])
        result = run(grep_files(str(tmpdir), "match", allowed, max_results=100))
        assert result.count("match") == 2

    # --- Binary file handling ---

    def test_binary_file_with_null_byte_skipped(self, tmpdir):
        binary = tmpdir / "data.bin"
        binary.write_bytes(b"match\x00binary\n")
        text = tmpdir / "text.txt"
        text.write_text("match text\n")
        allowed = resolve_allowed_directories([str(tmpdir)])
        result = run(grep_files(str(tmpdir), "match", allowed))
        assert "text.txt" in result
        assert "data.bin" not in result

    def test_non_binary_file_is_searched(self, tmpdir):
        (tmpdir / "f.txt").write_text("hello world\n")
        allowed = resolve_allowed_directories([str(tmpdir)])
        result = run(grep_files(str(tmpdir), "hello", allowed))
        assert "hello world" in result

    # --- Edge cases ---

    def test_empty_file_no_match(self, tmpdir):
        (tmpdir / "empty.txt").write_bytes(b"")
        allowed = resolve_allowed_directories([str(tmpdir)])
        result = run(grep_files(str(tmpdir), "anything", allowed))
        assert result == "No matches found"

    def test_single_line_no_trailing_newline(self, tmpdir):
        f = tmpdir / "f.txt"
        f.write_bytes(b"single line without newline")
        allowed = resolve_allowed_directories([str(tmpdir)])
        result = run(grep_files(str(tmpdir), "single", allowed))
        assert "single line" in result

    def test_crlf_line_endings_no_carriage_return_in_output(self, tmpdir):
        f = tmpdir / "f.txt"
        f.write_bytes(b"first\r\nmatch here\r\nlast\r\n")
        allowed = resolve_allowed_directories([str(tmpdir)])
        result = run(grep_files(str(tmpdir), "match", allowed))
        assert "match here" in result
        assert "\r" not in result  # no carriage return in output

    def test_unicode_content_emoji(self, tmpdir):
        (tmpdir / "f.txt").write_text("hello 🎉 world\n", encoding="utf-8")
        allowed = resolve_allowed_directories([str(tmpdir)])
        result = run(grep_files(str(tmpdir), "🎉", allowed))
        assert "🎉" in result

    def test_unicode_content_cjk(self, tmpdir):
        (tmpdir / "f.txt").write_text("日本語テスト\n", encoding="utf-8")
        allowed = resolve_allowed_directories([str(tmpdir)])
        result = run(grep_files(str(tmpdir), "テスト", allowed))
        assert "日本語テスト" in result

    def test_match_on_first_line(self, tmpdir):
        (tmpdir / "f.txt").write_text("FIRST\nsecond\nthird\n")
        allowed = resolve_allowed_directories([str(tmpdir)])
        result = run(grep_files(str(tmpdir), "FIRST", allowed))
        assert ":1:FIRST" in result

    def test_match_on_last_line(self, tmpdir):
        (tmpdir / "f.txt").write_text("first\nsecond\nLAST\n")
        allowed = resolve_allowed_directories([str(tmpdir)])
        result = run(grep_files(str(tmpdir), "LAST", allowed))
        assert ":3:LAST" in result

    def test_single_file_as_path_argument(self, tmpdir):
        f = tmpdir / "only.txt"
        f.write_text("needle in file\n")
        allowed = resolve_allowed_directories([str(tmpdir)])
        result = run(grep_files(str(f), "needle", allowed))
        assert "needle" in result

    def test_recursive_subdirectory_search(self, tmpdir):
        sub = tmpdir / "sub" / "deep"
        sub.mkdir(parents=True)
        (sub / "buried.txt").write_text("deep match\n")
        allowed = resolve_allowed_directories([str(tmpdir)])
        result = run(grep_files(str(tmpdir), "deep match", allowed))
        assert "buried.txt" in result

    # --- Security ---

    def test_path_outside_allowed_raises_permission_error(self, tmpdir):
        outside = make_tmp()
        try:
            (outside / "f.txt").write_text("TARGET\n")
            allowed = resolve_allowed_directories([str(tmpdir)])
            with pytest.raises(PermissionError, match="Access denied"):
                run(grep_files(str(outside), "TARGET", allowed))
        finally:
            import shutil
            shutil.rmtree(str(outside), ignore_errors=True)

    def test_symlinks_not_followed(self, tmpdir):
        import shutil
        outside = make_tmp()
        try:
            (outside / "secret.txt").write_text("SECRET_CONTENT\n")
            link = tmpdir / "link_to_outside"
            try:
                link.symlink_to(outside, target_is_directory=True)
            except (OSError, NotImplementedError):
                pytest.skip("symlink creation not supported on this system")
            allowed = resolve_allowed_directories([str(tmpdir)])
            result = run(grep_files(str(tmpdir), "SECRET_CONTENT", allowed))
            assert result == "No matches found"
        finally:
            shutil.rmtree(str(outside), ignore_errors=True)

    # --- Invalid patterns ---

    def test_invalid_regex_raises_value_error(self, tmpdir):
        (tmpdir / "f.txt").write_text("content\n")
        allowed = resolve_allowed_directories([str(tmpdir)])
        with pytest.raises(ValueError, match="Invalid regex pattern"):
            run(grep_files(str(tmpdir), "[unclosed", allowed))

    def test_fixed_strings_never_raises_for_regex_invalid_input(self, tmpdir):
        (tmpdir / "f.txt").write_text("[unclosed bracket here\n")
        allowed = resolve_allowed_directories([str(tmpdir)])
        # Would raise if not fixed_strings; must not raise with fixed_strings=True
        result = run(grep_files(str(tmpdir), "[unclosed", allowed, fixed_strings=True))
        assert "[unclosed" in result
