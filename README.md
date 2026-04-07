# filesystem-mcp

A Python MCP filesystem server that exactly mirrors the [Official MCP TypeScript filesystem server](https://github.com/modelcontextprotocol/servers/tree/main/src/filesystem) API — same tool names, parameter names, descriptions, and error strings. Built for Windows reliability where the official Node.js server has persistent path and atomic-write issues.

## Tools

### TS-compatible tools (mirrors the official MCP TypeScript filesystem server)

`read_file` · `read_text_file` · `read_multiple_files` · `write_file` · `edit_file` · `create_directory` · `list_directory` · `list_directory_with_sizes` · `directory_tree` · `move_file` · `search_files` · `get_file_info` · `list_allowed_directories`

### Python-only additions

#### `grep_files` — search file contents

Recursively search file contents for lines matching a pattern. Contract inspired by [mcp-ripgrep](https://github.com/mcollina/mcp-ripgrep).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `path` | string | required | Root directory or single file to search |
| `pattern` | string | required | Python regex (or literal string if `fixedStrings=True`) |
| `include` | string | `null` | Glob filter on relative file path, e.g. `**/*.py` |
| `caseSensitive` | bool | `true` | Case-sensitive matching |
| `fixedStrings` | bool | `false` | Treat pattern as literal string, not regex |
| `contextLines` | int | `0` | Lines of context before and after each match |
| `maxResults` | int | `null` | Cap on total matching lines returned |

**Output format:**
```
/abs/path/file.py:10:    matched line content
/abs/path/file.py-9-    context line before    ← context uses -
/abs/path/file.py:10:   matched line           ← match uses :
/abs/path/file.py-11-   context line after
--                                              ← separates non-adjacent groups
```

Binary files (null byte in first 8 KB) are silently skipped. Symlinks are never followed.

#### `read_media_file` — read image or audio files

Returns image/audio content with MIME type for vision-capable models.

## Usage

```
uv run run_server.py <allowed_dir> [<allowed_dir> ...]
```

For LM Studio or any MCP host:

```json
{
  "command": "uv",
  "args": ["--directory", "C:/path/to/filesystem-mcp", "run", "run_server.py", "C:/projects"]
}
```

## Windows path support

Accepts all AI-generated path formats: `C:\path`, `C:/path`, `/c/path`, `/mnt/c/path`, bare `C/path`.

## Security

All operations are restricted to the allowed directories specified at startup. Symlink targets are validated. No null bytes. Atomic writes via `tempfile` + `os.replace`.

## Requirements

- Python 3.10+
- [uv](https://github.com/astral-sh/uv)

Dependencies are declared in `pyproject.toml` and locked in `uv.lock`. `uv run` handles the rest.

## License

MIT © 2026 Alex Art

Portions of this software are based on [modelcontextprotocol/servers](https://github.com/modelcontextprotocol/servers), licensed under Apache 2.0.
