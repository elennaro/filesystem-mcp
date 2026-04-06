# filesystem-mcp

A Python MCP filesystem server that exactly mirrors the [Official MCP TypeScript filesystem server](https://github.com/modelcontextprotocol/servers/tree/main/src/filesystem) API — same tool names, parameter names, descriptions, and error strings. Built for Windows reliability where the official Node.js server has persistent path and atomic-write issues.

## Tools

Implements all 13 tools from the TS API (excluding `read_media_file`):

`read_file` · `read_text_file` · `read_multiple_files` · `write_file` · `edit_file` · `create_directory` · `list_directory` · `list_directory_with_sizes` · `directory_tree` · `move_file` · `search_files` · `get_file_info` · `list_allowed_directories`

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
