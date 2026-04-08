#!/usr/bin/env python3
"""
Entry point for the MCP filesystem server.

Usage:
    python run_server.py <allowed_dir> [<allowed_dir> ...]

Each positional argument is a directory path the server is allowed to access.
At least one accessible directory is required; the server exits if all
specified directories are inaccessible.

The server communicates via stdio using the MCP protocol.
"""

import io
import sys

# Force UTF-8 on stderr before FastMCP initialises its Rich console.
# On Windows, piped streams use the system code page by default; Rich's
# block-drawing characters can't be encoded and fall back to \uXXXX escapes.
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from mcp_filesystem.server import create_server


def main() -> None:
    if len(sys.argv) < 2:
        print(
            "Usage: python run_server.py <allowed_dir> [<allowed_dir> ...]",
            file=sys.stderr,
        )
        sys.exit(1)

    allowed_dirs = sys.argv[1:]
    mcp = create_server(allowed_dirs)
    mcp.run(transport="stdio", show_banner=False)


if __name__ == "__main__":
    main()
