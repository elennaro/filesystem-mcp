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

import sys

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
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
