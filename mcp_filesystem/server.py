"""
MCP server for filesystem access.

Registers 13 tools that mirror the Official MCP TypeScript filesystem server
API exactly (except read_media_file, which is excluded from this build).

Tool names, parameter names, descriptions, and error message strings all match
the TS server so that models trained on the TS server's tool definitions work
without adaptation.

Python-only additions vs the TS server:
- Bare-drive path format handled (F/work/...) for GLM model quirk
- write_file / create_directory auto-create parent directories
- edit_file.edits accepts a JSON string (GLM bug workaround)
- 50 MB file size limit
"""

from __future__ import annotations

import sys
from typing import Optional

from fastmcp import FastMCP
from pydantic import BaseModel, Field

from mcp_filesystem.operations import (
    create_directory,
    directory_tree,
    edit_file,
    get_file_info,
    list_directory,
    list_directory_with_sizes,
    move_file,
    read_multiple_files,
    read_text_file,
    search_files,
    write_file,
)
from mcp_filesystem.security import resolve_allowed_directories


class FileEdit(BaseModel):
    oldText: str = Field(description="Text to search for - must match exactly")
    newText: str = Field(description="Text to replace with")


def create_server(allowed_dirs: list[str]) -> FastMCP:
    """
    Create and return a configured FastMCP server.

    Args:
        allowed_dirs: List of directory paths the server is allowed to access.
                      Resolved and validated at startup; server exits if none
                      are accessible.
    """
    allowed = resolve_allowed_directories(allowed_dirs)

    mcp = FastMCP("filesystem")

    # -----------------------------------------------------------------------
    # read_file (deprecated — same handler as read_text_file)
    # -----------------------------------------------------------------------

    @mcp.tool(
        name="read_file",
        description=(
            "Read the complete contents of a file as text. "
            "DEPRECATED: Use read_text_file instead."
        ),
    )
    async def _read_file(
        path: str,
        head: Optional[int] = None,
        tail: Optional[int] = None,
    ) -> str:
        return await read_text_file(path, allowed, head=head, tail=tail)

    # -----------------------------------------------------------------------
    # read_text_file
    # -----------------------------------------------------------------------

    @mcp.tool(
        name="read_text_file",
        description=(
            "Read the complete contents of a file from the file system as text. "
            "Handles various text encodings and provides detailed error messages "
            "if the file cannot be read. Use this tool when you need to examine "
            "the contents of a single file. Use the 'head' parameter to read only "
            "the first N lines of a file, or the 'tail' parameter to read only "
            "the last N lines of a file. Operates on the file as text regardless "
            "of extension. Only works within allowed directories."
        ),
    )
    async def _read_text_file(
        path: str,
        head: Optional[int] = None,
        tail: Optional[int] = None,
    ) -> str:
        return await read_text_file(path, allowed, head=head, tail=tail)

    # -----------------------------------------------------------------------
    # read_multiple_files
    # -----------------------------------------------------------------------

    @mcp.tool(
        name="read_multiple_files",
        description=(
            "Read the contents of multiple files simultaneously. This is more "
            "efficient than reading files one by one when you need to analyze or "
            "compare multiple files. Each file's content is returned with its "
            "path as a reference. Failed reads for individual files won't stop "
            "the entire operation. Only works within allowed directories."
        ),
    )
    async def _read_multiple_files(paths: list[str]) -> str:
        return await read_multiple_files(paths, allowed)

    # -----------------------------------------------------------------------
    # write_file
    # -----------------------------------------------------------------------

    @mcp.tool(
        name="write_file",
        description=(
            "Create a new file or completely overwrite an existing file with new "
            "content. Use with caution as it will overwrite existing files without "
            "warning. Handles text content with proper encoding. Only works within "
            "allowed directories."
        ),
    )
    async def _write_file(path: str, content: str) -> str:
        return await write_file(path, content, allowed)

    # -----------------------------------------------------------------------
    # edit_file
    # -----------------------------------------------------------------------

    @mcp.tool(
        name="edit_file",
        description=(
            "Make line-based edits to a text file. Each edit replaces exact line "
            "sequences with new content. Returns a git-style diff showing the "
            "changes made. Only works within allowed directories."
        ),
    )
    async def _edit_file(
        path: str,
        edits: list[FileEdit],
        dryRun: bool = False,
    ) -> str:
        return await edit_file(path, [e.model_dump() for e in edits], allowed, dry_run=dryRun)

    # -----------------------------------------------------------------------
    # create_directory
    # -----------------------------------------------------------------------

    @mcp.tool(
        name="create_directory",
        description=(
            "Create a new directory or ensure a directory exists. Can create "
            "multiple nested directories in one operation. If the directory "
            "already exists, this operation will succeed silently. Perfect for "
            "setting up directory structures for projects or ensuring required "
            "paths exist. Only works within allowed directories."
        ),
    )
    async def _create_directory(path: str) -> str:
        return await create_directory(path, allowed)

    # -----------------------------------------------------------------------
    # list_directory
    # -----------------------------------------------------------------------

    @mcp.tool(
        name="list_directory",
        description=(
            "Get a detailed listing of all files and directories in a specified "
            "path. Results clearly distinguish between files and directories with "
            "[FILE] and [DIR] prefixes. This tool is essential for understanding "
            "directory structure and finding specific files within a directory. "
            "Only works within allowed directories."
        ),
    )
    async def _list_directory(path: str) -> str:
        return await list_directory(path, allowed)

    # -----------------------------------------------------------------------
    # list_directory_with_sizes
    # -----------------------------------------------------------------------

    @mcp.tool(
        name="list_directory_with_sizes",
        description=(
            "Get a detailed listing of all files and directories in a specified "
            "path, including sizes. Results clearly distinguish between files and "
            "directories with [FILE] and [DIR] prefixes. This tool is useful for "
            "understanding directory structure and finding specific files within a "
            "directory. Only works within allowed directories."
        ),
    )
    async def _list_directory_with_sizes(
        path: str,
        sortBy: str = "name",
    ) -> str:
        return await list_directory_with_sizes(path, allowed, sort_by=sortBy)

    # -----------------------------------------------------------------------
    # directory_tree
    # -----------------------------------------------------------------------

    @mcp.tool(
        name="directory_tree",
        description=(
            "Get a recursive tree view of files and directories as a JSON structure. "
            "Each entry includes 'name', 'type' (file/directory), and 'children' "
            "for directories. Files have no children array, while directories always "
            "have a children array (which may be empty). The output is formatted "
            "with 2-space indentation for readability. Only works within allowed "
            "directories."
        ),
    )
    async def _directory_tree(
        path: str,
        excludePatterns: Optional[list[str]] = None,
    ) -> str:
        return await directory_tree(
            path, allowed, exclude_patterns=excludePatterns or []
        )

    # -----------------------------------------------------------------------
    # move_file
    # -----------------------------------------------------------------------

    @mcp.tool(
        name="move_file",
        description=(
            "Move or rename files and directories. Can move files between "
            "directories and rename them in a single operation. If the destination "
            "exists, the operation will fail. Works across different directories "
            "and can be used for simple renaming within the same directory. Both "
            "source and destination must be within allowed directories."
        ),
    )
    async def _move_file(source: str, destination: str) -> str:
        return await move_file(source, destination, allowed)

    # -----------------------------------------------------------------------
    # search_files
    # -----------------------------------------------------------------------

    @mcp.tool(
        name="search_files",
        description=(
            "Recursively search for files and directories matching a pattern. "
            "The patterns should be glob-style patterns that match paths relative "
            "to the working directory. Use pattern like '*.ext' to match files in "
            "current directory, and '**/*.ext' to match files in all "
            "subdirectories. Returns full paths to all matching items. Great for "
            "finding files when you don't know their exact location. Only searches "
            "within allowed directories."
        ),
    )
    async def _search_files(
        path: str,
        pattern: str,
        excludePatterns: Optional[list[str]] = None,
    ) -> str:
        return await search_files(
            path, pattern, allowed, exclude_patterns=excludePatterns or []
        )

    # -----------------------------------------------------------------------
    # get_file_info
    # -----------------------------------------------------------------------

    @mcp.tool(
        name="get_file_info",
        description=(
            "Retrieve detailed metadata about a file or directory. Returns "
            "comprehensive information including size, creation time, last modified "
            "time, permissions, and type. This tool is perfect for understanding "
            "file characteristics without reading the actual content. Only works "
            "within allowed directories."
        ),
    )
    async def _get_file_info(path: str) -> str:
        return await get_file_info(path, allowed)

    # -----------------------------------------------------------------------
    # list_allowed_directories
    # -----------------------------------------------------------------------

    @mcp.tool(
        name="list_allowed_directories",
        description=(
            "Returns the list of directories that this server is allowed to "
            "access. Subdirectories within these allowed directories are also "
            "accessible. Use this to understand which directories and their "
            "nested paths are available before trying to access files."
        ),
    )
    async def _list_allowed_directories() -> str:
        dirs = "\n".join(str(d) for d in allowed)
        return f"Allowed directories:\n{dirs}"

    return mcp
