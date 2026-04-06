#!/usr/bin/env python3
"""
Pre-commit privacy guard for filesystem-mcp.

Scans staged files for machine-specific or personal data before every commit.
Patterns are intentionally generic — no actual personal data appears in this script.

Run manually:  python hooks/check_privacy.py
Auto-runs via: .git/hooks/pre-commit (install with: make setup-hooks)
"""

import re
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Patterns that indicate machine-specific or personal data.
# These are GENERIC — they catch any instance of the category,
# not specific values from this machine.
# ---------------------------------------------------------------------------
FORBIDDEN_PATTERNS = [
    # Windows user home directories (any username)
    (r'[Cc]:[/\\]Users[/\\][^/\\\s<>"]{2,}[/\\]',
     'Windows user home path (use <your-path> placeholders in docs, tempfile in tests)'),

    # Unix-style per-user config dirs
    (r'(?<![`\'"/\w])~/\.(claude|mcp|lmstudio|local)\b',
     'User config directory reference (tilde + dot-config dir)'),

    # LM Studio internal state directories
    (r'\.lmstudio[/\\](\.internal|server-logs|bin)\b',
     'LM Studio internal state path'),

    # API keys — sk- prefix style (OpenAI, LM Studio, Anthropic)
    (r'\bsk-[A-Za-z0-9_\-:]{10,}',
     'Possible API key (sk- prefix)'),

    # Bearer tokens
    (r'\bBearer\s+[A-Za-z0-9_\-:\.]{20,}',
     'Possible bearer token'),

    # Generic auth header values
    (r'[Aa]uthorization["\']?\s*[:=]\s*["\']?[A-Za-z0-9_\-:\.]{20,}',
     'Possible auth credential'),
]

# ---------------------------------------------------------------------------
# Files whose full content is checked (source code, configs).
# Documentation files (README, etc.) skip path pattern checks since they
# intentionally contain placeholder path examples.
# ---------------------------------------------------------------------------
DOC_FILES = {'README.md', 'CHANGELOG.md', 'CONTRIBUTING.md'}

# Binary extensions — skip content scanning
BINARY_EXTENSIONS = {
    '.pyc', '.pyo', '.pyd', '.so', '.dll', '.exe',
    '.png', '.jpg', '.jpeg', '.gif', '.ico', '.pdf',
    '.zip', '.tar', '.gz', '.whl',
}


def get_staged_files() -> list[str]:
    result = subprocess.run(
        ['git', 'diff', '--cached', '--name-only', '--diff-filter=ACM'],
        capture_output=True, text=True,
    )
    return [f.strip() for f in result.stdout.splitlines() if f.strip()]


def get_staged_content(filepath: str) -> str:
    result = subprocess.run(
        ['git', 'show', f':{filepath}'],
        capture_output=True, text=True, errors='replace',
    )
    return result.stdout


def check_content(filepath: str, content: str) -> list[tuple[int, str, str]]:
    """Return list of (line_number, description, line_preview) for violations."""
    filename = Path(filepath).name
    violations = []

    # Documentation files: only check for actual secrets, skip path patterns
    if filename in DOC_FILES:
        active_patterns = [
            (p, d) for p, d in FORBIDDEN_PATTERNS
            if any(kw in d.lower() for kw in ('key', 'token', 'credential'))
        ]
    else:
        active_patterns = FORBIDDEN_PATTERNS

    for line_num, line in enumerate(content.splitlines(), start=1):
        # Skip comment lines that are explaining the patterns themselves
        stripped = line.strip()
        if stripped.startswith('#'):
            continue
        for pattern, description in active_patterns:
            if re.search(pattern, line):
                violations.append((line_num, description, stripped[:120]))
                break  # one violation per line is enough

    return violations


def main() -> None:
    staged = get_staged_files()
    if not staged:
        sys.exit(0)

    found_any = False

    for filepath in staged:
        ext = Path(filepath).suffix.lower()
        if ext in BINARY_EXTENSIONS:
            continue

        content = get_staged_content(filepath)
        violations = check_content(filepath, content)

        if violations:
            found_any = True
            print(f'\n  {filepath}')
            for line_num, description, preview in violations:
                print(f'    line {line_num}: [{description}]')
                print(f'    > {preview}')

    if found_any:
        print('\nCommit blocked: personal or machine-specific data detected.')
        print('Fix the violations above, then re-stage and commit.')
        print('Docs: use <placeholder> paths. Tests: use tempfile.mkdtemp().')
        sys.exit(1)

    sys.exit(0)


if __name__ == '__main__':
    main()
