#!/usr/bin/env python3
"""
tests/test_e2e_jun.py — Standalone end-to-end test runner for Jun × filesystem-mcp.

Calls LM Studio /api/v1/chat directly with mcp/filesystem integration,
scores each tool on a 10-point rubric, and prints a summary report.

Run:       python tests/test_e2e_jun.py
Dry run:   python tests/test_e2e_jun.py --dry-run
"""

import argparse
import json
import os
import re
import shutil
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

# ── LM Studio config (mirrors lmstudio_mcp_server.py) ───────────────────────
LMSTUDIO_BASE_URL = "http://localhost:1234"
DEFAULT_MODEL = "qwen3.5-27b@q4_k_xl"
TIMEOUT = 300
INTEGRATIONS = ["mcp/filesystem"]

_MCP_SERVER_PATH = os.path.expanduser("~/.mcp/lmstudio_mcp_server.py")


def _load_api_key() -> str:
    """Read API key from env var or parse it from the MCP server source file."""
    key = os.environ.get("LMSTUDIO_API_KEY", "")
    if key:
        return key
    try:
        with open(_MCP_SERVER_PATH, encoding="utf-8") as f:
            for line in f:
                m = re.match(r'\s*LMSTUDIO_API_KEY\s*=\s*["\']([^"\']+)["\']', line)
                if m:
                    return m.group(1)
    except OSError:
        pass
    return ""


LMSTUDIO_API_KEY = _load_api_key()

SYSTEM_PROMPT = (
    "You are a file management assistant. Use the filesystem MCP tools available "
    "to you. Be precise: use the exact tool that best fits the task. Do not loop. "
    "Complete each task in as few tool calls as possible."
)


# ── LM Studio API ─────────────────────────────────────────────────────────────

def _chat(prompt: str) -> dict:
    """POST to /api/v1/chat and return parsed JSON response."""
    payload = {
        "model": DEFAULT_MODEL,
        "input": prompt,
        "system_prompt": SYSTEM_PROMPT,
        "integrations": INTEGRATIONS,
    }
    req = urllib.request.Request(
        f"{LMSTUDIO_BASE_URL}/api/v1/chat",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {LMSTUDIO_API_KEY}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return json.loads(resp.read())


def _extract_content(data: dict) -> str:
    """Extract message content, strip <think> tags."""
    messages = [
        item["content"]
        for item in data.get("output", [])
        if item.get("type") == "message"
    ]
    content = "\n".join(messages)
    return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()


def _extract_tool_calls(data: dict) -> list:
    """Extract tool_call entries from native API response output array."""
    calls = []
    for item in data.get("output", []):
        if item.get("type") != "tool_call":
            continue
        tool = item.get("tool", "unknown")
        args = item.get("arguments", {})
        raw_output = item.get("output", "")

        error = None
        success = True
        if isinstance(raw_output, str):
            try:
                parsed = json.loads(raw_output)
                if isinstance(parsed, list):
                    for entry in parsed:
                        text = entry.get("text", "")
                        if text.startswith("Error") or text.startswith("error"):
                            error = text[:200]
                            success = False
                            break
            except (json.JSONDecodeError, TypeError):
                pass
            if raw_output.startswith("Error") or raw_output.startswith("error"):
                error = raw_output[:200]
                success = False

        calls.append({
            "tool": tool,
            "arguments": args,
            "raw_output": raw_output,
            "success": success,
            "error": error,
        })
    return calls


# ── Loop detection ────────────────────────────────────────────────────────────

def _detect_loop(tool_calls: list) -> tuple:
    """Return (is_loop, reason). Checks identical args and > 5 calls per tool."""
    tool_counts: dict = {}
    tool_args_seen: dict = {}

    for tc in tool_calls:
        tool = tc["tool"]
        args_str = json.dumps(tc["arguments"], sort_keys=True)

        tool_counts[tool] = tool_counts.get(tool, 0) + 1
        if tool not in tool_args_seen:
            tool_args_seen[tool] = []

        if args_str in tool_args_seen[tool]:
            return True, f"{tool} called with identical args twice"
        tool_args_seen[tool].append(args_str)

        if tool_counts[tool] > 5:
            return True, f"{tool} called {tool_counts[tool]} times (> 5)"

    return False, ""


# ── Fixture setup ─────────────────────────────────────────────────────────────

def create_fixtures(test_dir: Path) -> None:
    test_dir.mkdir(parents=True, exist_ok=True)
    (test_dir / "subdir").mkdir(exist_ok=True)

    (test_dir / "fixture.txt").write_text(
        "Line 1\nLine 2\nLINE_THREE_CONTENT\nLine 4\nLine 5\n", encoding="utf-8"
    )
    (test_dir / "a.txt").write_text("Content of A", encoding="utf-8")
    (test_dir / "b.txt").write_text("Content of B", encoding="utf-8")
    (test_dir / "editable.txt").write_text("Some text BEFORE more text", encoding="utf-8")
    (test_dir / "rename_me.txt").write_text("to be renamed", encoding="utf-8")
    (test_dir / "script.py").write_text(
        "# TODO: fix this bug\nprint('hello')\n", encoding="utf-8"
    )
    (test_dir / "subdir" / "nested.py").write_text(
        "# nested python file\n", encoding="utf-8"
    )


# ── Test definitions ──────────────────────────────────────────────────────────

def make_tests(test_dir: Path) -> list:
    """Build the 13 test scenario dicts. Each has:
      - name, prompt, correct_tool, wrong_tool (optional)
      - verify_params(calls, content) -> (bool, str): params/side-effect check (+2)
      - verify_content(content) -> bool: keyword in response check (+1)
      - wrong_tool_check(calls) -> bool: returns True if wrong tool was used instead
    """
    td = str(test_dir).replace("\\", "/")

    def _has_path(content: str) -> bool:
        return bool(re.search(r"[A-Za-z]:[/\\]\S+|/\S+", content))

    def _has_number(content: str) -> bool:
        return bool(re.search(r"\b\d+\b", content))

    tests = [
        # 1. list_allowed_directories
        {
            "name": "list_allowed_directories",
            "prompt": "What directories are you allowed to access? List them.",
            "correct_tool": "list_allowed_directories",
            "wrong_tool": None,
            "verify_params": lambda calls, content: (
                _has_path(content),
                "no path found in response",
            ),
            "verify_content": lambda content: _has_path(content),
        },
        # 2. list_directory
        {
            "name": "list_directory",
            "prompt": f"List all files and directories inside {td}. Use the exact path.",
            "correct_tool": "list_directory",
            "wrong_tool": None,
            "verify_params": lambda calls, content: (
                "fixture.txt" in content and "script.py" in content,
                "fixture.txt or script.py not in response",
            ),
            "verify_content": lambda content: "fixture.txt" in content,
        },
        # 3. list_directory_with_sizes
        {
            "name": "list_directory_with_sizes",
            "prompt": f"List files in {td} showing sizes in bytes.",
            "correct_tool": "list_directory_with_sizes",
            "wrong_tool": "list_directory",
            "verify_params": lambda calls, content: (
                _has_number(content) and bool(
                    re.search(r"size|byte|\d+\s*B", content, re.IGNORECASE)
                ),
                "no size number found in response",
            ),
            "verify_content": lambda content: _has_number(content),
            "wrong_tool_check": lambda calls: (
                "list_directory_with_sizes" not in [tc["tool"] for tc in calls]
                and "list_directory" in [tc["tool"] for tc in calls]
            ),
        },
        # 4. directory_tree
        {
            "name": "directory_tree",
            "prompt": f"Show the full recursive directory tree of {td} as JSON.",
            "correct_tool": "directory_tree",
            "wrong_tool": None,
            "verify_params": lambda calls, content: (
                "type" in content and "children" in content,
                '"type" or "children" not in response',
            ),
            "verify_content": lambda content: "children" in content,
        },
        # 5. read_text_file
        {
            "name": "read_text_file",
            "prompt": (
                f"Read {td}/fixture.txt and tell me exactly what line 3 says."
            ),
            "correct_tool": "read_text_file",
            "wrong_tool": None,
            "verify_params": lambda calls, content: (
                "LINE_THREE_CONTENT" in content,
                "LINE_THREE_CONTENT not in response",
            ),
            "verify_content": lambda content: "LINE_THREE_CONTENT" in content,
        },
        # 6. read_multiple_files
        {
            "name": "read_multiple_files",
            "prompt": (
                f"Read {td}/a.txt and {td}/b.txt in a single operation."
            ),
            "correct_tool": "read_multiple_files",
            "wrong_tool": "read_text_file",
            "verify_params": lambda calls, content: (
                "Content of A" in content and "Content of B" in content,
                "Content of A or Content of B not in response",
            ),
            "verify_content": lambda content: (
                "Content of A" in content and "Content of B" in content
            ),
            "wrong_tool_check": lambda calls: (
                "read_multiple_files" not in [tc["tool"] for tc in calls]
                and len([tc for tc in calls if tc["tool"] == "read_text_file"]) >= 2
            ),
        },
        # 7. write_file
        {
            "name": "write_file",
            "prompt": (
                f"Create a new file at {td}/written.txt containing exactly: "
                f"WRITE_TEST_OK. Work only inside {td}."
            ),
            "correct_tool": "write_file",
            "wrong_tool": None,
            "verify_params": lambda calls, content, _d=test_dir: (
                (_d / "written.txt").exists()
                and (_d / "written.txt").read_text(encoding="utf-8").strip()
                == "WRITE_TEST_OK",
                "written.txt missing or has wrong content",
            ),
            "verify_content": lambda content: bool(
                re.search(r"creat|writ|success", content, re.IGNORECASE)
            ),
        },
        # 8. edit_file
        {
            "name": "edit_file",
            "prompt": (
                f"In {td}/editable.txt replace the word BEFORE with AFTER. "
                f"Use edit_file. Work only inside {td}."
            ),
            "correct_tool": "edit_file",
            "wrong_tool": "write_file",
            "verify_params": lambda calls, content, _d=test_dir: (
                (_d / "editable.txt").exists()
                and "AFTER"
                in (_d / "editable.txt").read_text(encoding="utf-8")
                and "BEFORE"
                not in (_d / "editable.txt").read_text(encoding="utf-8"),
                "BEFORE still present or AFTER absent in editable.txt",
            ),
            "verify_content": lambda content: bool(
                re.search(r"edit|replac|updat|AFTER", content, re.IGNORECASE)
            ),
            "wrong_tool_check": lambda calls: (
                "edit_file" not in [tc["tool"] for tc in calls]
                and "write_file" in [tc["tool"] for tc in calls]
            ),
        },
        # 9. create_directory
        {
            "name": "create_directory",
            "prompt": (
                f"Create a directory at {td}/newdir/. Work only inside {td}."
            ),
            "correct_tool": "create_directory",
            "wrong_tool": None,
            "verify_params": lambda calls, content, _d=test_dir: (
                (_d / "newdir").is_dir(),
                "newdir directory does not exist",
            ),
            "verify_content": lambda content: bool(
                re.search(r"creat|direct|mkdir|newdir", content, re.IGNORECASE)
            ),
        },
        # 10. move_file
        {
            "name": "move_file",
            "prompt": (
                f"Rename {td}/rename_me.txt to {td}/renamed.txt. "
                f"Work only inside {td}."
            ),
            "correct_tool": "move_file",
            "wrong_tool": None,
            "verify_params": lambda calls, content, _d=test_dir: (
                not (_d / "rename_me.txt").exists()
                and (_d / "renamed.txt").exists()
                and (_d / "renamed.txt").read_text(encoding="utf-8") == "to be renamed",
                "rename_me.txt still exists, renamed.txt missing, or wrong content",
            ),
            "verify_content": lambda content: bool(
                re.search(r"renam|mov|success", content, re.IGNORECASE)
            ),
        },
        # 11. search_files
        {
            "name": "search_files",
            "prompt": (
                f"Find all .py files under {td} by filename. I want file paths."
            ),
            "correct_tool": "search_files",
            "wrong_tool": "grep_files",
            "verify_params": lambda calls, content: (
                "nested.py" in content and "script.py" in content,
                "nested.py or script.py not in response",
            ),
            "verify_content": lambda content: "script.py" in content,
            "wrong_tool_check": lambda calls: (
                "search_files" not in [tc["tool"] for tc in calls]
                and "grep_files" in [tc["tool"] for tc in calls]
            ),
        },
        # 12. grep_files
        {
            "name": "grep_files",
            "prompt": f"Find all lines containing TODO inside files under {td}.",
            "correct_tool": "grep_files",
            "wrong_tool": "read_text_file",
            "verify_params": lambda calls, content: (
                "TODO" in content,
                "TODO not in response",
            ),
            "verify_content": lambda content: "TODO" in content,
            "wrong_tool_check": lambda calls: (
                "grep_files" not in [tc["tool"] for tc in calls]
                and "read_text_file" in [tc["tool"] for tc in calls]
            ),
        },
        # 13. get_file_info
        {
            "name": "get_file_info",
            "prompt": (
                f"What is the size and last-modified time of {td}/fixture.txt?"
            ),
            "correct_tool": "get_file_info",
            "wrong_tool": None,
            "verify_params": lambda calls, content: (
                bool(re.search(r"size", content, re.IGNORECASE))
                and bool(re.search(r"\d{4}", content)),
                "size or year-pattern not found in response",
            ),
            "verify_content": lambda content: bool(
                re.search(r"size", content, re.IGNORECASE)
            ),
        },
    ]

    return tests


# ── Score a single test ───────────────────────────────────────────────────────

def score_test(test: dict, tool_calls: list, content: str) -> tuple:
    """Return (score: int, notes: list[str], is_loop: bool, loop_reason: str)."""
    tools_called = [tc["tool"] for tc in tool_calls]
    correct_tool = test["correct_tool"]
    wrong_tool = test.get("wrong_tool")
    wrong_tool_check = test.get("wrong_tool_check")

    is_loop, loop_reason = _detect_loop(tool_calls)
    if is_loop:
        return 0, [f"LOOP: {loop_reason}"], True, loop_reason

    score = 0
    notes = []

    # +3: correct tool called at least once
    if correct_tool in tools_called:
        score += 3
    else:
        notes.append(f"correct tool '{correct_tool}' not called")

    # +2: no loop detected
    score += 2

    # +2: params correct (filesystem side-effect or response content)
    try:
        params_ok, params_note = test["verify_params"](tool_calls, content)
    except Exception as exc:
        params_ok, params_note = False, f"verify_params raised: {exc}"

    if params_ok:
        score += 2
    else:
        notes.append(f"params: {params_note}")

    # +1: response correctly reflects what the tool returned
    try:
        content_ok = test["verify_content"](content)
    except Exception:
        content_ok = False

    if content_ok:
        score += 1
    else:
        notes.append("response doesn't reflect tool output")

    # +1: completed in <= 2 total tool calls
    if len(tool_calls) <= 2:
        score += 1
    else:
        notes.append(f"{len(tool_calls)} tool calls used (> 2)")

    # +1: correct tool chosen over wrong one
    used_wrong_instead = False
    if wrong_tool_check:
        try:
            used_wrong_instead = wrong_tool_check(tool_calls)
        except Exception:
            pass
    elif wrong_tool and wrong_tool in tools_called and correct_tool not in tools_called:
        used_wrong_instead = True

    if not used_wrong_instead:
        score += 1
    else:
        notes.append(f"wrong tool '{wrong_tool}' used instead of '{correct_tool}'")

    return min(score, 10), notes, False, ""


# ── Run a single test ─────────────────────────────────────────────────────────

def run_test(test: dict) -> dict:
    name = test["name"]
    print(f"  {name} ...", end="", flush=True)

    try:
        data = _chat(test["prompt"])
        content = _extract_content(data)
        tool_calls = _extract_tool_calls(data)

        score, notes, is_loop, loop_reason = score_test(test, tool_calls, content)
        print(f" {score}/10")

        return {
            "name": name,
            "score": score,
            "is_loop": is_loop,
            "loop_reason": loop_reason,
            "tools_called": [tc["tool"] for tc in tool_calls],
            "total_calls": len(tool_calls),
            "notes": notes,
            "error": None,
        }

    except Exception as exc:
        print(f" ERROR")
        return {
            "name": name,
            "score": 0,
            "is_loop": False,
            "loop_reason": "",
            "tools_called": [],
            "total_calls": 0,
            "notes": [],
            "error": str(exc),
        }


# ── Discover base dir ─────────────────────────────────────────────────────────

def discover_base_dir() -> str:
    """Ask Jun what directories it can access; return the first writable path."""
    print("Asking Jun for allowed directories...")
    data = _chat("What directories can you access?")
    content = _extract_content(data)
    tool_calls = _extract_tool_calls(data)

    # Prefer path extracted from tool output
    for tc in tool_calls:
        if tc["tool"] == "list_allowed_directories":
            raw = str(tc.get("raw_output", ""))
            paths = re.findall(r"[A-Za-z]:[/\\][^\s\n\"',\]]+|/[^\s\n\"',\]]+", raw)
            if paths:
                return paths[0].rstrip("/\\")

    # Fall back to parsing response content
    paths = re.findall(r"[A-Za-z]:[/\\][^\s\n\"',\]]+|/[^\s\n\"',\]]+", content)
    if paths:
        return paths[0].rstrip("/\\")

    fallback = str(Path.home())
    print(f"Warning: could not parse base dir from response; using {fallback}")
    return fallback


# ── Report ────────────────────────────────────────────────────────────────────

def print_report(results: list, test_dir: Path, cleanup_ok: bool) -> None:
    total = sum(r["score"] for r in results)
    max_score = len(results) * 10
    pct = round(total / max_score * 100) if max_score else 0

    loops = [r["name"] for r in results if r["is_loop"]]
    errors = [f"{r['name']}: {r['error']}" for r in results if r["error"]]

    print()
    print("=" * 70)
    print("=== filesystem-mcp E2E Test Report ===")
    print(f"Date:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model:     {DEFAULT_MODEL}")
    print(f"Test dir:  {test_dir}")
    print()

    col_tool = 32
    print(f"{'TOOL':<{col_tool}} {'SCORE':>7}  {'LOOPS':>5}  {'CALLS':>5}  NOTES")
    print("-" * 80)

    for r in results:
        loop_marker = "LOOP" if r["is_loop"] else "-"
        notes_str = "; ".join(r["notes"]) if r["notes"] else ""
        if r["error"]:
            notes_str = f"ERROR: {r['error']}"
        # Truncate notes for display
        if len(notes_str) > 60:
            notes_str = notes_str[:57] + "..."
        print(
            f"{r['name']:<{col_tool}} {r['score']:>3}/10  "
            f"{loop_marker:>5}  {r['total_calls']:>5}  {notes_str}"
        )

    print("-" * 80)
    print(f"OVERALL: {total}/{max_score} ({pct}%)")
    print(f"LOOPS DETECTED: {', '.join(loops) if loops else 'none'}")
    print(f"ERRORS: {'; '.join(errors) if errors else 'none'}")
    print(f"Cleanup: {'OK' if cleanup_ok else 'FAILED'}")
    print("=" * 70)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="E2E test runner for Jun × filesystem-mcp integration"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print prompts and fixture plan without calling LM Studio",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.dry_run:
        base_dir = str(Path.home())
        test_dir = Path(base_dir) / f"mcp_e2e_test_{timestamp}"

        print("=== DRY RUN — no LM Studio calls will be made ===")
        print(f"Model:    {DEFAULT_MODEL}")
        print(f"Endpoint: {LMSTUDIO_BASE_URL}/api/v1/chat")
        print(f"Integrations: {INTEGRATIONS}")
        print(f"Test dir: {test_dir}")
        print()
        print("Fixtures to create inside test_dir:")
        fixtures = [
            "fixture.txt      5 lines; line 3 = 'LINE_THREE_CONTENT'",
            "a.txt            'Content of A'",
            "b.txt            'Content of B'",
            "editable.txt     'Some text BEFORE more text'",
            "rename_me.txt    'to be renamed'",
            "script.py        '# TODO: fix this bug\\nprint(\"hello\")'",
            "subdir/nested.py '# nested python file'",
        ]
        for f in fixtures:
            print(f"  {f}")
        print()
        print("System prompt:")
        print(f"  {SYSTEM_PROMPT}")
        print()
        print("Test prompts:")
        for i, test in enumerate(make_tests(test_dir), 1):
            wrong = f" (wrong: {test['wrong_tool']})" if test.get("wrong_tool") else ""
            print(f"  {i:2}. [{test['correct_tool']}{wrong}]")
            print(f"      {test['prompt']}")
            print()
        return

    # ── Check LM Studio is reachable ─────────────────────────────────────────
    print(f"Checking LM Studio at {LMSTUDIO_BASE_URL} ...")
    try:
        req = urllib.request.Request(
            f"{LMSTUDIO_BASE_URL}/v1/models",
            headers={"Authorization": f"Bearer {LMSTUDIO_API_KEY}"},
        )
        with urllib.request.urlopen(req, timeout=10):
            pass
        print("LM Studio reachable.")
    except Exception as exc:
        print(f"ERROR: LM Studio not reachable at {LMSTUDIO_BASE_URL}: {exc}")
        sys.exit(1)

    # ── Discover base dir ─────────────────────────────────────────────────────
    try:
        base_dir = discover_base_dir()
        print(f"Base dir: {base_dir}")
    except Exception as exc:
        print(f"ERROR: Could not discover base dir: {exc}")
        sys.exit(1)

    test_dir = Path(f"{base_dir}/mcp_e2e_test_{timestamp}")
    print(f"Test dir: {test_dir}")

    results: list = []
    cleanup_ok = False

    try:
        # Create fixtures
        print("\nCreating test fixtures ...")
        create_fixtures(test_dir)
        print("Fixtures ready.\n")

        # Run tests
        tests = make_tests(test_dir)
        print(f"Running {len(tests)} tests (timeout {TIMEOUT}s each):\n")
        for test in tests:
            result = run_test(test)
            results.append(result)

    finally:
        # Always attempt cleanup and always print report
        print("\nCleaning up test directory ...")
        try:
            if test_dir.exists():
                shutil.rmtree(test_dir)
                cleanup_ok = True
                print("Cleanup OK.")
            else:
                cleanup_ok = True  # nothing to clean
        except Exception as exc:
            print(f"Cleanup FAILED: {exc}")

        if results:
            print_report(results, test_dir, cleanup_ok)

    if any(r["error"] for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
