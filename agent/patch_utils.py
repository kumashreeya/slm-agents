from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PatchApplyResult:
    ok: bool
    message: str


def apply_unified_diff(repo_dir: Path, patch_text: str) -> PatchApplyResult:
    """
    Apply a unified diff patch using `git apply`.
    Returns ok=False with reason if patch cannot be applied.
    """
    if not patch_text.strip():
        return PatchApplyResult(False, "Empty patch text")

    # Basic sanity check: looks like a diff
    if "diff --git" not in patch_text:
        return PatchApplyResult(False, "Patch does not contain 'diff --git' header")

    try:
        p = subprocess.run(
            ["git", "apply", "--whitespace=nowarn", "-"],
            cwd=repo_dir,
            input=patch_text,
            text=True,
            capture_output=True,
            timeout=60,
        )
        if p.returncode != 0:
            err = (p.stderr or p.stdout or "").strip()
            return PatchApplyResult(False, f"git apply failed: {err[:800]}")
        return PatchApplyResult(True, "Patch applied")
    except Exception as e:
        return PatchApplyResult(False, f"Exception applying patch: {e}")


_REWRITE_BEGIN = "### BEGIN FILE:"
_REWRITE_END = "### END FILE"


def extract_rewrite_file(text: str, expected_path: str) -> tuple[str | None, list[str]]:
    """
    Extract full file contents from an LLM response in the exact format:

    ### BEGIN FILE: <path>
    <full file contents>
    ### END FILE

    Returns (content, errors). Content is normalized to end with exactly one '\n'.
    """
    errors: list[str] = []

    if _REWRITE_BEGIN not in text or _REWRITE_END not in text:
        return None, ["rewrite markers not found"]

    # Find the first matching begin marker for the expected file
    begin_pat = re.compile(rf"^{re.escape(_REWRITE_BEGIN)}\s*(.+?)\s*$", re.MULTILINE)
    begins = list(begin_pat.finditer(text))
    if not begins:
        return None, ["BEGIN marker not found"]

    begin_match = None
    for m in begins:
        path = m.group(1).strip()
        if path == expected_path:
            begin_match = m
            break

    if begin_match is None:
        found = [m.group(1).strip() for m in begins]
        return None, [f"BEGIN marker did not match expected_path={expected_path}", f"found={found}"]

    start_idx = begin_match.end()

    end_idx = text.find(_REWRITE_END, start_idx)
    if end_idx == -1:
        return None, ["END marker not found after BEGIN marker"]

    content = text[start_idx:end_idx]

    # Remove a single leading newline if present (common after marker line)
    if content.startswith("\n"):
        content = content[1:]

    # Normalize newlines and ensure exactly one trailing newline
    content = content.replace("\r\n", "\n").replace("\r", "\n")
    content = content.rstrip("\n") + "\n"

    if not content.strip():
        return None, ["extracted content is empty"]

    return content, errors
