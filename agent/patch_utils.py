from __future__ import annotations

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
