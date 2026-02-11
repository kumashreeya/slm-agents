from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from agent.llm_client import LLMClient


@dataclass
class QualityAgentResult:
    ok: bool
    patch: str
    message: str


def _read_text(path: Path, max_chars: int = 12000) -> str:
    if not path.exists():
        return ""
    txt = path.read_text(encoding="utf-8", errors="ignore")
    return txt[:max_chars]


def _restrict_patch_to_src_and_tests(patch_text: str) -> bool:
    """
    Safety rule: patch must only touch files under src/ or tests/.
    This prevents the quality agent from modifying orchestrator/agent code.
    """
    for line in patch_text.splitlines():
        if line.startswith("diff --git"):
            parts = line.strip().split()
            if len(parts) >= 4:
                a_path = parts[2].removeprefix("a/")
                b_path = parts[3].removeprefix("b/")
                ok_a = a_path.startswith("src/") or a_path.startswith("tests/")
                ok_b = b_path.startswith("src/") or b_path.startswith("tests/")
                if not (ok_a and ok_b):
                    return False
    return True


def _extract_file_paths(tool_text: str) -> list[str]:
    """
    Best-effort extraction of file paths from tool outputs like:
    - src/x.py:10:3: ...
    - tests/y.py:5: ...
    """
    paths: set[str] = set()
    for m in re.finditer(r"(?m)^([a-zA-Z0-9_\-./]+\.py):\d+:\d+:", tool_text):
        paths.add(m.group(1))
    for m in re.finditer(r"(?m)^([a-zA-Z0-9_\-./]+\.py):\d+:", tool_text):
        paths.add(m.group(1))
    # Only keep src/ and tests/
    return sorted([p for p in paths if p.startswith("src/") or p.startswith("tests/")])


def propose_quality_patch(
    repo_dir: Path,
    task_id: str,
    entrypoint_relpath: str,
    failing_tool_output: str,
    model: str = "llama3.2:latest",
) -> QualityAgentResult:
    """
    Ask local SLM to fix issues reported by quality tools (ruff/black/mypy/bandit/pip-audit).
    Output MUST be a unified diff patch touching only src/ or tests/.
    """
    entry_path = repo_dir / entrypoint_relpath
    src_text = _read_text(entry_path)

    related_files = _extract_file_paths(failing_tool_output)
    # Always include entrypoint (even if tools didn't mention it explicitly)
    if entrypoint_relpath not in related_files and entrypoint_relpath.startswith("src/"):
        related_files = [entrypoint_relpath] + related_files

    related_context = []
    for rel in related_files[:6]:  # limit context size
        related_context.append(f"--- FILE: {rel} ---\n{_read_text(repo_dir / rel)}")

    system = (
        "You are a Quality Agent for a Python project.\n"
        "Your job: fix code quality issues so the quality gates pass:\n"
        "- ruff check .\n"
        "- black --check .\n"
        "- mypy src\n"
        "- bandit -r src -q\n"
        "- pip-audit\n\n"
        "Rules:\n"
        "1) Output ONLY a unified diff patch (git style). No explanations.\n"
        "2) Modify ONLY files under src/ or tests/.\n"
        "3) Keep changes minimal. Do not refactor unnecessarily.\n"
        "4) Do not change dependencies or configs.\n"
    )

    user = (
        f"Task id: {task_id}\n"
        f"Entrypoint: {entrypoint_relpath}\n\n"
        "=== ENTRYPOINT SOURCE (for context) ===\n"
        f"{src_text}\n\n"
        "=== RELATED FILES (for context) ===\n" + "\n\n".join(related_context) + "\n\n"
        "=== FAILING TOOL OUTPUT ===\n"
        f"{failing_tool_output}\n\n"
        "Return ONLY the patch that fixes these issues."
    )

    client = LLMClient(model=model)
    patch = client.chat(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    )

    if not patch.strip():
        return QualityAgentResult(False, "", "Model returned empty output")

    if "diff --git" not in patch:
        return QualityAgentResult(False, patch, "Output is not a unified diff patch")

    if not _restrict_patch_to_src_and_tests(patch):
        return QualityAgentResult(False, patch, "Patch touched non src/tests files (blocked)")

    return QualityAgentResult(True, patch, "Patch proposed")
