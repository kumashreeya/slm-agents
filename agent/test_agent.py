from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agent.llm_client import LLMClient


@dataclass
class TestAgentResult:
    ok: bool
    patch: str
    message: str


def _read_text(path: Path, max_chars: int = 12000) -> str:
    if not path.exists():
        return ""
    txt = path.read_text(encoding="utf-8", errors="ignore")
    return txt[:max_chars]


def _restrict_patch_to_tests(patch_text: str) -> bool:
    """
    Safety rule: patch must only touch files under tests/.
    This keeps TestAgent focused.
    """
    for line in patch_text.splitlines():
        if line.startswith("diff --git"):
            # Example: diff --git a/tests/x.py b/tests/x.py
            parts = line.strip().split()
            if len(parts) >= 4:
                a_path = parts[2].removeprefix("a/")
                b_path = parts[3].removeprefix("b/")
                if not (a_path.startswith("tests/") and b_path.startswith("tests/")):
                    return False
    return True


def propose_test_patch(
    repo_dir: Path,
    task_id: str,
    target_source_relpath: str,
    pytest_output: str,
    model: str = "llama3.2:latest",
) -> TestAgentResult:
    """
    Ask the local SLM to generate/fix pytest tests.
    Returns a unified diff patch string.
    """
    src_path = repo_dir / target_source_relpath
    tests_path = repo_dir / "tests"

    src_text = _read_text(src_path)
    existing_tests = "\n\n".join(_read_text(p) for p in sorted(tests_path.glob("test_*.py")))

    system = (
        "You are a Test Agent for a Python project.\n"
        "Your job: create or fix pytest tests so that `pytest -q` passes.\n"
        "Rules:\n"
        "1) Output ONLY a unified diff patch (git style). No explanations.\n"
        "2) Modify ONLY files under tests/.\n"
        "3) Use pytest. Keep tests deterministic (no network, no randomness unless Hypothesis).\n"
        "4) Prefer simple unit tests + negative tests.\n"
    )

    user = (
        f"Task id: {task_id}\n"
        f"Target source file: {target_source_relpath}\n\n"
        "=== SOURCE CODE ===\n"
        f"{src_text}\n\n"
        "=== EXISTING TESTS (may be empty) ===\n"
        f"{existing_tests}\n\n"
        "=== PYTEST OUTPUT ===\n"
        f"{pytest_output}\n\n"
        "Generate or fix tests so pytest passes. Output ONLY the patch."
    )

    client = LLMClient(model=model)
    patch = client.chat(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    )

    if not patch.strip():
        return TestAgentResult(False, "", "Model returned empty output")

    if not _restrict_patch_to_tests(patch):
        return TestAgentResult(False, patch, "Patch touched non-tests/ files (blocked)")

    if "diff --git" not in patch:
        return TestAgentResult(False, patch, "Output is not a git unified diff patch")

    return TestAgentResult(True, patch, "Patch proposed")
