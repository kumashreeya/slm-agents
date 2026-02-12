# PASTE STARTS
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


def _restrict_rewrite_to_single_tests_file(text: str, target_test_relpath: str) -> tuple[bool, str]:
    """
    Rewrite mode safety:
    - must ONLY target the requested tests/ file via strict markers
    - must not include additional BEGIN FILE markers for other paths
    """
    if not target_test_relpath.startswith("tests/"):
        return False, f"target_test_relpath must be under tests/, got: {target_test_relpath}"

    begin_prefix = "### BEGIN FILE:"
    end_marker = "### END FILE"

    lines = text.splitlines()
    begin_lines = [ln for ln in lines if ln.strip().startswith(begin_prefix)]
    if len(begin_lines) != 1:
        return False, f"expected exactly 1 BEGIN FILE marker, found: {len(begin_lines)}"

    begin_line = begin_lines[0].strip()
    expected_begin = f"{begin_prefix} {target_test_relpath}"
    if begin_line != expected_begin:
        return False, f"BEGIN FILE path mismatch. expected: {expected_begin} got: {begin_line}"

    if end_marker not in text:
        return False, "END FILE marker not found"

    # Ensure END FILE occurs after BEGIN FILE
    begin_idx = text.find(expected_begin)
    end_idx = text.find(end_marker, begin_idx)
    if end_idx == -1:
        return False, "END FILE marker not found after BEGIN FILE marker"

    # Disallow multiple END FILE markers (keeps parsing unambiguous)
    if text.count(end_marker) != 1:
        return False, f"expected exactly 1 END FILE marker, found: {text.count(end_marker)}"

    # No additional BEGIN markers anywhere else (already checked count==1)
    return True, "rewrite markers look valid"


def propose_test_patch(
    repo_dir: Path,
    task_id: str,
    target_source_relpath: str,
    pytest_output: str,
    model: str = "llama3.2:latest",
    output_format: str = "diff",
    target_test_relpath: str = "tests/test_math_utils.py",
) -> TestAgentResult:
    """
    Ask the local SLM to generate/fix pytest tests.

    Modes:
    - output_format="diff" (default): returns a unified diff patch string touching only tests/
    - output_format="rewrite": returns full file content for ONE tests file between strict markers:
        ### BEGIN FILE: tests/...
        <content>
        ### END FILE
    """
    src_path = repo_dir / target_source_relpath
    tests_path = repo_dir / "tests"

    src_text = _read_text(src_path)
    existing_tests = "\n\n".join(_read_text(p) for p in sorted(tests_path.glob("test_*.py")))

    client = LLMClient(model=model)

    if output_format == "rewrite":
        target_test_path = repo_dir / target_test_relpath
        current_test_text = _read_text(target_test_path, max_chars=20000)

        system = (
            "You are a Test Agent for a Python project.\n"
            "Your job: strengthen or fix pytest tests so that `pytest -q` passes.\n"
            "You are operating in REWRITE mode.\n"
            "Rules:\n"
            f"1) You MUST ONLY modify this single file: {target_test_relpath}\n"
            "2) Do NOT create new files.\n"
            "3) Do NOT output a diff.\n"
            "4) Output ONLY the complete file contents using the exact marker format.\n"
            "5) Keep tests deterministic.\n"
        )

        user = (
            f"Task id: {task_id}\n"
            f"Target source file: {target_source_relpath}\n"
            f"Rewrite target test file: {target_test_relpath}\n\n"
            "OUTPUT FORMAT (exact, no extra text):\n"
            f"### BEGIN FILE: {target_test_relpath}\n"
            "<complete file content>\n"
            "### END FILE\n\n"
            "=== SOURCE CODE (reference) ===\n"
            f"{src_text}\n\n"
            "=== CURRENT TARGET TEST FILE CONTENT (rewrite this file) ===\n"
            f"{current_test_text}\n\n"
            "=== FEEDBACK / BRIEF (pytest or mutation brief) ===\n"
            f"{pytest_output}\n\n"
            "Return ONLY the rewritten file in the exact marker format."
        )

        text = client.chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
        )

        if not text.strip():
            return TestAgentResult(False, "", "Model returned empty output (rewrite mode)")

        ok, msg = _restrict_rewrite_to_single_tests_file(text, target_test_relpath)
        if not ok:
            return TestAgentResult(False, text, f"Rewrite output rejected: {msg}")

        return TestAgentResult(True, text, "Rewrite proposed")

    # -------------------------
    # DIFF MODE (existing)
    # -------------------------
    system = (
        "You are a Test Agent for a Python project.\n"
        "Your job: create or fix pytest tests so that `pytest -q` passes.\n"
        "Rules:\n"
        "1) Output ONLY a unified diff patch (git style). No explanations.\n"
        "2) Modify ONLY files under tests/.\n"
        "3) Use pytest. Keep tests deterministic (no network, no randomness unless Hypothesis).\n"
        "4) Prefer simple unit tests + negative tests.\n"
        "5) Patch must be a valid unified diff (diff --git ... @@ ...; hunk lines must start with +, -, or space).\n"
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


# PASTE ENDS
