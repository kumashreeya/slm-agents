from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agent.llm_client import LLMClient


@dataclass
class CodeAgentResult:
    ok: bool
    patch: str
    message: str


def _read_text(path: Path, max_chars: int = 12000) -> str:
    if not path.exists():
        return ""
    txt = path.read_text(encoding="utf-8", errors="ignore")
    return txt[:max_chars]


def _restrict_patch_to_src(patch_text: str) -> bool:
    """
    Safety rule: patch must only touch files under src/.
    This keeps CodeAgent focused on application code.
    """
    for line in patch_text.splitlines():
        if line.startswith("diff --git"):
            # Example: diff --git a/src/x.py b/src/x.py
            parts = line.strip().split()
            if len(parts) >= 4:
                a_path = parts[2].removeprefix("a/")
                b_path = parts[3].removeprefix("b/")
                if not (a_path.startswith("src/") and b_path.startswith("src/")):
                    return False
    return True


def _restrict_generate_to_single_src_file(text: str, target_src_relpath: str) -> tuple[bool, str]:
    """
    Generate mode safety:
    - must ONLY target the requested src/ file via strict markers
    - must not include additional BEGIN FILE markers for other paths
    """
    if not target_src_relpath.startswith("src/"):
        return False, f"target_src_relpath must be under src/, got: {target_src_relpath}"

    begin_prefix = "### BEGIN FILE:"
    end_marker = "### END FILE"

    lines = text.splitlines()
    begin_lines = [ln for ln in lines if ln.strip().startswith(begin_prefix)]
    if len(begin_lines) != 1:
        return False, f"expected exactly 1 BEGIN FILE marker, found: {len(begin_lines)}"

    begin_line = begin_lines[0].strip()
    expected_begin = f"{begin_prefix} {target_src_relpath}"
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

    return True, "generate markers look valid"


def propose_code_patch(
    repo_dir: Path,
    task_id: str,
    target_source_relpath: str,
    model: str = "llama3.2:latest",
    mode: str = "fix",
    feature_description: str = "",
    test_failure_output: str = "",
) -> CodeAgentResult:
    """
    Ask the local SLM to generate or fix source code.

    Modes:
    - mode="fix" (default): Fix code based on test failures. Returns unified diff patch touching only src/
    - mode="generate": Generate new code from description. Returns full file content between markers:
        ### BEGIN FILE: src/...
        <content>
        ### END FILE

    Args:
        repo_dir: Repository root directory
        task_id: Task identifier
        target_source_relpath: Target source file path (e.g., "src/example_pkg/math_utils.py")
        model: LLM model name
        mode: "fix" or "generate"
        feature_description: Description for generate mode
        test_failure_output: Test failures for fix mode
    """
    src_path = repo_dir / target_source_relpath
    tests_path = repo_dir / "tests"

    # Read existing code and tests for context
    existing_code = _read_text(src_path)
    existing_tests = "\n\n".join(_read_text(p) for p in sorted(tests_path.glob("test_*.py")))

    client = LLMClient(model=model)

    # -------------------------
    # GENERATE MODE (new code from description)
    # -------------------------
    if mode == "generate":
        if not feature_description.strip():
            return CodeAgentResult(False, "", "Generate mode requires feature_description")

        system = (
            "You are a Code Agent for a Python project.\n"
            "Your job: generate clean, working Python code based on user requirements.\n"
            "You are operating in GENERATE mode.\n"
            "Rules:\n"
            f"1) You MUST ONLY output code for this single file: {target_source_relpath}\n"
            "2) Output ONLY the complete file contents using the exact marker format.\n"
            "3) Do NOT output a diff.\n"
            "4) Include proper imports, type hints, and docstrings.\n"
            "5) Keep code simple, readable, and well-structured.\n"
            "6) Follow PEP 8 style guidelines.\n"
        )

        user = (
            f"Task id: {task_id}\n"
            f"Target source file: {target_source_relpath}\n\n"
            "FEATURE DESCRIPTION:\n"
            f"{feature_description}\n\n"
            "OUTPUT FORMAT (exact, no extra text):\n"
            f"### BEGIN FILE: {target_source_relpath}\n"
            "<complete file content with imports, functions, classes>\n"
            "### END FILE\n\n"
        )

        if existing_code.strip():
            user += "=== EXISTING CODE (reference, extend if needed) ===\n" f"{existing_code}\n\n"

        text = client.chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
        )

        if not text.strip():
            return CodeAgentResult(False, "", "Model returned empty output (generate mode)")

        ok, msg = _restrict_generate_to_single_src_file(text, target_source_relpath)
        if not ok:
            return CodeAgentResult(False, text, f"Generate output rejected: {msg}")

        return CodeAgentResult(True, text, "Code generation proposed")

    # -------------------------
    # FIX MODE (patch existing code based on test failures)
    # -------------------------
    if mode == "fix":
        if not test_failure_output.strip():
            return CodeAgentResult(False, "", "Fix mode requires test_failure_output")

        system = (
            "You are a Code Agent for a Python project.\n"
            "Your job: fix bugs in source code so that tests pass.\n"
            "Rules:\n"
            "1) Output ONLY a unified diff patch (git style). No explanations.\n"
            "2) Modify ONLY files under src/.\n"
            "3) Fix only the specific bug causing test failures.\n"
            "4) Keep changes minimal - do not refactor unnecessarily.\n"
            "5) Preserve existing functionality that works.\n"
            "6) Ensure type hints are correct.\n"
            "7) Patch must be a valid unified diff (diff --git ... @@ ...; hunk lines must start with +, -, or space).\n"
        )

        user = (
            f"Task id: {task_id}\n"
            f"Target source file: {target_source_relpath}\n\n"
            "=== CURRENT SOURCE CODE ===\n"
            f"{existing_code}\n\n"
            "=== EXISTING TESTS (for context) ===\n"
            f"{existing_tests}\n\n"
            "=== TEST FAILURE OUTPUT ===\n"
            f"{test_failure_output}\n\n"
            "Fix the bug in the source code so tests pass. Output ONLY the patch."
        )

        patch = client.chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
        )

        if not patch.strip():
            return CodeAgentResult(False, "", "Model returned empty output")

        if not _restrict_patch_to_src(patch):
            return CodeAgentResult(False, patch, "Patch touched non-src/ files (blocked)")

        if "diff --git" not in patch:
            return CodeAgentResult(False, patch, "Output is not a git unified diff patch")

        return CodeAgentResult(True, patch, "Bug fix patch proposed")

    # Invalid mode
    return CodeAgentResult(False, "", f"Invalid mode: {mode}. Must be 'fix' or 'generate'")
