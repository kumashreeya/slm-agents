from __future__ import annotations

import argparse
import difflib
import json
import platform
import re
import shutil
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from agent.code_agent import propose_code_patch
from agent.mutmut_metrics import read_mutmut_meta
from agent.patch_utils import apply_unified_diff, extract_rewrite_file
from agent.quality_agent import propose_quality_patch
from agent.test_agent import propose_test_patch

# --- paths ---
REPO_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_DIR / "configs"
RESULTS_DIR = REPO_DIR / "results" / "runs"


@dataclass
class ToolEvent:
    name: str
    cmd: list[str]
    returncode: int
    seconds: float
    stdout: str
    stderr: str


@dataclass
class RunLog:
    task_id: str
    workflow: str
    run_id: str
    started_at_utc: str
    finished_at_utc: str
    status: str
    tool_events: list[ToolEvent]
    environment: dict[str, Any]
    notes: dict[str, Any]


def utc_iso(ts: float | None = None) -> str:
    if ts is None:
        ts = time.time()
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))


def run_cmd(name: str, cmd: list[str], cwd: Path, timeout_s: int = 600) -> ToolEvent:
    start = time.time()
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout_s)
    end = time.time()
    return ToolEvent(
        name=name,
        cmd=cmd,
        returncode=p.returncode,
        seconds=round(end - start, 3),
        stdout=p.stdout or "",
        stderr=p.stderr or "",
    )


def run_pytest(repo: Path) -> ToolEvent:
    return run_cmd(name="pytest", cmd=["python", "-m", "pytest", "-q"], cwd=repo)


def autoformat(repo: Path) -> list[ToolEvent]:
    cmds = [
        ("ruff_fix", ["python", "-m", "ruff", "check", ".", "--fix"]),
        ("black", ["python", "-m", "black", "."]),
    ]
    events: list[ToolEvent] = []
    for name, cmd in cmds:
        events.append(run_cmd(name=name, cmd=cmd, cwd=repo))
    return events


def gates_quality(repo: Path) -> list[ToolEvent]:
    cmds = [
        ("ruff", ["python", "-m", "ruff", "check", "."]),
        ("black_check", ["python", "-m", "black", "--check", "."]),
        ("mypy", ["python", "-m", "mypy", "src"]),
        ("bandit", ["python", "-m", "bandit", "-r", "src", "-q"]),
        ("pip_audit", ["python", "-m", "pip_audit"]),
    ]
    events: list[ToolEvent] = []
    for name, cmd in cmds:
        events.append(run_cmd(name=name, cmd=cmd, cwd=repo))
    return events


def safe_run_text(cmd: list[str], cwd: Path) -> str:
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=30)
        out = (p.stdout or "").strip()
        err = (p.stderr or "").strip()
        return out if out else err
    except Exception:
        return ""


def build_environment_snapshot(repo: Path) -> dict[str, Any]:
    return {
        "python": safe_run_text(["python", "--version"], repo),
        "platform": platform.platform(),
        "ruff": safe_run_text(["ruff", "--version"], repo),
        "black": safe_run_text(["black", "--version"], repo),
        "mypy": safe_run_text(["mypy", "--version"], repo),
        "bandit": safe_run_text(["bandit", "--version"], repo),
        "pip_audit": safe_run_text(["pip-audit", "--version"], repo),
        "git_commit": safe_run_text(["git", "rev-parse", "HEAD"], repo),
        "git_branch": safe_run_text(["git", "rev-parse", "--abbrev-ref", "HEAD"], repo),
    }


def load_workflow(workflow_name: str) -> dict[str, Any]:
    data = yaml.safe_load((CONFIG_DIR / "workflows.yaml").read_text(encoding="utf-8"))
    return data["workflows"][workflow_name]


def load_task(task_id: str) -> dict[str, Any]:
    tasks = json.loads((CONFIG_DIR / "tasks.json").read_text(encoding="utf-8"))
    return tasks[task_id]


def failing_summary(events: list[ToolEvent]) -> str:
    parts: list[str] = []
    for e in events:
        if e.returncode != 0:
            out = (e.stdout or "").strip()
            err = (e.stderr or "").strip()
            parts.append(
                f"[{e.name}] returncode={e.returncode}\nSTDOUT:\n{out}\n\nSTDERR:\n{err}\n"
            )
    return "\n---\n".join(parts)


def extract_rewrite_file_content(text: str, expected_relpath: str) -> tuple[bool, str, str]:
    """
    Parse rewrite-mode output:

    ### BEGIN FILE: <expected_relpath>
    <full file content>
    ### END FILE
    """
    begin = f"### BEGIN FILE: {expected_relpath}"
    end = "### END FILE"

    if begin not in text:
        return False, "", f"BEGIN marker not found: {begin}"
    if end not in text:
        return False, "", "END marker not found"

    if text.count("### BEGIN FILE:") != 1:
        return (
            False,
            "",
            f"expected exactly 1 BEGIN FILE marker, found {text.count('### BEGIN FILE:')}",
        )
    if text.count(end) != 1:
        return False, "", f"expected exactly 1 END FILE marker, found {text.count(end)}"

    begin_idx = text.find(begin)
    begin_line_end = text.find("\n", begin_idx)
    if begin_line_end == -1:
        return False, "", "BEGIN marker line has no newline"

    end_idx = text.find(end, begin_line_end + 1)
    if end_idx == -1:
        return False, "", "END marker not found after BEGIN marker"

    content = text[begin_line_end + 1 : end_idx]

    if content.startswith("\n"):
        content = content[1:]

    content = content.replace("\r\n", "\n").replace("\r", "\n")
    content = content.rstrip("\n") + "\n"

    if not content.strip():
        return False, "", "extracted content is empty"

    return True, content, "rewrite content parsed"


def build_patch_from_text(relpath: str, before_text: str, after_text: str) -> str:
    """
    Build a git-style unified diff patch between BEFORE and AFTER text.

    IMPORTANT: This compares the *two strings*, not git index vs working tree.
    That avoids 'patch does not apply' when the working tree is already modified.
    """
    before = (before_text or "").replace("\r\n", "\n").replace("\r", "\n")
    after = (after_text or "").replace("\r\n", "\n").replace("\r", "\n")

    # normalize: ensure final newline
    if before and not before.endswith("\n"):
        before += "\n"
    if after and not after.endswith("\n"):
        after += "\n"

    if before == after:
        return ""

    diff_lines = list(
        difflib.unified_diff(
            before.splitlines(),
            after.splitlines(),
            fromfile=f"a/{relpath}",
            tofile=f"b/{relpath}",
            lineterm="",
            n=3,
        )
    )

    body = "\n".join(diff_lines) + "\n"
    return f"diff --git a/{relpath} b/{relpath}\n{body}"


def parse_coverage_report(report_text: str, entry_rel: str) -> dict[str, Any]:
    total_pct: float | None = None
    entry_line: str | None = None
    entry_missing: str = ""

    lines = report_text.splitlines()
    for line in lines:
        if not line.strip():
            continue
        if line.startswith("Name") or line.startswith("---"):
            continue

        if line.startswith("TOTAL"):
            m = re.search(r"(\d+)%", line)
            if m:
                total_pct = float(m.group(1))
            continue

        m = re.search(r"^(\S+)\s+\d+\s+\d+\s+(\d+)%\s*(.*)$", line)
        if not m:
            continue

        name = m.group(1)
        missing = (m.group(3) or "").strip()

        if name == entry_rel or name.endswith(entry_rel):
            entry_line = line
            entry_missing = missing

    return {
        "total_coverage_pct": total_pct,
        "entry_rel": entry_rel,
        "entry_file_line": entry_line,
        "entry_missing": entry_missing,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Orchestrator: pytest loop (TestAgent) + quality loop (QualityAgent) + JSON logging"
    )
    parser.add_argument("--task", required=True, help="Task id from configs/tasks.json")
    parser.add_argument(
        "--workflow", default="iterative_loop", help="Workflow from configs/workflows.yaml"
    )
    parser.add_argument("--model", default="llama3.2:latest", help="Ollama model name (local)")
    args = parser.parse_args()

    task = load_task(args.task)
    workflow = load_workflow(args.workflow)

    entrypoints = task.get("entrypoints", [])
    entry_rel = entrypoints[0] if entrypoints else "src/example_pkg/math_utils.py"

    run_id = uuid.uuid4().hex[:10]
    started = time.time()
    tool_events: list[ToolEvent] = []

    max_iter = int(workflow.get("budget", {}).get("max_iterations", 3))

    # -----------------------------
    # Phase 0: Code Generation (CodeAgent) - NEW!
    # -----------------------------
    code_mode = task.get("mode", "")
    feature_description = task.get("feature_description", "")

    if code_mode == "generate" and feature_description:
        # Generate new code from description
        cr = propose_code_patch(
            repo_dir=REPO_DIR,
            task_id=args.task,
            target_source_relpath=entry_rel,
            model=args.model,
            mode="generate",
            feature_description=feature_description,
        )

        if not cr.ok:
            tool_events.append(
                ToolEvent(
                    name="code_agent_generate_error",
                    cmd=["code_agent", "generate"],
                    returncode=1,
                    seconds=0.0,
                    stdout=cr.message,
                    stderr=(cr.patch or "")[:1500],
                )
            )
        else:
            # Extract generated code from markers
            content, errors = extract_rewrite_file(cr.patch, entry_rel)

            if content is None:
                tool_events.append(
                    ToolEvent(
                        name="code_agent_parse_error",
                        cmd=["parse_generated_code"],
                        returncode=1,
                        seconds=0.0,
                        stdout="; ".join(errors),
                        stderr=cr.patch[:1500],
                    )
                )
            else:
                # Write generated code to file
                target_path = REPO_DIR / entry_rel
                target_path.parent.mkdir(parents=True, exist_ok=True)
                target_path.write_text(content, encoding="utf-8")

                tool_events.append(
                    ToolEvent(
                        name="code_agent_generate_success",
                        cmd=["code_agent", "generate", entry_rel],
                        returncode=0,
                        seconds=0.0,
                        stdout=f"Generated {len(content)} bytes to {entry_rel}",
                        stderr="",
                    )
                )

                # Autoformat generated code
                tool_events.extend(autoformat(REPO_DIR))

    # -----------------------------
    # Phase 1: pytest loop (TestAgent)
    # -----------------------------
    pytest_ok = False
    last_pytest_text = ""

    for i in range(max_iter):
        ev = run_pytest(REPO_DIR)
        tool_events.append(ev)

        last_pytest_text = (ev.stdout or "") + "\n" + (ev.stderr or "")
        if ev.returncode == 0:
            pytest_ok = True
            break

        tr = propose_test_patch(
            repo_dir=REPO_DIR,
            task_id=args.task,
            target_source_relpath=entry_rel,
            pytest_output=last_pytest_text,
            model=args.model,
        )

        if not tr.ok:
            tool_events.append(
                ToolEvent(
                    name="test_agent_error",
                    cmd=["test_agent"],
                    returncode=1,
                    seconds=0.0,
                    stdout=tr.message,
                    stderr=(tr.patch or "")[:1500],
                )
            )
            break

        pr = apply_unified_diff(REPO_DIR, tr.patch)
        tool_events.append(
            ToolEvent(
                name="apply_test_patch",
                cmd=["git", "apply"],
                returncode=0 if pr.ok else 1,
                seconds=0.0,
                stdout=pr.message,
                stderr="" if pr.ok else (tr.patch or "")[:1500],
            )
        )
        if not pr.ok:
            break

        tool_events.extend(autoformat(REPO_DIR))

        tool_events.append(
            ToolEvent(
                name="pytest_iteration",
                cmd=["loop", str(i + 1), "of", str(max_iter)],
                returncode=0,
                seconds=0.0,
                stdout="",
                stderr="",
            )
        )

    # -----------------------------
    # Week 3A/3B: Coverage measurement + coverage-guided strengthening
    # -----------------------------
    enable_coverage = bool(workflow.get("enable_coverage", False))
    coverage_target = float(workflow.get("coverage_target_pct", 95.0))
    coverage_strengthen = bool(workflow.get("coverage_strengthen", False))
    coverage_max_iter = int(workflow.get("coverage_max_iter", 0))
    coverage_test_target = str(workflow.get("coverage_test_target", "tests/test_math_utils.py"))

    cov_meta: dict[str, Any] | None = None

    if pytest_ok and enable_coverage:
        tool_events.append(
            run_cmd("coverage_erase", ["python", "-m", "coverage", "erase"], cwd=REPO_DIR)
        )

        cov_run = run_cmd(
            name="coverage_pytest",
            cmd=["python", "-m", "coverage", "run", "-m", "pytest", "-q"],
            cwd=REPO_DIR,
        )
        tool_events.append(cov_run)

        cov_rep = run_cmd(
            name="coverage_report",
            cmd=["python", "-m", "coverage", "report", "-m"],
            cwd=REPO_DIR,
        )
        tool_events.append(cov_rep)

        cov_meta = parse_coverage_report(cov_rep.stdout or "", entry_rel)
        tool_events.append(
            ToolEvent(
                name="coverage_meta",
                cmd=["coverage", "meta"],
                returncode=0 if cov_meta.get("total_coverage_pct") is not None else 1,
                seconds=0.0,
                stdout=json.dumps(cov_meta, sort_keys=True),
                stderr="",
            )
        )

    if pytest_ok and enable_coverage and coverage_strengthen and coverage_max_iter > 0 and cov_meta:
        for ci in range(coverage_max_iter):
            total_pct = cov_meta.get("total_coverage_pct")
            missing = (cov_meta.get("entry_missing") or "").strip()

            if total_pct is None:
                break
            if float(total_pct) >= coverage_target:
                break
            if not missing:
                break

            coverage_brief = (
                f"Coverage is below target.\n"
                f"Current TOTAL coverage: {total_pct}%\n"
                f"Target coverage: {coverage_target}%\n\n"
                f"Uncovered lines for {entry_rel} (from coverage report): {missing}\n\n"
                "Please strengthen tests to execute these missing lines.\n"
                "REWRITE MODE OUTPUT REQUIREMENTS:\n"
                f"- Return ONLY the complete contents of {coverage_test_target}\n"
                f"- Use EXACT markers:\n"
                f"  ### BEGIN FILE: {coverage_test_target}\n"
                f"  <complete file content>\n"
                f"  ### END FILE\n"
                "- Do NOT output a diff.\n"
                "- Do NOT create new files.\n"
            )

            tr = propose_test_patch(
                repo_dir=REPO_DIR,
                task_id=args.task,
                target_source_relpath=entry_rel,
                pytest_output=coverage_brief,
                model=args.model,
                output_format="rewrite",
                target_test_relpath=coverage_test_target,
            )

            if not tr.ok:
                tool_events.append(
                    ToolEvent(
                        name="test_agent_coverage_error",
                        cmd=["test_agent", "coverage_strengthen"],
                        returncode=1,
                        seconds=0.0,
                        stdout=tr.message,
                        stderr=(tr.patch or "")[:1500],
                    )
                )
                break

            ok_parse, new_content, parse_msg = extract_rewrite_file_content(
                tr.patch, coverage_test_target
            )
            tool_events.append(
                ToolEvent(
                    name="coverage_rewrite_parse",
                    cmd=["parse_rewrite", coverage_test_target],
                    returncode=0 if ok_parse else 1,
                    seconds=0.0,
                    stdout=parse_msg,
                    stderr="" if ok_parse else (tr.patch or "")[:1500],
                )
            )
            if not ok_parse:
                break

            target_path = REPO_DIR / coverage_test_target
            before_text = target_path.read_text(encoding="utf-8") if target_path.exists() else ""
            patch_text = build_patch_from_text(coverage_test_target, before_text, new_content)

            tool_events.append(
                ToolEvent(
                    name="coverage_deterministic_diff",
                    cmd=["build_patch_from_text", coverage_test_target],
                    returncode=0 if patch_text.strip() else 1,
                    seconds=0.0,
                    stdout=patch_text[:2000],
                    stderr="" if patch_text.strip() else "no diff produced",
                )
            )

            if not patch_text.strip():
                break

            pr = apply_unified_diff(REPO_DIR, patch_text)
            tool_events.append(
                ToolEvent(
                    name="apply_coverage_test_patch",
                    cmd=["git", "apply"],
                    returncode=0 if pr.ok else 1,
                    seconds=0.0,
                    stdout=pr.message,
                    stderr="" if pr.ok else patch_text[:1500],
                )
            )
            if not pr.ok:
                break

            tool_events.extend(autoformat(REPO_DIR))

            ev = run_pytest(REPO_DIR)
            tool_events.append(ev)
            if ev.returncode != 0:
                break

            tool_events.append(
                run_cmd("coverage_erase", ["python", "-m", "coverage", "erase"], cwd=REPO_DIR)
            )
            cov_run2 = run_cmd(
                name="coverage_pytest",
                cmd=["python", "-m", "coverage", "run", "-m", "pytest", "-q"],
                cwd=REPO_DIR,
            )
            tool_events.append(cov_run2)

            cov_rep2 = run_cmd(
                name="coverage_report",
                cmd=["python", "-m", "coverage", "report", "-m"],
                cwd=REPO_DIR,
            )
            tool_events.append(cov_rep2)

            cov_meta = parse_coverage_report(cov_rep2.stdout or "", entry_rel)
            tool_events.append(
                ToolEvent(
                    name="coverage_meta",
                    cmd=["coverage", "meta"],
                    returncode=0 if cov_meta.get("total_coverage_pct") is not None else 1,
                    seconds=0.0,
                    stdout=json.dumps(cov_meta, sort_keys=True),
                    stderr="",
                )
            )

            tool_events.append(
                ToolEvent(
                    name="coverage_iteration",
                    cmd=["loop", str(ci + 1), "of", str(coverage_max_iter)],
                    returncode=0,
                    seconds=0.0,
                    stdout="",
                    stderr="",
                )
            )

    # -----------------------------
    # Week 3C: Mutation + mutation-guided strengthening (rewrite mode)
    # -----------------------------
    enable_mutation = bool(workflow.get("enable_mutation")) or ("week3" in args.workflow)
    mut_scope = workflow.get("mutation_scope") or "example_pkg.math_utils*"

    if pytest_ok and enable_mutation:
        shutil.rmtree(REPO_DIR / "mutants", ignore_errors=True)
        t0 = time.time()
        proc = subprocess.run(
            ["mutmut", "run", mut_scope],
            cwd=str(REPO_DIR),
            capture_output=True,
            text=True,
        )
        tool_events.append(
            ToolEvent(
                name="mutmut_run",
                cmd=["mutmut", "run", mut_scope],
                returncode=proc.returncode,
                seconds=round(time.time() - t0, 3),
                stdout=proc.stdout,
                stderr=proc.stderr,
            )
        )

    mut_meta_result = None
    try:
        mut_meta_result = read_mutmut_meta(REPO_DIR)
        tool_events.append(
            ToolEvent(
                name="mutmut_meta",
                cmd=["mutmut", "meta"],
                returncode=0,
                seconds=0.0,
                stdout=json.dumps(mut_meta_result, sort_keys=True),
                stderr="",
            )
        )
    except Exception as e:
        tool_events.append(
            ToolEvent(
                name="mutmut_meta",
                cmd=["mutmut", "meta"],
                returncode=1,
                seconds=0.0,
                stdout="",
                stderr=str(e),
            )
        )

    mutation_strengthen = bool(workflow.get("mutation_strengthen", False))
    mutation_max_iter = int(workflow.get("mutation_max_iter", 0))
    mutation_max_mutants = int(workflow.get("mutation_max_mutants_per_iter", 3))
    mutation_target = float(workflow.get("mutation_target_score", 1.0))
    test_target = "tests/test_math_utils.py"

    if (
        pytest_ok
        and enable_mutation
        and mutation_strengthen
        and mutation_max_iter > 0
        and mut_meta_result
    ):
        for mi in range(mutation_max_iter):
            score = mut_meta_result.get("mutation_score")
            survivors = list(mut_meta_result.get("survivors") or [])

            if score is None:
                break
            if float(score) >= mutation_target:
                break
            if not survivors:
                break

            target_ids = survivors[:mutation_max_mutants]

            diffs: list[str] = []
            for mid in target_ids:
                t0 = time.time()
                show = subprocess.run(
                    ["mutmut", "show", mid],
                    cwd=str(REPO_DIR),
                    capture_output=True,
                    text=True,
                )
                tool_events.append(
                    ToolEvent(
                        name="mutmut_show",
                        cmd=["mutmut", "show", mid],
                        returncode=show.returncode,
                        seconds=round(time.time() - t0, 3),
                        stdout=show.stdout,
                        stderr=show.stderr,
                    )
                )
                diffs.append(show.stdout or show.stderr or "")

            mutation_brief = (
                "Mutation testing found surviving mutants.\n"
                "Strengthen tests to kill these mutants.\n"
                "OUTPUT REQUIREMENTS (REWRITE MODE):\n"
                f"- Return ONLY the complete contents of {test_target}\n"
                f"- Use EXACT markers:\n"
                f"  ### BEGIN FILE: {test_target}\n"
                f"  <complete file content>\n"
                f"  ### END FILE\n"
                "- Do NOT output a diff.\n"
                "- Do NOT create new files.\n\n"
                "Surviving mutant diffs (diagnostic):\n\n" + "\n\n".join(diffs)
            )

            tr = propose_test_patch(
                repo_dir=REPO_DIR,
                task_id=args.task,
                target_source_relpath=entry_rel,
                pytest_output=mutation_brief,
                model=args.model,
                output_format="rewrite",
                target_test_relpath=test_target,
            )

            if not tr.ok:
                tool_events.append(
                    ToolEvent(
                        name="test_agent_mutation_error",
                        cmd=["test_agent", "mutation_strengthen", "rewrite_mode"],
                        returncode=1,
                        seconds=0.0,
                        stdout=tr.message,
                        stderr=(tr.patch or "")[:1500],
                    )
                )
                break

            ok_parse, new_content, parse_msg = extract_rewrite_file_content(tr.patch, test_target)
            tool_events.append(
                ToolEvent(
                    name="mutation_rewrite_parse",
                    cmd=["parse_rewrite", test_target],
                    returncode=0 if ok_parse else 1,
                    seconds=0.0,
                    stdout=parse_msg,
                    stderr="" if ok_parse else (tr.patch or "")[:1500],
                )
            )
            if not ok_parse:
                break

            target_path = REPO_DIR / test_target
            before_text = target_path.read_text(encoding="utf-8") if target_path.exists() else ""
            patch_text = build_patch_from_text(test_target, before_text, new_content)

            tool_events.append(
                ToolEvent(
                    name="mutation_deterministic_diff",
                    cmd=["build_patch_from_text", test_target],
                    returncode=0 if patch_text.strip() else 1,
                    seconds=0.0,
                    stdout=patch_text[:2000],
                    stderr="" if patch_text.strip() else "no diff produced",
                )
            )

            if not patch_text.strip():
                break

            pr = apply_unified_diff(REPO_DIR, patch_text)
            tool_events.append(
                ToolEvent(
                    name="apply_mutation_test_patch",
                    cmd=["git", "apply"],
                    returncode=0 if pr.ok else 1,
                    seconds=0.0,
                    stdout=pr.message,
                    stderr="" if pr.ok else patch_text[:1500],
                )
            )
            if not pr.ok:
                break

            tool_events.extend(autoformat(REPO_DIR))

            ev = run_pytest(REPO_DIR)
            tool_events.append(ev)
            if ev.returncode != 0:
                break

            shutil.rmtree(REPO_DIR / "mutants", ignore_errors=True)
            t0 = time.time()
            proc = subprocess.run(
                ["mutmut", "run", mut_scope],
                cwd=str(REPO_DIR),
                capture_output=True,
                text=True,
            )
            tool_events.append(
                ToolEvent(
                    name="mutmut_run",
                    cmd=["mutmut", "run", mut_scope],
                    returncode=proc.returncode,
                    seconds=round(time.time() - t0, 3),
                    stdout=proc.stdout,
                    stderr=proc.stderr,
                )
            )

            try:
                mut_meta_result = read_mutmut_meta(REPO_DIR)
                tool_events.append(
                    ToolEvent(
                        name="mutmut_meta",
                        cmd=["mutmut", "meta"],
                        returncode=0,
                        seconds=0.0,
                        stdout=json.dumps(mut_meta_result, sort_keys=True),
                        stderr="",
                    )
                )
            except Exception as e:
                tool_events.append(
                    ToolEvent(
                        name="mutmut_meta",
                        cmd=["mutmut", "meta"],
                        returncode=1,
                        seconds=0.0,
                        stdout="",
                        stderr=str(e),
                    )
                )
                break

            tool_events.append(
                ToolEvent(
                    name="mutation_iteration",
                    cmd=["loop", str(mi + 1), "of", str(mutation_max_iter)],
                    returncode=0,
                    seconds=0.0,
                    stdout="",
                    stderr="",
                )
            )

    # -----------------------------
    # Phase 2: quality loop (QualityAgent)
    # -----------------------------
    quality_ok = False

    if pytest_ok:
        for i in range(max_iter):
            tool_events.extend(autoformat(REPO_DIR))

            q_events = gates_quality(REPO_DIR)
            tool_events.extend(q_events)

            if all(e.returncode == 0 for e in q_events):
                quality_ok = True
                break

            summary = failing_summary(q_events)

            qr = propose_quality_patch(
                repo_dir=REPO_DIR,
                task_id=args.task,
                entrypoint_relpath=entry_rel,
                failing_tool_output=summary,
                model=args.model,
            )

            if not qr.ok:
                tool_events.append(
                    ToolEvent(
                        name="quality_agent_error",
                        cmd=["quality_agent"],
                        returncode=1,
                        seconds=0.0,
                        stdout=qr.message,
                        stderr=(qr.patch or "")[:1500],
                    )
                )
                break

            pr = apply_unified_diff(REPO_DIR, qr.patch)
            tool_events.append(
                ToolEvent(
                    name="apply_quality_patch",
                    cmd=["git", "apply"],
                    returncode=0 if pr.ok else 1,
                    seconds=0.0,
                    stdout=pr.message,
                    stderr="" if pr.ok else (qr.patch or "")[:1500],
                )
            )
            if not pr.ok:
                break

            tool_events.append(
                ToolEvent(
                    name="quality_iteration",
                    cmd=["loop", str(i + 1), "of", str(max_iter)],
                    returncode=0,
                    seconds=0.0,
                    stdout="",
                    stderr="",
                )
            )

    ok = pytest_ok and quality_ok and all(e.returncode == 0 for e in tool_events)
    status = "PASS" if ok else "FAIL"
    finished = time.time()

    log = RunLog(
        task_id=args.task,
        workflow=args.workflow,
        run_id=run_id,
        started_at_utc=utc_iso(started),
        finished_at_utc=utc_iso(finished),
        status=status,
        tool_events=tool_events,
        environment=build_environment_snapshot(REPO_DIR),
        notes={
            "task_description": task.get("description", ""),
            "workflow_description": workflow.get("description", ""),
            "budget": workflow.get("budget", {}),
            "entrypoint": entry_rel,
            "model": args.model,
        },
    )

    out_dir = RESULTS_DIR / args.task / args.workflow / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run.json").write_text(json.dumps(asdict(log), indent=2), encoding="utf-8")

    print(f"Run {run_id} status: {status}")
    print(f"Saved log: {out_dir / 'run.json'}")

    if not ok:
        print("\n--- Failing tools summary ---")
        for e in tool_events:
            if e.returncode != 0:
                print(f"\n[{e.name}] returncode={e.returncode}")
                if e.stdout.strip():
                    print(e.stdout[:1500])
                if e.stderr.strip():
                    print(e.stderr[:1500])


if __name__ == "__main__":
    main()
