from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from agent.mutmut_metrics import read_mutmut_meta
from agent.patch_utils import apply_unified_diff
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

    Returns: (ok, content, message)
    Content is normalized to end with exactly one newline.
    """
    begin = f"### BEGIN FILE: {expected_relpath}"
    end = "### END FILE"

    if begin not in text:
        return False, "", f"BEGIN marker not found: {begin}"
    if end not in text:
        return False, "", "END marker not found"

    # Require exactly one BEGIN and one END (keeps parsing unambiguous)
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

    # Common model behavior: adds a blank line right after BEGIN marker
    if content.startswith("\n"):
        content = content[1:]

    # Normalize newlines and ensure exactly one trailing newline
    content = content.replace("\r\n", "\n").replace("\r", "\n")
    content = content.rstrip("\n") + "\n"

    if not content.strip():
        return False, "", "extracted content is empty"

    return True, content, "rewrite content parsed"


def git_diff_file(repo: Path, relpath: str) -> tuple[int, str, str]:
    """
    Return (returncode, stdout, stderr) for: git diff -- <relpath>
    """
    p = subprocess.run(
        ["git", "diff", "--", relpath],
        cwd=str(repo),
        capture_output=True,
        text=True,
    )
    return p.returncode, p.stdout or "", p.stderr or ""


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

    # --- Week 3: run mutmut during the run (when enabled) ---
    enable_mutation = bool(workflow.get("enable_mutation")) or ("week3" in args.workflow)
    mut_scope = workflow.get("mutation_scope") or "example_pkg.math_utils*"

    if pytest_ok and enable_mutation:
        # Clean any previous mutation artifacts so results are fresh per run
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

    # --- Week 3: mutation metrics (paper-ready logging) ---
    # Keep a parsed copy for the strengthen loop, and also log it as a ToolEvent.
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

    # --- Week 3: mutation-guided strengthen loop (TestAgent) ---
    # If mutants survive, ask TestAgent to strengthen tests to kill them.
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

            # Collect mutant diffs to give TestAgent concrete targets
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
                "IMPORTANT: The next block is DIAGNOSTIC output, NOT a patch to apply.\n\n"
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

            # ---- REWRITE MODE CALL (robust) ----
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
                # Retry once with even stricter “no extra text” reminder
                retry_brief = mutation_brief + (
                    "\n\nFORMAT VIOLATION. Retry.\n"
                    "Return ONLY the file using markers. No extra text before/after.\n"
                    f"### BEGIN FILE: {test_target}\n"
                    "<complete file content>\n"
                    "### END FILE\n"
                )
                tr2 = propose_test_patch(
                    repo_dir=REPO_DIR,
                    task_id=args.task,
                    target_source_relpath=entry_rel,
                    pytest_output=retry_brief,
                    model=args.model,
                    output_format="rewrite",
                    target_test_relpath=test_target,
                )
                if tr2.ok:
                    tr = tr2
                    tool_events.append(
                        ToolEvent(
                            name="test_agent_mutation_retry",
                            cmd=["test_agent", "mutation_strengthen", "rewrite_retry1"],
                            returncode=0,
                            seconds=0.0,
                            stdout="rewrite retry succeeded",
                            stderr="",
                        )
                    )
                else:
                    tool_events.append(
                        ToolEvent(
                            name="test_agent_mutation_error",
                            cmd=["test_agent", "mutation_strengthen", "rewrite_mode"],
                            returncode=1,
                            seconds=0.0,
                            stdout=tr2.message,
                            stderr=(tr2.patch or "")[:1500],
                        )
                    )
                    break

            # Parse rewrite content
            t0 = time.time()
            ok_parse, new_content, parse_msg = extract_rewrite_file_content(tr.patch, test_target)
            tool_events.append(
                ToolEvent(
                    name="mutation_rewrite_parse",
                    cmd=["parse_rewrite", test_target],
                    returncode=0 if ok_parse else 1,
                    seconds=round(time.time() - t0, 3),
                    stdout=parse_msg,
                    stderr="" if ok_parse else (tr.patch or "")[:1500],
                )
            )
            if not ok_parse:
                break

            # Deterministic diff generation + apply via git apply
            target_path = REPO_DIR / test_target
            before_text = target_path.read_text(encoding="utf-8") if target_path.exists() else ""

            # Write proposed content to working tree to let git compute a correct diff
            target_path.write_text(new_content, encoding="utf-8", newline="\n")

            rc, diff_out, diff_err = git_diff_file(REPO_DIR, test_target)
            tool_events.append(
                ToolEvent(
                    name="mutation_deterministic_diff",
                    cmd=["git", "diff", "--", test_target],
                    returncode=0 if (rc == 0 and diff_out.strip()) else 1,
                    seconds=0.0,
                    stdout=diff_out if diff_out else "",
                    stderr=(
                        diff_err
                        if diff_err
                        else ("no diff produced" if not diff_out.strip() else "")
                    ),
                )
            )

            # Restore before state BEFORE applying via apply_unified_diff (keeps your “apply patch” design)
            target_path.write_text(before_text, encoding="utf-8", newline="\n")

            if not diff_out.strip():
                # Model didn’t change the file meaningfully
                tool_events.append(
                    ToolEvent(
                        name="mutation_rewrite_no_changes",
                        cmd=["rewrite_mode", "no_changes"],
                        returncode=1,
                        seconds=0.0,
                        stdout="rewrite produced no diff; stopping mutation iteration",
                        stderr="",
                    )
                )
                break

            pr = apply_unified_diff(REPO_DIR, diff_out)
            tool_events.append(
                ToolEvent(
                    name="apply_mutation_test_patch",
                    cmd=["git", "apply"],
                    returncode=0 if pr.ok else 1,
                    seconds=0.0,
                    stdout=pr.message,
                    stderr="" if pr.ok else diff_out[:1500],
                )
            )
            if not pr.ok:
                break

            tool_events.extend(autoformat(REPO_DIR))

            # Re-run pytest to ensure we didn't break the suite
            ev = run_pytest(REPO_DIR)
            tool_events.append(ev)
            if ev.returncode != 0:
                break

            # Re-run mutmut fresh to see if we killed survivors
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

            # Refresh meta + log it again
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

    # Phase 2: quality loop (QualityAgent)
    # -----------------------------
    quality_ok = False

    if pytest_ok:
        for i in range(max_iter):
            # First run safe auto-fixers (ruff/black) before checking
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
