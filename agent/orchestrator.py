from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from agent.patch_utils import apply_unified_diff
from agent.test_agent import propose_test_patch

# --- paths ---
REPO_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_DIR / "configs"
RESULTS_DIR = REPO_DIR / "results" / "runs"


# --- data structures ---
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


def safe_run_text(cmd: list[str], cwd: Path) -> str:
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=30)
        out = (p.stdout or "").strip()
        err = (p.stderr or "").strip()
        return out if out else err
    except Exception:
        return ""


def load_workflow(workflow_name: str) -> dict[str, Any]:
    data = yaml.safe_load((CONFIG_DIR / "workflows.yaml").read_text(encoding="utf-8"))
    return data["workflows"][workflow_name]


def load_task(task_id: str) -> dict[str, Any]:
    tasks = json.loads((CONFIG_DIR / "tasks.json").read_text(encoding="utf-8"))
    return tasks[task_id]


def autoformat(repo: Path) -> list[ToolEvent]:
    # Safe auto-fixers (especially useful after LLM edits tests)
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Orchestrator: run pytest loop (with TestAgent) + quality gates + log JSON results"
    )
    parser.add_argument("--task", required=True, help="Task id from configs/tasks.json")
    parser.add_argument(
        "--workflow", default="code_first", help="Workflow name from configs/workflows.yaml"
    )
    parser.add_argument("--model", default="llama3.2:latest", help="Ollama model name (local)")
    args = parser.parse_args()

    task = load_task(args.task)
    workflow = load_workflow(args.workflow)

    entrypoints = task.get("entrypoints", [])
    target_source_relpath = entrypoints[0] if entrypoints else "src/example_pkg/math_utils.py"

    run_id = uuid.uuid4().hex[:10]
    started = time.time()
    tool_events: list[ToolEvent] = []

    max_iter = int(workflow.get("budget", {}).get("max_iterations", 3))

    # --- Phase 1: pytest loop with TestAgent ---
    pytest_ok = False
    last_pytest_text = ""

    for i in range(max_iter):
        ev = run_pytest(REPO_DIR)
        tool_events.append(ev)

        last_pytest_text = (ev.stdout or "") + "\n" + (ev.stderr or "")
        if ev.returncode == 0:
            pytest_ok = True
            break

        # Ask TestAgent to propose a patch for tests/
        tr = propose_test_patch(
            repo_dir=REPO_DIR,
            task_id=args.task,
            target_source_relpath=target_source_relpath,
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

        # Apply patch
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

        # Auto-format after patch (helps future gates)
        tool_events.extend(autoformat(REPO_DIR))

        # Log iteration count as a lightweight event
        tool_events.append(
            ToolEvent(
                name="iteration",
                cmd=["loop", str(i + 1), "of", str(max_iter)],
                returncode=0,
                seconds=0.0,
                stdout="",
                stderr="",
            )
        )

    # --- Phase 2: quality gates ---
    if pytest_ok:
        tool_events.extend(gates_quality(REPO_DIR))

    ok = pytest_ok and all(e.returncode == 0 for e in tool_events)
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
            "target_source_relpath": target_source_relpath,
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
