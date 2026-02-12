from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_mutmut_meta(repo_root: Path) -> dict[str, Any]:
    """
    Compute mutation outcomes by reading mutmut's *.py.meta files under mutants/src/**.
    Mutmut stores a mapping: exit_code_by_key = {mutant_id: exit_code}.
      - exit_code 1 => tests FAILED => mutant KILLED
      - exit_code 0 => tests PASSED => mutant SURVIVED
    """
    meta_files = sorted((repo_root / "mutants" / "src").rglob("*.py.meta"))

    all_exit_codes: dict[str, int] = {}
    for mf in meta_files:
        try:
            data = json.loads(mf.read_text(encoding="utf-8"))
        except Exception:
            continue
        m = data.get("exit_code_by_key", {})
        if isinstance(m, dict):
            for k, v in m.items():
                if isinstance(k, str) and isinstance(v, int):
                    all_exit_codes[k] = v

    killed = [mid for mid, code in all_exit_codes.items() if code == 1]
    survived = [mid for mid, code in all_exit_codes.items() if code == 0]
    total = len(all_exit_codes)
    mutation_score = (len(killed) / total) if total else None

    return {
        "meta_files": [str(p) for p in meta_files],
        "total_mutants": total,
        "killed": len(killed),
        "survived": len(survived),
        "survivors": sorted(survived),
        "mutation_score": mutation_score,  # 0..1 or None
    }


if __name__ == "__main__":
    repo = Path(".").resolve()
    print(read_mutmut_meta(repo))
