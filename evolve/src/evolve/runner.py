"""Running a genome against the task suite through the TypeScript harness."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .genome import path_of


def read_suite(path: Path) -> list[dict[str, Any]]:
    tasks = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for task in tasks:
        missing = {"id", "task", "check"} - task.keys()
        if missing:
            raise ValueError(f"suite task {task.get('id', '?')} is missing {', '.join(sorted(missing))}")
    return tasks


def evaluate(
    root: Path, version: int, suite: list[dict[str, Any]], repeats: int, episodes: Path, offline: bool
) -> list[str]:
    """Run every suite task `repeats` times in a fresh directory. Returns the cycle ids that ran."""
    ran = []
    for task in suite:
        for repeat in range(repeats):
            cycle_id = f"eval-v{version:04d}-{task['id']}-{repeat}"
            work = Path(tempfile.mkdtemp(prefix=f"ooda-{task['id']}-"))
            try:
                for name, text in task.get("files", {}).items():
                    target = (work / name).resolve()
                    if not target.is_relative_to(work.resolve()):
                        raise ValueError(f"suite task {task['id']} writes outside its directory: {name}")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(text)
                command = [
                    "node",
                    str(root / "harness/src/cli.ts"),
                    "--genome", str(path_of(root, version)),
                    # A store per evaluation run keeps cycles independent of each other.
                    "--store", str(work / ".ooda.sqlite"),
                    "--episodes", str(episodes),
                    "--cwd", str(work),
                    "--id", cycle_id,
                    "--check", task["check"],
                    *[arg for criterion in task.get("criteria", []) for arg in ("--criterion", criterion)],
                    *(["--offline"] if offline else []),
                    task["task"],
                ]
                result = subprocess.run(command, capture_output=True, text=True, timeout=1800)
                print(result.stdout.strip() or result.stderr.strip())
                ran.append(cycle_id)
            finally:
                shutil.rmtree(work, ignore_errors=True)
    return ran
