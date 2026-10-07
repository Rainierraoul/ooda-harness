"""Loading, checking and saving genomes.

A genome is the JSON file of settings the harness reads at start-up: strategy
prompts, thresholds and fusion weights. The checks here are the same as in
harness/src/genome.ts. Keep the two in step.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

DOMAINS = ("clear", "complicated", "complex", "chaotic", "confused")
THINKING = ("minimal", "low", "medium", "high")
PROMPT_LIMIT = 2000


class GenomeError(ValueError):
    pass


def find_root(start: Path | None = None) -> Path:
    """The project root: the nearest directory upward that has genome/CURRENT."""
    here = (start or Path.cwd()).resolve()
    for directory in (here, *here.parents):
        if (directory / "genome" / "CURRENT").is_file():
            return directory
    raise GenomeError(f"no genome/CURRENT found above {here}")


def validate(genome: dict[str, Any]) -> dict[str, Any]:
    problems: list[str] = []

    def unit(path: str, value: Any) -> None:
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 <= value <= 1:
            problems.append(f"{path} must be a number from 0 to 1")

    def text(path: str, value: Any) -> None:
        if not isinstance(value, str) or not value.strip() or len(value) > PROMPT_LIMIT:
            problems.append(f"{path} must be non-empty text of at most {PROMPT_LIMIT} characters")

    if not isinstance(genome, dict):
        raise GenomeError("genome must be an object")
    if not isinstance(genome.get("version"), int) or genome["version"] < 1:
        problems.append("version must be a positive integer")
    fusion = genome.get("fusion") or {}
    unit("fusion.decay", fusion.get("decay"))
    unit("fusion.confusionEntropy", fusion.get("confusionEntropy"))
    weight = fusion.get("jevWeight")
    if not isinstance(weight, (int, float)) or not 0 < weight <= 5:
        problems.append("fusion.jevWeight must be above 0 and at most 5")
    cynefin = genome.get("cynefin") or {}
    text("cynefin.instructions", cynefin.get("instructions"))
    strategies = genome.get("strategies") or {}
    for domain in DOMAINS:
        text(f"cynefin.criteria.{domain}", (cynefin.get("criteria") or {}).get(domain))
        strategy = strategies.get(domain)
        if strategy is None:
            problems.append(f"strategies.{domain} is missing")
            continue
        if strategy.get("thinking") not in THINKING:
            problems.append(f"strategies.{domain}.thinking must be one of {', '.join(THINKING)}")
        unit(f"strategies.{domain}.riskThreshold", strategy.get("riskThreshold"))
        text(f"strategies.{domain}.prompt", strategy.get("prompt"))
    gates = genome.get("gates") or {}
    unit("gates.doneThreshold", gates.get("doneThreshold"))
    runs = gates.get("maxRuns")
    if not isinstance(runs, int) or not 1 <= runs <= 10:
        problems.append("gates.maxRuns must be an integer from 1 to 10")
    text("gates.riskInstructions", gates.get("riskInstructions"))
    text("gates.criterionInstructions", gates.get("criterionInstructions"))
    if problems:
        raise GenomeError("invalid genome:\n  " + "\n  ".join(problems))
    return genome


def path_of(root: Path, version: int) -> Path:
    return root / "genome" / f"v{version:04d}.json"


def load(path: Path) -> dict[str, Any]:
    return validate(json.loads(path.read_text()))


def current_version(root: Path) -> int:
    name = (root / "genome" / "CURRENT").read_text().strip()
    match = re.search(r"v(\d+)\.json$", name)
    if match is None:
        raise GenomeError(f"genome/CURRENT names {name!r}, which is not a vNNNN.json file")
    return int(match.group(1))


def load_version(root: Path, version: int) -> dict[str, Any]:
    return load(path_of(root, version))


def next_version(root: Path) -> int:
    versions = [int(p.stem[1:]) for p in (root / "genome").glob("v[0-9][0-9][0-9][0-9].json")]
    return max(versions, default=0) + 1


def derive(parent: dict[str, Any], version: int, note: str) -> dict[str, Any]:
    child = copy.deepcopy(parent)
    child["version"] = version
    child["parent"] = parent["version"]
    child["note"] = note
    return child


def save(root: Path, genome: dict[str, Any]) -> Path:
    validate(genome)
    path = path_of(root, genome["version"])
    if path.exists():
        raise GenomeError(f"{path} already exists; genome versions are never overwritten")
    path.write_text(json.dumps(genome, indent=2, ensure_ascii=False) + "\n")
    return path


def promote(root: Path, version: int) -> None:
    load_version(root, version)
    (root / "genome" / "CURRENT").write_text(f"genome/v{version:04d}.json\n")


def get_gene(genome: dict[str, Any], gene: str) -> Any:
    value: Any = genome
    for part in gene.split("."):
        value = value[part]
    return value


def set_gene(genome: dict[str, Any], gene: str, value: Any) -> None:
    *parents, last = gene.split(".")
    target = genome
    for part in parents:
        target = target[part]
    if last not in target:
        raise GenomeError(f"{gene} is not a gene of this genome")
    target[last] = value
