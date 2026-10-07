"""Reading the episode log the harness writes, and scoring it."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


def read(path: Path) -> list[dict[str, Any]]:
    """Episodes in file order. A cycle logged twice (after a crash) keeps its last line."""
    if not path.exists():
        return []
    by_id: dict[str, dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            episode = json.loads(line)
            by_id.pop(episode["id"], None)
            by_id[episode["id"]] = episode
    return list(by_id.values())


def succeeded(episode: dict[str, Any]) -> bool:
    """An objective check wins when there is one. Otherwise fall back to the harness's own verdict.

    The fallback is Jev judging its own cycle's work, so a genome could learn to
    satisfy Jev without doing the task. Evaluation suites should give every task a check.
    """
    verified = episode.get("verified")
    if verified is not None:
        return bool(verified)
    return episode.get("outcome") == "met"


def final_domain(episode: dict[str, Any]) -> str:
    strategies = episode.get("strategies") or []
    return strategies[-1] if strategies else "none"


@dataclass
class Tally:
    successes: int = 0
    failures: int = 0
    runs: int = 0
    tokens: int = 0

    @property
    def total(self) -> int:
        return self.successes + self.failures

    @property
    def rate(self) -> float:
        return self.successes / self.total if self.total else 0.0

    def add(self, episode: dict[str, Any]) -> None:
        if succeeded(episode):
            self.successes += 1
        else:
            self.failures += 1
        self.runs += episode.get("runs", 0)
        tokens = episode.get("tokens") or {}
        self.tokens += tokens.get("input", 0) + tokens.get("output", 0)


def tally(episodes: Iterable[dict[str, Any]], key=lambda e: e["genomeVersion"]) -> dict[Any, Tally]:
    result: dict[Any, Tally] = defaultdict(Tally)
    for episode in episodes:
        result[key(episode)].add(episode)
    return dict(result)


def probability_better(candidate: Tally, incumbent: Tally, samples: int = 20000, seed: int = 0) -> float:
    """Chance that the candidate's true success rate is higher than the incumbent's.

    Each rate gets a Beta(1 + successes, 1 + failures) posterior, which is what a
    flat prior and the observed counts imply. The chance is estimated by sampling.
    """
    rng = random.Random(seed)
    wins = 0
    for _ in range(samples):
        c = rng.betavariate(1 + candidate.successes, 1 + candidate.failures)
        i = rng.betavariate(1 + incumbent.successes, 1 + incumbent.failures)
        wins += c > i
    return wins / samples
