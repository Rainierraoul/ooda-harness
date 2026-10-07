import json
import random
import shutil
from pathlib import Path

import pytest

from evolve import cli, mutate
from evolve import episodes as ep
from evolve import genome as gn

PROJECT = Path(__file__).resolve().parents[2]


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """A copy of the project's genome folder, so tests can write new versions."""
    shutil.copytree(PROJECT / "genome", tmp_path / "genome")
    (tmp_path / "genome" / "CURRENT").write_text("genome/v0001.json\n")
    return tmp_path


def episode(id: str, version: int, ok: bool, domain: str = "clear", verified=None) -> dict:
    return {
        "id": id,
        "genomeVersion": version,
        "task": "t",
        "criteria": [{"text": "c", "probability": 0.9 if ok else 0.1}],
        "outcome": "met" if ok else "exhausted",
        "runs": 1,
        "strategies": [domain],
        "orientations": [],
        "tokens": {"input": 10, "output": 5},
        "startedAt": "",
        "endedAt": "",
        **({} if verified is None else {"verified": verified}),
    }


def write_log(path: Path, episodes: list[dict]) -> Path:
    path.write_text("".join(json.dumps(e) + "\n" for e in episodes))
    return path


def test_shipped_genome_is_valid(root: Path):
    assert gn.load_version(root, 1)["version"] == 1


def test_validation_lists_every_problem(root: Path):
    g = gn.load_version(root, 1)
    g["fusion"]["decay"] = 2
    del g["strategies"]["chaotic"]
    with pytest.raises(gn.GenomeError, match=r"(?s)fusion\.decay.*strategies\.chaotic is missing"):
        gn.validate(g)


def test_numeric_mutations_stay_in_bounds(root: Path):
    g = gn.load_version(root, 1)
    rng = random.Random(1)
    for gene, (low, high, _) in mutate.NUMERIC.items():
        for _ in range(50):
            assert low <= mutate.mutate_value(g, gene, rng, []) <= high


def test_genome_versions_are_never_overwritten(root: Path):
    g = gn.derive(gn.load_version(root, 1), 2, "test")
    gn.save(root, g)
    with pytest.raises(gn.GenomeError, match="already exists"):
        gn.save(root, g)


def test_propose_writes_a_valid_child_with_one_gene_changed(root: Path, tmp_path: Path):
    version = cli.propose_child(root, tmp_path / "none.jsonl", random.Random(3), "fusion.decay")
    parent, child = gn.load_version(root, 1), gn.load_version(root, version)
    assert child["parent"] == 1 and child["version"] == 2
    assert child["fusion"]["decay"] != parent["fusion"]["decay"]
    child["fusion"]["decay"] = parent["fusion"]["decay"]
    assert {k: v for k, v in child.items() if k not in ("version", "parent", "note")} == {
        k: v for k, v in parent.items() if k not in ("version", "parent", "note")
    }


def test_objective_check_overrides_the_harness_verdict():
    assert ep.succeeded(episode("a", 1, ok=True, verified=False)) is False
    assert ep.succeeded(episode("b", 1, ok=False, verified=True)) is True
    assert ep.succeeded(episode("c", 1, ok=True)) is True


def test_a_cycle_logged_twice_counts_once(tmp_path: Path):
    log = write_log(tmp_path / "e.jsonl", [episode("x", 1, False), episode("x", 1, True)])
    history = ep.read(log)
    assert len(history) == 1 and ep.succeeded(history[0])


def test_probability_better_tracks_the_evidence():
    strong, weak = ep.Tally(successes=18, failures=2), ep.Tally(successes=8, failures=12)
    assert ep.probability_better(strong, weak) > 0.99
    assert ep.probability_better(weak, strong) < 0.01
    assert 0.3 < ep.probability_better(ep.Tally(1, 1), ep.Tally(1, 1)) < 0.7


def test_promotion_needs_enough_evidence_and_a_clear_win(root: Path, tmp_path: Path):
    gn.save(root, gn.derive(gn.load_version(root, 1), 2, "candidate"))
    few = [episode(f"eval-v0002-t-{i}", 2, True) for i in range(3)] + [
        episode(f"eval-v0001-t-{i}", 1, False) for i in range(3)
    ]
    log = write_log(tmp_path / "few.jsonl", few)
    assert cli.promote_if_better(root, log, 2, 0.9, 10, False) == 1
    assert gn.current_version(root) == 1

    many = [episode(f"eval-v0002-t-{i}", 2, i % 10 != 0) for i in range(20)] + [
        episode(f"eval-v0001-t-{i}", 1, i % 2 == 0) for i in range(20)
    ]
    log = write_log(tmp_path / "many.jsonl", many)
    assert cli.promote_if_better(root, log, 2, 0.9, 10, False) == 0
    assert gn.current_version(root) == 2


def test_gene_choice_targets_the_failing_domain():
    history = [episode(f"e{i}", 1, ok=False, domain="complex") for i in range(5)] + [
        episode(f"f{i}", 1, ok=True, domain="clear") for i in range(5)
    ]
    for seed in range(10):
        assert ".complex." in mutate.pick_gene(history, 1, random.Random(seed))
