"""Command line for the evolution loop.

    evolve stats                      success rates per genome and per Cynefin domain
    evolve propose [--gene G]         write a child of the current genome with one gene changed
    evolve evaluate VERSION           run the task suite with that genome
    evolve promote VERSION            make VERSION current, if the evidence says it is better
    evolve cycle                      propose, evaluate both, promote if better

Nothing here changes code. The harness only ever reads a genome, and a genome
becomes current only through `promote`, which compares evaluation results.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from urllib.error import URLError

from . import episodes as ep
from . import genome as gn
from . import mutate, runner


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evolve", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, help="project root (default: found from the working directory)")
    parser.add_argument("--episodes", type=Path, help="episode log (default: runs/episodes.jsonl)")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("stats")

    propose = commands.add_parser("propose")
    propose.add_argument("--gene", choices=mutate.all_genes())
    propose.add_argument("--seed", type=int)

    for name in ("evaluate", "cycle"):
        command = commands.add_parser(name)
        if name == "evaluate":
            command.add_argument("version", type=int)
        command.add_argument("--suite", type=Path, help="default: tasks/suite.jsonl")
        command.add_argument("--repeats", type=int, default=3)
        command.add_argument("--offline", action="store_true", help="use the offline stand-in models")
        command.add_argument("--confidence", type=float, default=0.9)
        command.add_argument("--seed", type=int)

    promote = commands.add_parser("promote")
    promote.add_argument("version", type=int)
    promote.add_argument("--confidence", type=float, default=0.9)
    promote.add_argument("--min-episodes", type=int, default=10)
    promote.add_argument("--force", action="store_true", help="promote without comparing")

    args = parser.parse_args(argv)
    root = (args.root or gn.find_root()).resolve()
    log = args.episodes or root / "runs" / "episodes.jsonl"

    if args.command == "stats":
        return stats(root, log)
    if args.command == "propose":
        print(propose_child(root, log, random.Random(args.seed), args.gene))
        return 0
    suite = runner.read_suite(args.suite or root / "tasks" / "suite.jsonl")
    if args.command == "evaluate":
        runner.evaluate(root, args.version, suite, args.repeats, log, args.offline)
        return stats(root, log)
    if args.command == "promote":
        return promote_if_better(root, log, args.version, args.confidence, args.min_episodes, args.force)
    # cycle
    incumbent = gn.current_version(root)
    # Offline there is no chat model to rewrite prompts, so only numbers and choices change.
    child = propose_child(root, log, random.Random(args.seed), prompts=not args.offline)
    for version in (incumbent, child):
        if suite_episodes(log, version, suite) < args.repeats * len(suite):
            runner.evaluate(root, version, suite, args.repeats, log, args.offline)
    return promote_if_better(root, log, child, args.confidence, args.repeats * len(suite), False)


def suite_episodes(log: Path, version: int, suite: list[dict]) -> int:
    ids = {t["id"] for t in suite}
    return sum(
        1 for e in ep.read(log) if e.get("genomeVersion") == version and e["id"].split("-", 2)[-1].rsplit("-", 1)[0] in ids
    )


def propose_child(root: Path, log: Path, rng: random.Random, gene: str | None = None, prompts: bool = True) -> int:
    parent = gn.load_version(root, gn.current_version(root))
    try:
        chosen, old, new = mutate.propose(parent, ep.read(log), rng, gene, prompts)
    except URLError as error:
        raise SystemExit(f"could not reach the vLLM server to rewrite a prompt ({error.reason}). "
                         "Start it, or choose a numeric gene with --gene.") from error
    version = gn.next_version(root)
    child = gn.derive(parent, version, f"{chosen}: {summarise(old)} -> {summarise(new)}")
    gn.set_gene(child, chosen, new)
    path = gn.save(root, child)
    print(f"wrote {path.relative_to(root)} ({child['note']})", file=sys.stderr)
    return version


def summarise(value) -> str:
    text = str(value)
    return text if len(text) <= 60 else text[:57] + "..."


def promote_if_better(root: Path, log: Path, version: int, confidence: float, minimum: int, force: bool) -> int:
    incumbent = gn.current_version(root)
    if force:
        gn.promote(root, version)
        print(f"v{version:04d} is now current (forced)")
        return 0
    if version == incumbent:
        print(f"v{version:04d} is already current")
        return 0
    # Compare on evaluation episodes only, so ad-hoc runs do not skew the comparison.
    evals = [e for e in ep.read(log) if e["id"].startswith("eval-")]
    tallies = ep.tally(evals)
    cand, inc = tallies.get(version, ep.Tally()), tallies.get(incumbent, ep.Tally())
    if cand.total < minimum or inc.total < minimum:
        print(f"not enough evidence: v{version:04d} has {cand.total}, v{incumbent:04d} has {inc.total}; need {minimum} each")
        return 1
    p = ep.probability_better(cand, inc)
    print(
        f"v{version:04d} {cand.successes}/{cand.total} vs v{incumbent:04d} {inc.successes}/{inc.total}: "
        f"{p:.0%} chance the candidate is better (need {confidence:.0%})"
    )
    if p < confidence:
        return 1
    gn.promote(root, version)
    print(f"v{version:04d} is now current")
    return 0


def stats(root: Path, log: Path) -> int:
    history = ep.read(log)
    if not history:
        print(f"no episodes in {log}")
        return 0
    current = gn.current_version(root)
    print("genome   episodes  success  mean runs  mean tokens")
    for version, t in sorted(ep.tally(history).items()):
        mark = "*" if version == current else " "
        print(f"{mark}v{version:04d}  {t.total:8d}  {t.rate:7.0%}  {t.runs / t.total:9.1f}  {t.tokens / t.total:11.0f}")
    print("\nby final Cynefin domain, current genome:")
    for domain, t in sorted(ep.tally([e for e in history if e.get("genomeVersion") == current], key=ep.final_domain).items()):
        print(f"  {domain:12s} {t.successes}/{t.total} succeeded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
