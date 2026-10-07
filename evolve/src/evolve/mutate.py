"""Proposing one change to a genome.

Numbers move by a small random step and stay inside fixed bounds. Prompts are
rewritten by the small chat model, which is shown the episodes where the prompt
failed. Every change is one gene at a time, so an evaluation can tell what helped.
"""

from __future__ import annotations

import json
import os
import random
import sys
from typing import Any
from urllib.request import Request, urlopen

from . import episodes as ep
from .genome import DOMAINS, THINKING, get_gene

# gene -> (low, high, step size)
NUMERIC = {
    "fusion.decay": (0.1, 0.95, 0.1),
    "fusion.jevWeight": (0.2, 3.0, 0.25),
    "fusion.confusionEntropy": (0.5, 0.98, 0.05),
    "gates.doneThreshold": (0.5, 0.95, 0.05),
    **{f"strategies.{d}.riskThreshold": (0.05, 0.95, 0.1) for d in DOMAINS},
}
INTEGER = {"gates.maxRuns": (1, 8)}
CHOICE = {f"strategies.{d}.thinking": THINKING for d in DOMAINS}
PROMPTS = [f"strategies.{d}.prompt" for d in DOMAINS] + [f"cynefin.criteria.{d}" for d in DOMAINS]


def all_genes() -> list[str]:
    return [*NUMERIC, *INTEGER, *CHOICE, *PROMPTS]


def pick_gene(history: list[dict[str, Any]], version: int, rng: random.Random, prompts: bool = True) -> str:
    """Prefer genes of the domain where the current genome fails most. With no history, pick at random.

    `prompts=False` leaves out genes that need the chat model to rewrite them.
    """
    mine = [e for e in history if e.get("genomeVersion") == version]
    by_domain = ep.tally(mine, key=ep.final_domain)
    failing = [(t.failures / t.total, d) for d, t in by_domain.items() if d in DOMAINS and t.failures]
    if not failing:
        return rng.choice([g for g in all_genes() if prompts or g not in PROMPTS])
    domain = max(failing)[1]
    genes = [f"strategies.{domain}.riskThreshold", f"strategies.{domain}.thinking"]
    return rng.choice(genes + [f"strategies.{domain}.prompt"] if prompts else genes)


def mutate_value(genome: dict[str, Any], gene: str, rng: random.Random, history: list[dict[str, Any]]) -> Any:
    old = get_gene(genome, gene)
    if gene in NUMERIC:
        low, high, step = NUMERIC[gene]
        return round(min(high, max(low, old + rng.gauss(0, step))), 3)
    if gene in INTEGER:
        low, high = INTEGER[gene]
        return min(high, max(low, old + rng.choice((-1, 1))))
    if gene in CHOICE:
        return rng.choice([c for c in CHOICE[gene] if c != old])
    if gene in PROMPTS:
        return rewrite_prompt(gene, old, failures_for(gene, genome["version"], history))
    raise ValueError(f"{gene} is not an evolvable gene")


def failures_for(gene: str, version: int, history: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    domain = gene.split(".")[-2] if gene.startswith("strategies.") else gene.split(".")[-1]
    failed = [
        e
        for e in history
        if e.get("genomeVersion") == version and not ep.succeeded(e) and domain in (e.get("strategies") or [])
    ]
    return [
        {"task": e["task"], "criteria": e["criteria"], "outcome": e["outcome"], "strategies": e["strategies"]}
        for e in failed[-limit:]
    ]


def rewrite_prompt(gene: str, old: str, failures: list[dict[str, Any]]) -> str:
    """Ask the vLLM-served chat model for a better version of one prompt."""
    evidence = json.dumps(failures, indent=1) if failures else "(no failed episodes recorded yet)"
    request = (
        f"You are improving one setting of an AI agent harness. The setting is `{gene}`.\n\n"
        f"Current text:\n{old}\n\n"
        f"Episodes where the agent failed while this setting was in force:\n{evidence}\n\n"
        "Write a better version of the text. Keep what works, fix what the failures suggest, "
        "and stay under 1200 characters. Reply with the new text only."
    )
    text = chat(request).strip().strip('"')
    if not text:
        raise RuntimeError("the chat model returned an empty rewrite")
    return text[:1200]


def chat(prompt: str) -> str:
    base = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
    default = "mlx-community/Qwen3-4B-Instruct-2507-8bit" if sys.platform == "darwin" else "Qwen/Qwen3-4B-Instruct-2507"
    model = os.environ.get("VLLM_MODEL", default)
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.7, "max_tokens": 600}
    request = Request(
        f"{base}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {os.environ.get('VLLM_API_KEY', 'none')}"},
    )
    with urlopen(request, timeout=120) as response:
        return json.load(response)["choices"][0]["message"]["content"]


def propose(
    genome: dict[str, Any],
    history: list[dict[str, Any]],
    rng: random.Random,
    gene: str | None = None,
    prompts: bool = True,
) -> tuple[str, Any, Any]:
    """Return (gene, old value, new value). The caller builds and saves the child genome."""
    chosen = gene or pick_gene(history, genome["version"], rng, prompts)
    old = get_gene(genome, chosen)
    return chosen, old, mutate_value(genome, chosen, rng, history)
