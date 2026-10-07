# ooda-harness

A durable OODA-loop agent harness, with Jev for fast decisions and a small vLLM-served model for the work.

This is an agent harness that works through a task in repeated cycles of observe, orient, decide and act (the OODA loop). It uses two local models. A small chat model served by vLLM does the work with file and shell tools. Jev, a decision model that answers typed questions with calibrated probabilities instead of generating text, makes the quick judgements: what kind of situation this is, whether each acceptance criterion is met, and whether a tool call looks destructive.

The loop runs on `@earendil-works/pi-durable`, the durable harness from the Pi project. Every step is committed to SQLite before it takes effect, so if the process dies, `--resume` continues from the last finished step.

A separate Python loop improves the harness over time. It reads the log of finished cycles, changes one setting, runs a task suite with each version, and adopts the change only if the new version does measurably better on objective checks.

## Words used here

- **Cycle**: one task, from the request to the final verdict. It may contain several runs.
- **Run**: one hand-over to the worker, which then works until it gives an answer.
- **Worker**: the small chat model with its tools, in its own pi-durable conversation.
- **Strategy**: the instructions, thinking level and risk limit used for a run. There is one per Cynefin domain.
- **Genome**: the JSON file holding every setting evolution may change (`genome/vNNNN.json`). `genome/CURRENT` names the one in use.
- **Episode**: the record of one finished cycle, appended to `runs/episodes.jsonl`.

## Layout

```
harness/src/        TypeScript harness (Node 22.19 or later; runs .ts directly)
  ooda.ts           the cycle as a durable task, plus the in-run tool gate
  fusion.ts         combining Jev answers over time
  state.ts          durable documents: task model, situation, episode
  models.ts         vLLM and Jev endpoints; offline stand-ins in offline.ts
  genome.ts         genome type and checks
  app.ts, cli.ts    opening the store, starting, resuming, logging
harness/test/       node:test tests, including a crash-and-resume test
evolve/             Python evolution loop (uv project, standard library only)
genome/             genome versions and the CURRENT pointer
tasks/suite.jsonl   evaluation tasks, one per line, each with an objective shell check
serve/              scripts that start vLLM and Open-Jev (mac-*.sh for Apple Silicon)
docs/architecture.md  how the pieces map onto Cynefin, OODA and information fusion
```

## Try it without a GPU

`--offline` swaps both models for scripted stand-ins. They can only handle "create FILE containing TEXT", which is enough to exercise the durable loop, fusion and logging.

```bash
npm install
```

```bash
node harness/src/cli.ts --offline --cwd "$(mktemp -d)" --check "grep -qx hello hello.txt" "Create hello.txt containing hello"
```

```bash
npm test
```

```bash
cd evolve && uv run pytest
```

## Run with the real models on a Mac

On Apple Silicon, vLLM runs through the vllm-metal plugin and Open-Jev runs on the GPU through PyTorch's MPS backend. Install vllm-metal once:

```bash
brew tap vllm-project/vllm-metal https://github.com/vllm-project/vllm-metal
```

```bash
brew install vllm-project/vllm-metal/vllm-metal
```

Then start each server in its own terminal. `mac-jev.sh` installs Open-Jev into `~/Development/Open-Jev` and downloads its model on first use. The worker is the 8-bit MLX build of Qwen3-4B-Instruct (4.3 GB). Jev takes about 6 GB. The vLLM script caps its cache at 30% of memory (`VLLM_GPU_UTIL`), since vllm-metal otherwise lets it grow to most of the machine.

```bash
serve/mac-vllm.sh
```

```bash
serve/mac-jev.sh
```

The harness picks the MLX model name automatically on macOS, so the commands below work unchanged.

## Run with the real models on a CUDA machine

Start both servers on a machine with a CUDA GPU. The scripts take their settings from environment variables listed at the top of each file.

```bash
serve/vllm.sh
```

```bash
serve/jev.sh
```

If the servers are on another machine, forward the ports or set `VLLM_BASE_URL` and `JEV_BASE_URL`. Then:

```bash
node harness/src/cli.ts --cwd ~/some/project "The tests in test_calc.py fail. Fix calc.py so they pass."
```

Without `--criterion`, the worker model writes the acceptance criteria itself before the first run. If the process is killed partway, run `node harness/src/cli.ts --resume` with the same `--store`.

## Evolve the settings

```bash
cd evolve && uv run evolve stats
```

```bash
cd evolve && uv run evolve cycle --repeats 3
```

`cycle` proposes a child of the current genome with one setting changed, runs the suite with both versions, and makes the child current only if there is at least a 90% chance it has the higher success rate. Prompts are rewritten by the vLLM model, which is shown the episodes where the old prompt failed. Numbers move by small steps within fixed bounds. The pieces can also be run one at a time with `propose`, `evaluate` and `promote`.

## Limits worth knowing

- The offline stand-ins say nothing about how well the real models do.
- A first baseline on an M4 Pro Mac (48 GB), with genome v1, Open-Jev 2B and Qwen3-4B-Instruct (8-bit MLX), ran each suite task once and passed all six checks. That means the suite is too easy to tell genomes apart, and evolution has nothing to select on until harder tasks are added.
- Jev was trained on general decision tasks, not on Cynefin. In that baseline the 2B model was confident only about the chaotic task (93%). Its answers for the other five were spread evenly enough to cross the 0.85 confusion cutoff, so they ran under the "confused" strategy. Fusing answers over several runs smooths single mistakes but cannot fix answers that are uninformative from the start.
- Open-Jev's context is 4,096 tokens, so the observation Jev sees is cut to the goal, the last answer and the last three tool results.
- With six tasks and three repeats, each genome gets 18 episodes. That only detects large improvements. Add tasks before trusting small ones.
- Success is judged by each task's shell check. When a task has no check, the fallback is Jev's own verdict, and a genome could then learn to satisfy Jev without doing the work.
- A cycle resumed after `genome/CURRENT` has changed continues with the new genome but its episode still records the old version.
- pi-durable is marked experimental and its API may change between releases. This project pins version 1.0.4.

## License

MIT. See [LICENSE](LICENSE). Open-Jev, Pi and the models are separate projects under their own licenses; nothing from them is included here.
