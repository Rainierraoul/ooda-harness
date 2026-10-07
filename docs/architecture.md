# Architecture

This note explains how the ideas behind the harness turn into code, and why some choices were made. Words defined in the README (cycle, run, worker, strategy, genome, episode) are used without explanation.

## Two models with different jobs

The worker is a small instruction-tuned model served by vLLM (by default `Qwen/Qwen3-4B-Instruct-2507`). It reads files, edits them and runs commands. It is slow compared with a classifier and can make things up, so the harness never asks it to judge its own progress.

Jev answers structured questions about a piece of state. A question is a choice between named options, a yes/no, or a score on a scale, and the answer is a probability for each option, produced in one forward pass with no generated text to parse. That makes it fast and its answers directly usable as evidence. The harness uses the self-hosted Open-Jev server, which speaks the same `/v1/systemone` protocol as TypeSafe's hosted Jev. pi-ai already has a client for that protocol (`models.classify()`), so switching to the hosted service, or to OpenRouter's Jev, is a change of provider in `harness/src/models.ts`.

## The OODA cycle as a durable task

`harness/src/ooda.ts` defines the cycle as a pi-durable task with five phases. Each phase ends by committing its checkpoint together with any documents it changed, in one transaction. After a crash, pi-durable restarts the task at the phase whose checkpoint was last committed.

| Phase | What it does | Model used |
|---|---|---|
| model | Turns the request into a goal and two to five acceptance criteria, unless criteria were given. Creates the worker conversation. | worker |
| observe | Collects the worker's last answer and its last three tool results. | none |
| orient | Asks Jev for the Cynefin domain and, after the first run, whether each criterion is met. Fuses the answers into the situation document. | Jev |
| decide | Ends the cycle if every criterion is believed met or the run budget is spent. Otherwise picks the domain to act on and sets the worker's thinking level. | none |
| act | Sends the worker one message and waits for its answer. | worker |

The act phase sends its message with a request ID built from the cycle and run number. If the process dies while the worker is busy, pi-durable finds the same submission on restart and waits for it instead of sending the message again. The test `a cycle stopped mid-answer finishes after the store is reopened` shuts the harness down during the worker's answer and checks that the cycle completes correctly afterwards.

Inside a run, two things from the same extension apply. A prompt section shows the worker its goal, which criteria are still open, the current domain and that domain's strategy text. A `beforeTool` hook asks Jev whether each tool call looks destructive and blocks it if the probability is above the strategy's limit. The hook stores Jev's answer with pi-durable's `memo`, so a replay after a crash reuses the first answer rather than asking again and possibly getting a different one. If Jev cannot be reached, the hook uses 0.5 as the probability. The complex, chaotic and confused strategies have limits below 0.5, so they block calls when Jev cannot be asked. The clear and complicated strategies allow them.

## Cynefin

Cynefin sorts situations by how cause and effect relate, and each kind calls for a different way of working. The genome holds one strategy per domain:

- Clear: apply the standard approach and stop when done. Minimal thinking, lenient risk limit.
- Complicated: analyse and plan first, then check each criterion. High thinking.
- Complex: small reversible experiments, keep what works. Strict risk limit.
- Chaotic: stop the damage with the smallest action, then report. Strictest limit.
- Confused: change nothing yet; split the task and say what would settle it.

The thinking level only has an effect when the vLLM model is a reasoning model and `VLLM_REASONING=1` is set.

The domain is not fixed at the start. Jev is asked again after every run, so a task that looked complicated can be reclassified as complex when the first plan fails. The next run then gets the complex strategy and a message saying the situation was reassessed.

Cynefin treats not knowing which domain applies as a domain of its own. The harness follows that literally. When the fused probabilities over the five domains are close to even (normalised entropy above `fusion.confusionEntropy`), the decide phase picks "confused" whatever the most likely single domain is.

## Information fusion

Jev's answers are noisy, and one answer per run is a thin basis for a decision. `harness/src/fusion.ts` combines them over time.

Each criterion's belief is kept as log-odds. A new answer with probability p adds `weight × logit(p)`, which is Bayes' rule when the answers are independent. Before adding, the old belief is multiplied by `decay`, a number below 1, so older answers count for less. Without decay, a criterion that was met and then broken by a later edit would stay believed met for several runs.

The domain belief uses the same rule per domain, a weighted logarithmic opinion pool with decay, and is turned back into probabilities with softmax.

The design has room for more evidence sources, but only Jev feeds it today. Test results, linters or the worker's own report could be added as further evidence with their own weights. They are left out because each needs a reliability weight that should itself be learned from episodes.

## Evolution

The Python package in `evolve/` changes the genome, never the code. A genome is checked against fixed bounds before it is saved, and versions are never overwritten, so every past genome stays available and can be made current again.

One step of evolution works like this:

1. Pick a setting to change. If the current genome has failures, pick one of the strategy settings for the domain where it fails most. If not, pick any setting.
2. Change it. Numbers take a random step within bounds. A prompt is rewritten by the vLLM model, which is shown up to five failed episodes for that domain.
3. Run the suite with the new genome, and with the current one if it has not been run yet. Each task starts in a fresh directory with its own store.
4. Compare. Each genome's success rate gets a Beta posterior from its successes and failures. The child becomes current only if the chance that its rate is higher is at least 90% and each genome has enough episodes.

Success means the task's shell check passed. The checks are written by people and test the actual result, for example that `python3 test_calc.py` prints ok. Using Jev's verdict instead would let evolution reward settings that persuade Jev rather than settings that finish the work.

Because only one setting changes per step, a promoted genome's `note` says exactly what changed, and `evolve stats` shows how each version did per domain.
