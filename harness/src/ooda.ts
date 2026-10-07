// The OODA cycle as a durable pi-durable task.
//
//   model    turn the request into a goal and acceptance criteria (task modelling)
//   observe  collect what the worker has done so far
//   orient   ask Jev what kind of situation this is and which criteria are met,
//            and fuse the answers with what was believed before
//   decide   stop if the criteria are met or the budget is spent; otherwise pick
//            the strategy for the current Cynefin domain
//   act      hand the worker (the small chat model with tools) one run under that strategy
//
// Each phase ends with a commit of its checkpoint, so a crash resumes at the last
// finished phase. The worker is a separate conversation owned by this task.
import type { Context } from "@earendil-works/chord";
import type { ClassifierModel, ClassifierQuestion, Message, Models } from "@earendil-works/pi-ai";
import {
	type ConversationId,
	configure,
	defineExtension,
	defineTask,
	hook,
	type JsonObject,
	section,
	type TaskRuntime,
	ToolTask,
	UsageDoc,
} from "@earendil-works/pi-durable";
import { fuseBool, fuseCategorical, softmax } from "./fusion.ts";
import type { Domain, Genome } from "./genome.ts";
import { ask, boolOf, choiceOf, truncate } from "./jev.ts";
import { chooseDomain, criterionProbabilities, type Episode, Situation, TaskModel } from "./state.ts";

export interface CycleInput {
	id: string;
	task: string;
	/** Acceptance criteria. When empty, the model phase asks the chat model to write them. */
	criteria: string[];
	genomeVersion: number;
}

type Common = { worker: ConversationId; run: number; startedAt: string; strategies: Domain[] };
type CycleState =
	| { phase: "model" }
	| ({ phase: "observe" } & Common)
	| ({ phase: "orient"; observation: JsonObject } & Common)
	| ({ phase: "decide" } & Common)
	| ({ phase: "act"; message: string } & Common);

const textOf = (message: Message | undefined): string =>
	message === undefined
		? ""
		: typeof message.content === "string"
			? message.content
			: message.content.map((part) => ("text" in part ? part.text : "")).join("");

export function createOoda(genome: Genome, models: Models, jev: ClassifierModel<string>) {
	const Cycle = defineTask<CycleInput, CycleState, Episode>({
		name: "ooda.cycle",
		version: 1,
		initial: () => ({ phase: "model" }),
		phases: {
			model: async (task, runtime, ctx) => {
				const model =
					task.input.criteria.length > 0
						? { goal: task.input.task, criteria: task.input.criteria }
						: await draftTaskModel(runtime, task.input.task, ctx);
				await runtime.commit(async (tx) => {
					const worker = await tx.createConversation({ ownership: { kind: "task", taskId: task.id } });
					Object.assign(await tx.doc(TaskModel, worker.id), model);
					(await tx.doc(Situation, worker.id)).criteria = model.criteria.map(() => 0);
					const startedAt = new Date().toISOString();
					return { status: "running", checkpoint: { phase: "observe", worker: worker.id, run: 0, startedAt, strategies: [] } };
				}, ctx);
			},

			observe: async (task, runtime, ctx) => {
				const { phase: _, ...common } = task.state.checkpoint;
				const { messages } = await runtime.context(common.worker, ctx);
				const model = await runtime.snapshot(TaskModel, common.worker, ctx);
				const answer = messages.findLast((m) => m.role === "assistant");
				const tools = messages
					.filter((m) => m.role === "toolResult")
					.slice(-3)
					.map((m) => ({ tool: m.toolName, failed: m.isError, output: truncate(textOf(m), 300) }));
				const observation: JsonObject = {
					goal: model?.goal ?? task.input.task,
					runsSoFar: common.run,
					lastAnswer: common.run === 0 ? "(nothing has been done yet)" : truncate(textOf(answer), 1500),
					recentToolResults: tools,
				};
				await runtime.commit(() => ({ status: "running", checkpoint: { ...common, phase: "orient", observation } }), ctx);
			},

			orient: async (task, runtime, ctx) => {
				const { phase: _, observation, ...common } = task.state.checkpoint;
				const model = await runtime.snapshot(TaskModel, common.worker, ctx);
				const criteria = model?.criteria ?? [];
				const questions: Record<string, ClassifierQuestion> = {
					domain: { type: "choice", instructions: genome.cynefin.instructions, criteria: genome.cynefin.criteria },
				};
				// Before the first run there is no evidence about the criteria, so they are not asked.
				if (common.run > 0) {
					criteria.forEach((text, i) => {
						questions[`c${i}`] = {
							type: "bool",
							instructions: `${genome.gates.criterionInstructions}\nCriterion: ${text}`,
							criteria: {
								true: "The evidence shows this criterion is met.",
								false: "It is not met, or the evidence does not say.",
							},
						};
					});
				}
				let answers: Awaited<ReturnType<typeof ask>> | undefined;
				let jevError: string | undefined;
				try {
					answers = await ask(runtime.models, jev, observation, questions, runtime.signal);
				} catch (error) {
					jevError = error instanceof Error ? error.message : String(error);
				}
				await runtime.commit(async (tx) => {
					const s = await tx.doc(Situation, common.worker);
					const { decay, jevWeight: weight } = genome.fusion;
					// No answer means no evidence: the belief is left as it was, not decayed.
					if (answers !== undefined) {
						s.domainScores = fuseCategorical<Domain>(s.domainScores, { value: choiceOf(answers.domain), weight }, decay);
						if (common.run > 0) {
							s.criteria = s.criteria.map((l, i) => fuseBool(l, { value: boolOf(answers[`c${i}`]), weight }, decay));
						}
					}
					s.history.push({
						run: common.run,
						domain: softmax(s.domainScores),
						criteria: criterionProbabilities(s.criteria),
						...(jevError === undefined ? {} : { jevError }),
					});
					return { status: "running", checkpoint: { ...common, phase: "decide" } };
				}, ctx);
			},

			decide: async (task, runtime, ctx) => {
				const { phase: _, ...common } = task.state.checkpoint;
				const s = (await runtime.snapshot(Situation, common.worker, ctx))!;
				const model = (await runtime.snapshot(TaskModel, common.worker, ctx))!;
				const met = common.run > 0 && criterionProbabilities(s.criteria).every((p) => p >= genome.gates.doneThreshold);
				if (met || common.run >= genome.gates.maxRuns) {
					return finish(task.input, common, met ? "met" : "exhausted", runtime, ctx);
				}
				const { domain, p } = chooseDomain(s.domainScores, genome);
				const strategy = genome.strategies[domain];
				const open = model.criteria.filter((_, i) => criterionProbabilities(s.criteria)[i]! < genome.gates.doneThreshold);
				const message =
					common.run === 0
						? `${task.input.task}\n\nAcceptance criteria:\n${model.criteria.map((c) => `- ${c}`).join("\n")}`
						: `Reassessment after run ${common.run}: the task now looks ${domain} ` +
							`(${Math.round(p[domain] * 100)}%). Criteria not yet met:\n${open.map((c) => `- ${c}`).join("\n")}\n` +
							`Continue under the strategy in your instructions.`;
				await runtime.commit(async (tx) => {
					await configure(tx, common.worker, { thinkingLevel: strategy.thinking });
					(await tx.doc(Situation, common.worker)).strategy = domain;
					return {
						status: "running",
						checkpoint: { ...common, phase: "act", message, strategies: [...common.strategies, domain] },
					};
				}, ctx);
			},

			act: async (task, runtime, ctx) => {
				const { phase: _, message, ...common } = task.state.checkpoint;
				const worker = (await runtime.conversation(common.worker, ctx))!;
				// The request ID makes this safe to repeat: after a crash the same submission is found, not sent twice.
				const request = { type: "input", content: message, requestId: `ooda:${task.id}:${common.run}` } as const;
				const settled = await (await worker.submit(request, ctx)).wait(ctx);
				if (settled.status === "unanswered") {
					return finish(task.input, { ...common, run: common.run + 1 }, "failed", runtime, ctx, settled.reason);
				}
				await runtime.commit(
					() => ({ status: "running", checkpoint: { ...common, phase: "observe", run: common.run + 1 } }),
					ctx,
				);
			},
		},
		abort: (_task, runtime, ctx) => runtime.commit(() => ({ status: "terminal", outcome: { status: "aborted" } }), ctx),
	});

	async function finish(
		input: CycleInput,
		common: Common,
		outcome: Episode["outcome"],
		runtime: TaskRuntime<CycleInput, CycleState, Episode, object>,
		ctx: Context,
		failure?: string,
	) {
		const s = (await runtime.snapshot(Situation, common.worker, ctx))!;
		const model = (await runtime.snapshot(TaskModel, common.worker, ctx))!;
		const usage = await runtime.snapshot(UsageDoc, common.worker, ctx);
		const tokens = { input: 0, output: 0 };
		for (const u of Object.values(usage?.models ?? {})) {
			tokens.input += u.input;
			tokens.output += u.output;
		}
		const probabilities = criterionProbabilities(s.criteria);
		const episode: Episode = {
			id: input.id,
			genomeVersion: input.genomeVersion,
			task: input.task,
			criteria: model.criteria.map((text, i) => ({ text, probability: probabilities[i]! })),
			outcome,
			runs: common.run,
			strategies: common.strategies,
			orientations: [...s.history],
			tokens,
			startedAt: common.startedAt,
			endedAt: new Date().toISOString(),
			...(failure === undefined ? {} : { failure }),
		};
		await runtime.commit(() => ({ status: "terminal", outcome: { status: "completed", result: episode } }), ctx);
	}

	// Inside a run: show the worker its goal and strategy, and screen its tool calls.
	const extension = defineExtension({
		name: "ooda",
		tasks: [Cycle],
		sections: [
			section("ooda", async (input, ctx) => {
				const model = await input.read.snapshot(TaskModel, input.conversationId, ctx);
				const s = await input.read.snapshot(Situation, input.conversationId, ctx);
				if (model === undefined || s === undefined || model.goal === "") return undefined;
				const probabilities = criterionProbabilities(s.criteria);
				const criteria = model.criteria.map(
					(c, i) => `- [${(probabilities[i] ?? 0) >= genome.gates.doneThreshold ? "met" : "open"}] ${c}`,
				);
				return [
					`Goal: ${model.goal}`,
					`Acceptance criteria:\n${criteria.join("\n")}`,
					`The task currently looks ${s.strategy} (Cynefin domain).`,
					`Strategy: ${genome.strategies[s.strategy].prompt}`,
				].join("\n\n");
			}),
		],
		hooks: [
			hook(ToolTask, {
				beforeTool: async (call, api, ctx) => {
					const s = await api.snapshot(Situation, api.conversationId, ctx);
					if (s === undefined) return undefined;
					// Memoised so a replay after a crash reuses the first answer instead of asking Jev again.
					const key = `ooda.risk:${call.id}`;
					let p = await api.memo<number>(key, ctx);
					if (p === undefined) {
						const state = { tool: call.name, arguments: truncate(JSON.stringify(call.arguments), 2000) };
						const question: ClassifierQuestion = {
							type: "bool",
							instructions: genome.gates.riskInstructions,
							criteria: { true: "Yes, it is destructive or reaches outside the task.", false: "No, it is safe." },
						};
						// If Jev cannot answer, 0.5 means "unknown": strategies with a threshold below 0.5 then block.
						p = await ask(models, jev, state, { risk: question })
							.then((answers) => boolOf(answers.risk))
							.catch(() => 0.5);
						p = await api.memo(key, p, ctx);
					}
					const threshold = genome.strategies[s.strategy].riskThreshold;
					return p > threshold
						? {
								block:
									`Blocked under the ${s.strategy} strategy: Jev puts the chance that this call is destructive ` +
									`at ${Math.round(p * 100)}%, above the ${Math.round(threshold * 100)}% limit. Find a safer, reversible way.`,
							}
						: undefined;
				},
			}),
		],
	});

	return { Cycle, extension };
}

/** Task modelling: ask the chat model for a goal and checkable acceptance criteria. */
async function draftTaskModel(
	runtime: TaskRuntime<CycleInput, CycleState, Episode, object>,
	task: string,
	ctx: Context,
): Promise<{ goal: string; criteria: string[] }> {
	const fallback = { goal: task, criteria: ["The request is fully carried out."] };
	const ref = (await runtime.agent(ctx)).model;
	const model = ref === undefined ? undefined : runtime.models.getModel(ref.provider, ref.modelId);
	if (model === undefined) return fallback;
	const prompt =
		"TASK MODEL\nRestate the task below as one goal sentence and two to five acceptance criteria. " +
		"Each criterion must be something an observer could check from the work's output. " +
		'Reply with JSON only, in the form {"goal": "...", "criteria": ["...", "..."]}.\n\nTASK: ' +
		task;
	const reply = await runtime.models.completeSimple(
		model,
		{ messages: [{ role: "user", content: prompt, timestamp: Date.now() }] },
		{ signal: runtime.signal },
	);
	try {
		const text = textOf(reply);
		const parsed = JSON.parse(text.slice(text.indexOf("{"), text.lastIndexOf("}") + 1)) as { goal?: unknown; criteria?: unknown };
		const criteria = Array.isArray(parsed.criteria) ? parsed.criteria.filter((c): c is string => typeof c === "string") : [];
		if (typeof parsed.goal !== "string" || criteria.length === 0) return fallback;
		return { goal: parsed.goal, criteria: criteria.slice(0, 5) };
	} catch {
		return fallback;
	}
}
