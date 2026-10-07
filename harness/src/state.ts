// Durable documents. Each is committed together with the task checkpoint that
// produced it, so after a crash the belief state and the checkpoint always agree.
import { defineDoc } from "@earendil-works/pi-durable";
import { argmax, normalisedEntropy, sigmoid, softmax } from "./fusion.ts";
import { DOMAINS, type Domain, type Genome } from "./genome.ts";

/** What the task is and how we will know it is finished. Lives on the worker conversation. */
export const TaskModel = defineDoc<{ goal: string; criteria: string[] }>({
	kind: "ooda.task-model",
	version: 1,
	scope: "conversation",
	history: "latest",
	fork: "current",
	initial: () => ({ goal: "", criteria: [] }),
});

export type Orientation = {
	run: number;
	domain: Record<Domain, number>;
	criteria: number[];
	jevError?: string;
};

/** The fused picture of the situation. Lives on the worker conversation. */
export const Situation = defineDoc<{
	/** Log-scores per Cynefin domain; softmax gives probabilities. */
	domainScores: Record<Domain, number>;
	/** Log-odds per acceptance criterion, in TaskModel order. */
	criteria: number[];
	/** The domain whose strategy is in force. */
	strategy: Domain;
	history: Orientation[];
}>({
	kind: "ooda.situation",
	version: 1,
	scope: "conversation",
	history: "latest",
	fork: "current",
	initial: () => ({
		domainScores: Object.fromEntries(DOMAINS.map((d) => [d, 0])) as Record<Domain, number>,
		criteria: [],
		strategy: "confused",
		history: [],
	}),
});

/** OODA cycles started from the controller conversation, so a restarted CLI can find and log them. */
export const Cycles = defineDoc<{ cycles: Record<string, { taskId: number; logged: boolean }> }>({
	kind: "ooda.cycles",
	version: 1,
	scope: "conversation",
	history: "latest",
	fork: "initial",
	initial: () => ({ cycles: {} }),
});

/** One finished cycle, as written to runs/episodes.jsonl for the evolution loop. */
export type Episode = {
	id: string;
	genomeVersion: number;
	task: string;
	criteria: { text: string; probability: number }[];
	outcome: "met" | "exhausted" | "failed";
	runs: number;
	/** The strategy used in each run, in order. */
	strategies: Domain[];
	orientations: Orientation[];
	tokens: { input: number; output: number };
	startedAt: string;
	endedAt: string;
	/** Filled in by the evaluation runner when the task has an objective check. */
	verified?: boolean | null;
	failure?: string;
};

/** The domain to act on. High uncertainty about the domain is itself the "confused" domain. */
export function chooseDomain(domainScores: Record<Domain, number>, genome: Genome): { domain: Domain; p: Record<Domain, number> } {
	const p = softmax(domainScores);
	const domain = normalisedEntropy(p) > genome.fusion.confusionEntropy ? "confused" : argmax(p);
	return { domain, p };
}

export const criterionProbabilities = (logOdds: readonly number[]) => logOdds.map(sigmoid);
