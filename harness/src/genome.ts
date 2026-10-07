// The genome is every setting the evolution loop may change. It is plain data:
// prompts, thresholds and weights. Code never comes from a genome.
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";

export const DOMAINS = ["clear", "complicated", "complex", "chaotic", "confused"] as const;
export type Domain = (typeof DOMAINS)[number];

export const THINKING = ["minimal", "low", "medium", "high"] as const;
export type Thinking = (typeof THINKING)[number];

export interface Strategy {
	thinking: Thinking;
	/** Block a tool call when Jev's probability that it is destructive exceeds this. */
	riskThreshold: number;
	prompt: string;
}

export interface Genome {
	version: number;
	parent: number | null;
	note: string;
	fusion: {
		/** How much of the previous belief survives each new observation, 0 to 1. */
		decay: number;
		/** How much a Jev answer counts as evidence. */
		jevWeight: number;
		/** Normalised entropy of the domain belief above which the task is treated as confused. */
		confusionEntropy: number;
	};
	cynefin: { instructions: string; criteria: Record<Domain, string> };
	strategies: Record<Domain, Strategy>;
	gates: {
		doneThreshold: number;
		maxRuns: number;
		riskInstructions: string;
		criterionInstructions: string;
	};
}

const PROMPT_LIMIT = 2000;

/** Throws with a list of every problem found. The Python side checks the same rules. */
export function validateGenome(value: unknown): Genome {
	const problems: string[] = [];
	const g = value as Genome;
	const unit = (path: string, n: unknown) => {
		if (typeof n !== "number" || !(n >= 0 && n <= 1)) problems.push(`${path} must be a number from 0 to 1`);
	};
	const text = (path: string, s: unknown) => {
		if (typeof s !== "string" || s.trim() === "" || s.length > PROMPT_LIMIT) {
			problems.push(`${path} must be non-empty text of at most ${PROMPT_LIMIT} characters`);
		}
	};
	if (typeof g !== "object" || g === null) throw new Error("genome must be an object");
	if (!Number.isInteger(g.version) || g.version < 1) problems.push("version must be a positive integer");
	unit("fusion.decay", g.fusion?.decay);
	unit("fusion.confusionEntropy", g.fusion?.confusionEntropy);
	if (typeof g.fusion?.jevWeight !== "number" || g.fusion.jevWeight <= 0 || g.fusion.jevWeight > 5) {
		problems.push("fusion.jevWeight must be above 0 and at most 5");
	}
	text("cynefin.instructions", g.cynefin?.instructions);
	for (const domain of DOMAINS) {
		text(`cynefin.criteria.${domain}`, g.cynefin?.criteria?.[domain]);
		const s = g.strategies?.[domain];
		if (s === undefined) {
			problems.push(`strategies.${domain} is missing`);
			continue;
		}
		if (!THINKING.includes(s.thinking)) problems.push(`strategies.${domain}.thinking must be one of ${THINKING.join(", ")}`);
		unit(`strategies.${domain}.riskThreshold`, s.riskThreshold);
		text(`strategies.${domain}.prompt`, s.prompt);
	}
	unit("gates.doneThreshold", g.gates?.doneThreshold);
	if (!Number.isInteger(g.gates?.maxRuns) || g.gates.maxRuns < 1 || g.gates.maxRuns > 10) {
		problems.push("gates.maxRuns must be an integer from 1 to 10");
	}
	text("gates.riskInstructions", g.gates?.riskInstructions);
	text("gates.criterionInstructions", g.gates?.criterionInstructions);
	if (problems.length > 0) throw new Error(`invalid genome:\n  ${problems.join("\n  ")}`);
	return g;
}

/** Loads a genome file, or the one named in genome/CURRENT when `path` is a directory's CURRENT file. */
export function loadGenome(path: string): Genome {
	const resolved = path.endsWith("CURRENT")
		? resolve(dirname(dirname(path)), readFileSync(path, "utf8").trim())
		: path;
	return validateGenome(JSON.parse(readFileSync(resolved, "utf8")));
}
