// Asking Jev typed questions. Jev returns calibrated probabilities from one
// forward pass, so these checks are fast and cheap compared with asking the chat model.
import type { ClassifierAnswer, ClassifierModel, ClassifierQuestion, Models } from "@earendil-works/pi-ai";
import type { JsonObject } from "@earendil-works/pi-durable";

// Open-Jev's default context is 4096 tokens. Keep the state well under it.
const STATE_CHAR_BUDGET = 6000;

export class JevError extends Error {}

export async function ask(
	models: Models,
	jev: ClassifierModel<string>,
	state: JsonObject,
	questions: Record<string, ClassifierQuestion>,
	signal?: AbortSignal,
): Promise<Record<string, ClassifierAnswer>> {
	const size = JSON.stringify(state).length;
	if (size > STATE_CHAR_BUDGET) throw new JevError(`state is ${size} characters, over the ${STATE_CHAR_BUDGET} budget`);
	const result = await models.classify(jev, { state, questions }, { signal, timeoutMs: 30_000 });
	if (result.stopReason !== "stop") throw new JevError(result.errorMessage ?? result.stopReason);
	return result.answers;
}

export function boolOf(answer: ClassifierAnswer | undefined): number {
	if (answer?.type !== "bool") throw new JevError("expected a yes/no answer");
	return answer.probability;
}

export function choiceOf(answer: ClassifierAnswer | undefined): Record<string, number> {
	if (answer?.type !== "choice") throw new JevError("expected a choice answer");
	return answer.probabilities;
}

export const truncate = (text: string, limit: number) =>
	text.length <= limit ? text : `${text.slice(0, limit)} [${text.length - limit} more characters]`;
