// Offline stand-ins. They let the harness run end to end on a laptop, so the
// durable loop, the fusion and the episode log can be tested. Their answers are
// crude rules, so offline runs say nothing about how well the real models perform.
import {
	type AssistantMessage,
	type ClassifierAnswer,
	type ClassifierModel,
	type ClassifierResult,
	createProvider,
	type Message,
} from "@earendil-works/pi-ai";
import { fauxAssistantMessage, fauxProvider, fauxToolCall } from "@earendil-works/pi-ai/providers/faux";

const textOf = (message: Message | undefined): string => {
	if (message === undefined) return "";
	if (typeof message.content === "string") return message.content;
	return message.content.map((part) => ("text" in part ? part.text : "")).join("");
};

/**
 * A scripted worker. Asked for a task model, it returns JSON. Asked to create a file
 * ("create NAME containing TEXT"), it calls `write`, then reports. Otherwise it answers in one line.
 */
function respond(messages: Message[]): AssistantMessage {
	const last = messages.at(-1);
	if (last?.role === "toolResult") return fauxAssistantMessage(`Done. ${textOf(last).slice(0, 200)}`);
	const prompt = textOf(messages.findLast((m) => m.role === "user"));
	if (prompt.includes("TASK MODEL")) {
		const task = prompt.split("TASK:").at(-1)?.trim() ?? "";
		return fauxAssistantMessage(JSON.stringify({ goal: task, criteria: [`The request "${task}" is done.`] }));
	}
	const file = /create (\S+) containing "?([^"\n]+)"?/i.exec(prompt);
	if (file !== null) {
		return fauxAssistantMessage(fauxToolCall("write", { path: file[1]!, content: `${file[2]!}\n` }), {
			stopReason: "toolUse",
		});
	}
	return fauxAssistantMessage("I have looked at the task and have nothing to change.");
}

/** `tokensPerSecond` slows the stream down, which the restart test uses to stop the harness mid-answer. */
export function offlineChat(tokensPerSecond?: number) {
	const faux = fauxProvider({ provider: "offline", models: [{ id: "offline-worker" }], ...(tokensPerSecond ? { tokensPerSecond } : {}) });
	faux.setResponses(Array.from({ length: 500 }, () => (context) => respond(context.messages)));
	return { provider: faux.provider, modelId: "offline-worker" };
}

/** Keyword rules standing in for Jev. */
function judge(state: string, id: string, question: { type: string; criteria: unknown }): ClassifierAnswer {
	const lower = state.toLowerCase();
	if (question.type === "bool") {
		if (id === "risk") return { type: "bool", probability: /\brm -rf|drop table|--force\b/.test(lower) ? 0.95 : 0.05 };
		// Criteria count as met once the worker has reported back.
		return { type: "bool", probability: lower.includes('"lastanswer":"done') ? 0.9 : 0.2 };
	}
	if (question.type === "choice") {
		const keys = Object.keys(question.criteria as Record<string, string>);
		const pick = /outage|down|urgent|failing in production/.test(lower)
			? "chaotic"
			: /why|investigate|flaky|intermittent/.test(lower)
				? "complex"
				: /design|refactor|plan/.test(lower)
					? "complicated"
					: "clear";
		const probabilities = Object.fromEntries(keys.map((k) => [k, k === pick ? 0.7 : 0.3 / (keys.length - 1)]));
		return { type: "choice", choice: pick, probabilities, confidence: 0.7 };
	}
	return { type: "score", score: 0, confidence: 0.5 };
}

export function offlineJev() {
	const model: ClassifierModel<string> = {
		type: "classifier",
		id: "offline-jev",
		name: "Offline Jev",
		api: "offline-rules",
		provider: "offline-jev",
		baseUrl: "",
		input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 1_000_000,
	};
	const provider = createProvider({
		id: "offline-jev",
		auth: { apiKey: { name: "offline", resolve: async () => ({ auth: {} }) } },
		models: [model],
		classifiers: {
			"offline-rules": {
				classify: async (m, context): Promise<ClassifierResult> => ({
					api: m.api,
					provider: m.provider,
					model: m.id,
					answers: Object.fromEntries(
						Object.entries(context.questions).map(([id, q]) => [id, judge(JSON.stringify(context.state), id, q)]),
					),
					stopReason: "stop",
					timestamp: Date.now(),
				}),
			},
		},
	});
	return { provider, model };
}
