// Two models, both served locally:
//   - a small chat model behind vLLM's OpenAI-compatible API, which does the work;
//   - Jev behind Open-Jev's /v1/systemone, which answers typed questions with probabilities.
// pi-ai already speaks both protocols, so this file only describes where they are.
import { type ClassifierModel, createModels, createProvider, type Model, type MutableModels } from "@earendil-works/pi-ai";
import { openAICompletionsApi } from "@earendil-works/pi-ai/api/openai-completions.lazy";
import { typesafeSystemOneApi } from "@earendil-works/pi-ai/api/typesafe-system-one.lazy";
import { offlineChat, offlineJev } from "./offline.ts";

export interface ModelSetup {
	models: MutableModels;
	chat: { provider: string; modelId: string };
	jev: ClassifierModel<string>;
}

const env = (name: string, fallback: string) => process.env[name] ?? fallback;
// pi-ai's clients refuse to send a request without a key. Local servers ignore it,
// so a placeholder is used unless the environment variable is set.
const localKey = (variable: string) => ({
	apiKey: { name: variable, resolve: async () => ({ auth: { apiKey: process.env[variable] ?? "local" } }) },
});
const free = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 };

export function localModels(): ModelSetup {
	// On a Mac, serve/mac-vllm.sh serves the MLX build of the same model through vllm-metal.
	const modelId = env(
		"VLLM_MODEL",
		process.platform === "darwin" ? "mlx-community/Qwen3-4B-Instruct-2507-8bit" : "Qwen/Qwen3-4B-Instruct-2507",
	);
	const reasoning = process.env.VLLM_REASONING === "1";
	const chatModel: Model<"openai-completions"> = {
		id: modelId,
		name: modelId,
		api: "openai-completions",
		provider: "vllm",
		baseUrl: env("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"),
		reasoning,
		input: ["text"],
		cost: free,
		contextWindow: Number(env("VLLM_CONTEXT", "32768")),
		maxTokens: Number(env("VLLM_MAX_TOKENS", "4096")),
		// vLLM does not accept the `developer` role or `reasoning_effort`. Qwen models switch
		// thinking on and off through the chat template instead.
		compat: {
			supportsDeveloperRole: false,
			supportsReasoningEffort: false,
			...(reasoning ? { thinkingFormat: "qwen-chat-template" as const } : {}),
		},
	};
	const jevModel: ClassifierModel<"typesafe-system-one"> = {
		type: "classifier",
		id: env("JEV_MODEL", "open-jev"),
		name: "Jev",
		api: "typesafe-system-one",
		provider: "jev",
		baseUrl: env("JEV_BASE_URL", "http://127.0.0.1:8791/v1"),
		input: ["text"],
		cost: free,
		contextWindow: Number(env("JEV_CONTEXT", "4096")),
	};

	const models = createModels();
	models.setProvider(
		createProvider({ id: "vllm", name: "vLLM", auth: localKey("VLLM_API_KEY"), models: [chatModel], api: openAICompletionsApi() }),
	);
	models.setProvider(
		createProvider({
			id: "jev",
			name: "Jev",
			auth: localKey("JEV_API_KEY"),
			models: [jevModel],
			classifiers: { "typesafe-system-one": typesafeSystemOneApi() },
		}),
	);
	return { models, chat: { provider: "vllm", modelId }, jev: jevModel };
}

/** Stand-ins for both models, so the plumbing can be run and tested without a GPU. */
export function offlineModels(options: { tokensPerSecond?: number } = {}): ModelSetup {
	const models = createModels();
	const chat = offlineChat(options.tokensPerSecond);
	models.setProvider(chat.provider);
	const jev = offlineJev();
	models.setProvider(jev.provider);
	return { models, chat: { provider: chat.provider.id, modelId: chat.modelId }, jev: jev.model };
}
