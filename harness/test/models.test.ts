import assert from "node:assert/strict";
import { test } from "node:test";
import { localModels } from "../src/models.ts";

// pi-ai's clients refuse to send a request without a key, and the offline tests never reach them.
test("both local providers resolve an API key", async () => {
	const { models, chat, jev } = localModels();
	for (const provider of [chat.provider, jev.provider]) {
		const auth = await models.getAuth(provider);
		assert.ok(auth?.auth.apiKey, `${provider} has no key`);
	}
});
