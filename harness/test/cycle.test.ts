import assert from "node:assert/strict";
import { existsSync, mkdirSync, mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { test } from "node:test";
import { BACKGROUND_CONTEXT } from "@earendil-works/chord/context";
import { openApp } from "../src/app.ts";
import { loadGenome, validateGenome } from "../src/genome.ts";
import { offlineModels } from "../src/models.ts";

const genome = loadGenome(resolve(import.meta.dirname, "../../genome/CURRENT"));

function scratch() {
	const dir = mkdtempSync(join(tmpdir(), "ooda-test-"));
	const cwd = join(dir, "work");
	mkdirSync(cwd);
	return { store: join(dir, "store.sqlite"), episodes: join(dir, "episodes.jsonl"), cwd };
}

test("the shipped genome is valid, and a broken one is rejected with reasons", () => {
	assert.equal(genome.version, 1);
	const broken = structuredClone(genome) as unknown as Record<string, any>;
	broken.fusion.decay = 2;
	delete broken.strategies.chaotic;
	assert.throws(() => validateGenome(broken), /fusion\.decay[\s\S]*strategies\.chaotic is missing/);
});

test("an offline cycle models the task, acts, and logs one episode", async () => {
	const { store, episodes, cwd } = scratch();
	const app = await openApp({ genome, setup: offlineModels(), store, cwd });
	try {
		await app.start("c1", "Create hello.txt containing hello");
		const episode = await app.collect("c1", episodes);
		assert.equal(episode?.outcome, "met");
		assert.deepEqual(episode?.strategies, ["clear"]);
		assert.equal(readFileSync(join(cwd, "hello.txt"), "utf8"), "hello\n");
		// Collecting again neither reruns the cycle nor logs it twice.
		await app.collect("c1", episodes);
		assert.equal(readFileSync(episodes, "utf8").trim().split("\n").length, 1);
	} finally {
		await app.close();
	}
});

test("a cycle stopped mid-answer finishes after the store is reopened", async () => {
	const { store, episodes, cwd } = scratch();
	const first = await openApp({ genome, setup: offlineModels({ tokensPerSecond: 20 }), store, cwd });
	await first.start("c2", "Create notes.txt containing remember the milk");
	first.resume();
	// Wait until the worker's generation is running, then shut down underneath it.
	const graph = await first.harness.taskGraph(BACKGROUND_CONTEXT);
	await new Promise<void>((done) => {
		const check = (value: typeof graph.value) => {
			if (Object.values(value.tasks).some((t) => t.kind === "pi.generation")) done();
		};
		check(graph.value);
		graph.subscribe(check);
	});
	await first.close();
	assert.equal(existsSync(join(cwd, "notes.txt")), false);

	const second = await openApp({ genome, setup: offlineModels(), store, cwd });
	try {
		assert.deepEqual(await second.pending(), ["c2"]);
		second.resume();
		const episode = await second.collect("c2", episodes);
		assert.equal(episode?.outcome, "met");
		assert.equal(readFileSync(join(cwd, "notes.txt"), "utf8"), "remember the milk\n");
	} finally {
		await second.close();
	}
});
