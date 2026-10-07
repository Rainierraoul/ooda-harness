// Run one OODA cycle, or resume cycles a previous process left unfinished.
//
//   node harness/src/cli.ts "Create hello.txt containing hello"
//   node harness/src/cli.ts --offline --cwd /tmp/work "Create hello.txt containing hello"
//   node harness/src/cli.ts --resume
//
// Every finished cycle is appended to the episodes file, which the Python
// evolution loop reads.
import { spawnSync } from "node:child_process";
import { dirname, resolve } from "node:path";
import { parseArgs } from "node:util";
import { openApp } from "./app.ts";
import { loadGenome } from "./genome.ts";
import { localModels, offlineModels } from "./models.ts";
import type { Episode } from "./state.ts";

const root = resolve(dirname(new URL(import.meta.url).pathname), "../..");
const { values: opts, positionals } = parseArgs({
	allowPositionals: true,
	options: {
		offline: { type: "boolean", default: false },
		resume: { type: "boolean", default: false },
		genome: { type: "string", default: resolve(root, "genome/CURRENT") },
		store: { type: "string", default: resolve(root, "runs/session.sqlite") },
		episodes: { type: "string", default: resolve(root, "runs/episodes.jsonl") },
		cwd: { type: "string", default: process.cwd() },
		criterion: { type: "string", multiple: true, default: [] },
		id: { type: "string" },
		// A shell command run in --cwd after the cycle. Exit code 0 marks the episode verified.
		check: { type: "string" },
	},
});

const task = positionals.join(" ").trim();
if (!opts.resume && task === "") {
	console.error('usage: cli.ts [--offline] [--cwd DIR] [--criterion TEXT]... [--check CMD] [--id ID] "task"');
	console.error("       cli.ts --resume");
	process.exit(2);
}

const app = await openApp({
	genome: loadGenome(resolve(opts.genome)),
	setup: opts.offline ? offlineModels() : localModels(),
	store: opts.store,
	cwd: opts.cwd,
});

const check = opts.check;
const verify =
	check === undefined ? undefined : () => spawnSync(check, { shell: true, cwd: opts.cwd, stdio: "ignore" }).status === 0;

function report(id: string, episode: Episode | undefined) {
	if (episode === undefined) {
		console.error(`${id}: did not complete`);
		return;
	}
	const path = episode.strategies.join(" → ") || "(none)";
	const verified = episode.verified === undefined ? "" : `, verified: ${episode.verified}`;
	console.log(`${id}: ${episode.outcome} after ${episode.runs} run(s); strategies: ${path}${verified}`);
	for (const c of episode.criteria) console.log(`  ${(c.probability * 100).toFixed(0).padStart(3)}%  ${c.text}`);
}

try {
	if (opts.resume) {
		app.resume();
		const ids = await app.pending();
		if (ids.length === 0) console.log("nothing to resume");
		for (const id of ids) report(id, await app.collect(id, opts.episodes));
	} else {
		const id = opts.id ?? `cycle-${new Date().toISOString().replace(/[:.]/g, "-")}`;
		await app.start(id, task, opts.criterion);
		report(id, await app.collect(id, opts.episodes, verify));
	}
} finally {
	await app.close();
}
