// Opening the harness on a store, and starting and collecting OODA cycles.
// Shared by the CLI and the tests.
import { appendFileSync, mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { BACKGROUND_CONTEXT } from "@earendil-works/chord/context";
import { createRegistry, Harness, type TaskId } from "@earendil-works/pi-durable";
import { NodeExecutionEnv } from "@earendil-works/pi-durable/env/node";
import { openNodeSqliteStorage } from "@earendil-works/pi-durable/storage/sqlite/node";
import { CodingTools } from "@earendil-works/pi-durable/tools";
import type { Genome } from "./genome.ts";
import type { ModelSetup } from "./models.ts";
import { createOoda } from "./ooda.ts";
import { Cycles, type Episode } from "./state.ts";

const context = BACKGROUND_CONTEXT;

export interface AppOptions {
	genome: Genome;
	setup: ModelSetup;
	store: string;
	cwd: string;
}

export async function openApp({ genome, setup, store, cwd }: AppOptions) {
	const { Cycle, extension } = createOoda(genome, setup.models, setup.jev);
	const registry = createRegistry();
	registry.install(CodingTools);
	registry.install(extension);

	mkdirSync(dirname(resolve(store)), { recursive: true });
	const harness = await Harness.open(
		await openNodeSqliteStorage(resolve(store)),
		{
			models: setup.models,
			registry,
			env: ({ cwd: dir }) => new NodeExecutionEnv({ cwd: dir ?? cwd }),
			settings: { retry: { maxRetries: 2 }, stream: { timeoutMs: 300_000 } },
		},
		context,
	);
	const controller = await harness.root(context);
	// Workers copy the controller's agent when they are created.
	await controller.configure({ model: setup.chat, cwd: resolve(cwd) }, context);

	/** Start a cycle. Starting the same id twice returns the existing one. */
	async function start(id: string, task: string, criteria: string[] = []): Promise<TaskId<Episode>> {
		return controller.commit(async (tx) => {
			const doc = await tx.doc(Cycles, controller.id);
			const existing = doc.cycles[id];
			if (existing !== undefined) return existing.taskId as TaskId<Episode>;
			const input = { id, task, criteria, genomeVersion: genome.version };
			const taskId = await tx.createTask(Cycle, input, { ownership: { kind: "conversation" } });
			doc.cycles[id] = { taskId, logged: false };
			return taskId;
		}, context);
	}

	/** Wait for a cycle, append its episode to the log once, and mark it logged. */
	async function collect(id: string, episodes: string, verify?: (e: Episode) => boolean): Promise<Episode | undefined> {
		const cycle = (await harness.snapshot(Cycles, controller.id, context))?.cycles[id];
		if (cycle === undefined) throw new Error(`no cycle named ${id}`);
		const settled = await harness.waitForTask(cycle.taskId as TaskId<Episode>, context);
		const outcome = settled.state.outcome;
		let episode: Episode | undefined;
		if (outcome.status === "completed") {
			episode = { ...outcome.result };
			if (verify !== undefined) episode.verified = verify(episode);
			if (!cycle.logged) {
				mkdirSync(dirname(resolve(episodes)), { recursive: true });
				appendFileSync(resolve(episodes), `${JSON.stringify(episode)}\n`);
			}
		}
		await controller.commit(async (tx) => {
			const entry = (await tx.doc(Cycles, controller.id)).cycles[id];
			if (entry !== undefined) entry.logged = true;
		}, context);
		return episode;
	}

	async function pending(): Promise<string[]> {
		const cycles = (await harness.snapshot(Cycles, controller.id, context))?.cycles ?? {};
		return Object.entries(cycles)
			.filter(([, c]) => !c.logged)
			.map(([id]) => id);
	}

	return {
		harness,
		start,
		collect,
		pending,
		resume: () => harness.resume(),
		close: () => harness.close(context),
	};
}
