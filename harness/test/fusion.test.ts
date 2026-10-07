import assert from "node:assert/strict";
import { test } from "node:test";
import { argmax, fuseBool, fuseCategorical, logit, normalisedEntropy, sigmoid, softmax } from "../src/fusion.ts";

const close = (a: number, b: number, tolerance = 1e-6) => assert.ok(Math.abs(a - b) < tolerance, `${a} is not ${b}`);

test("one piece of evidence from a 50/50 prior gives back its own probability", () => {
	close(sigmoid(fuseBool(0, { value: 0.8, weight: 1 }, 0.6)), 0.8);
});

test("two agreeing pieces of evidence are stronger than one", () => {
	const once = fuseBool(0, { value: 0.8, weight: 1 }, 1);
	const twice = fuseBool(once, { value: 0.8, weight: 1 }, 1);
	assert.ok(sigmoid(twice) > 0.8);
	close(twice, 2 * logit(0.8));
});

test("decay lets new evidence overturn an old belief", () => {
	const old = 3 * logit(0.9); // three earlier "yes" answers
	const withDecay = fuseBool(old, { value: 0.05, weight: 1 }, 0.3);
	const withoutDecay = fuseBool(old, { value: 0.05, weight: 1 }, 1);
	assert.ok(sigmoid(withDecay) < 0.5);
	assert.ok(sigmoid(withoutDecay) > 0.5);
});

test("categorical fusion from a flat prior returns the evidence distribution", () => {
	const prior = { a: 0, b: 0, c: 0 };
	const p = softmax(fuseCategorical(prior, { value: { a: 0.6, b: 0.3, c: 0.1 }, weight: 1 }, 0.5));
	close(p.a, 0.6, 1e-3);
	close(p.b, 0.3, 1e-3);
	assert.equal(argmax(p), "a");
});

test("normalised entropy is 1 for a uniform distribution and near 0 for a certain one", () => {
	close(normalisedEntropy({ a: 0.25, b: 0.25, c: 0.25, d: 0.25 }), 1);
	assert.ok(normalisedEntropy({ a: 0.999, b: 0.0005, c: 0.0005 }) < 0.02);
});
