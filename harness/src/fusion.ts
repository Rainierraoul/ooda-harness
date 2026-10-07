// Information fusion: combining a stream of uncertain answers into one belief.
//
// Yes/no beliefs are kept as log-odds. Each new probability p adds weight * logit(p),
// which is Bayes' rule when observations are independent. Before adding, the old
// belief is multiplied by `decay`, so older evidence counts for less. That matters
// in complex and chaotic situations, where what was true three steps ago may not be now.
//
// Categorical beliefs (the Cynefin domain) use the same idea per category: a
// weighted logarithmic opinion pool with decay, normalised with softmax.

const EPS = 1e-4;
const clamp = (p: number) => Math.min(1 - EPS, Math.max(EPS, p));

export const logit = (p: number) => Math.log(clamp(p) / (1 - clamp(p)));
export const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

export interface Evidence<T> {
	value: T;
	weight: number;
}

/** Update a yes/no belief held as log-odds. A prior of 0 means 50/50. */
export function fuseBool(prior: number, evidence: Evidence<number>, decay: number): number {
	return decay * prior + evidence.weight * logit(evidence.value);
}

/** Update a categorical belief held as unnormalised log-scores. Missing categories count as near-zero probability. */
export function fuseCategorical<K extends string>(
	prior: Record<K, number>,
	evidence: Evidence<Partial<Record<K, number>>>,
	decay: number,
): Record<K, number> {
	const next = {} as Record<K, number>;
	for (const key of Object.keys(prior) as K[]) {
		next[key] = decay * prior[key] + evidence.weight * Math.log(clamp(evidence.value[key] ?? 0));
	}
	return next;
}

export function softmax<K extends string>(scores: Record<K, number>): Record<K, number> {
	const keys = Object.keys(scores) as K[];
	const max = Math.max(...keys.map((k) => scores[k]));
	const exps = keys.map((k) => Math.exp(scores[k] - max));
	const total = exps.reduce((a, b) => a + b, 0);
	return Object.fromEntries(keys.map((k, i) => [k, exps[i]! / total])) as Record<K, number>;
}

/** Entropy divided by its maximum, so 0 is certain and 1 is uniform. */
export function normalisedEntropy(probabilities: Record<string, number>): number {
	const values = Object.values(probabilities);
	if (values.length < 2) return 0;
	const h = -values.reduce((sum, p) => (p > 0 ? sum + p * Math.log(p) : sum), 0);
	return h / Math.log(values.length);
}

export function argmax<K extends string>(probabilities: Record<K, number>): K {
	return (Object.keys(probabilities) as K[]).reduce((best, k) => (probabilities[k] > probabilities[best] ? k : best));
}
