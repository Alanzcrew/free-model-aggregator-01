# Free Model Aggregator

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/Alanzcrew/free-model-aggregator-01/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/Alanzcrew/free-model-aggregator-01/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

One server that serves an OpenAI- and Anthropic-compatible API over the best free models it can reach. It reads a brief of free providers, their quality ratings and the workload, probes each provider's catalog, and answers every request with the strongest free model that meets it — failing over to the next best when a provider refuses.

## Run it

```sh
MAKEFUN_ENV_URL=<env-url> MAKEFUN_PROMPT_FILE=<prompt-file> node main.ts
```

`--list` prints the catalog the server would register, as one JSON document, and exits:

```sh
MAKEFUN_ENV_URL=<env-url> MAKEFUN_PROMPT_FILE=<prompt-file> node main.ts --list
```

Node 22 or newer, standard library only; nothing is installed.

## How it routes

- **Requirements first.** Each request's requirements come from the workload manifest (tools, vision, JSON mode, minimum context), with the body's own signals merged on top. A model that lacks something the request needs is never picked, and a paid model is never a legal pick.
- **Best free model wins.** Candidates are ranked by the quality table's rating, ties by provider id then model id. Quality is the score's largest share, so the best available model answers first.
- **Failover.** A provider that refuses (402, 429, 5xx, a timeout, an unparseable answer) costs one failure and the request is retried on the next candidate's provider. A 402 retires a provider for the trial; two consecutive failures retire it for later requests. That is how a spent budget or an outage stops being retried.
- **All three wires.** Upstream calls are non-streaming JSON in the provider's own wire — OpenAI Chat Completions, Anthropic Messages, Gemini generateContent — then rendered back in the wire the client used, as JSON or as server-sent events. A request carrying JSON mode never goes to an Anthropic-wire provider, whose wire cannot carry it.
- **Deterministic.** No wall clock or unseeded randomness reaches a request: ids and timestamps are fixed constants, and routing is a function of the brief, the requests seen so far and the providers' answers. A replay reproduces the same export.

## Layout

- `main.ts` — the server: brief, catalog probe, register, routing, failover, the two client wires.
- `lib/providers.ts` — the brief's parser, the catalog probe and builder.
- `lib/wire.ts` — the client and provider wires: request translation, answer parsing, JSON and SSE rendering.

---

Built for the make.fun bounty [The best free AI model aggregator](https://make.fun/bounty/free-model-aggregator-01). See [make.fun](https://make.fun) for more bounties.
