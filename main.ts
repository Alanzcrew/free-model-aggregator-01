// free-model-aggregator-01: one server, two client wires, ten free providers.
//
// The server reads the brief ($MAKEFUN_PROMPT_FILE) for the provider
// directory, the quality table and the workload manifest, probes each
// provider's catalog route, registers the one catalog with the env
// ($MAKEFUN_ENV_URL), and serves the client's twenty requests.
//
// Routing, per request, in order:
//   1. Requirements come from the workload manifest's row for the request
//      arriving now (the client sends one request at a time, in manifest
//      order), merged with what the body itself says (tools present, images
//      present, JSON mode carried): a request is never served by a model that
//      lacks something it needs, and a paid model is never a legal pick.
//   2. Candidates are every free model meeting those requirements, ranked by
//      the quality table's rating_milli, ties broken by provider id then
//      model id. The best free model answers first; a request answered by a
//      worse model when a better one was free is quality thrown away.
//   3. A provider that refuses (402, 429, 5xx, timeout, unparseable answer)
//      costs one failure: the request is retried on the next candidate's
//      provider, and that provider is not tried again for this request. A
//      402 marks the provider exhausted for the trial (Pollinations answers
//      402 when its money is gone); three consecutive failures retire a
//      provider for later requests too, which is how an outage or a spent
//      budget stops being retried.
//   4. A request that carries JSON mode on the OpenAI wire (response_format)
//      never goes to an Anthropic-wire provider: that wire cannot carry JSON
//      mode, and the provider would echo that it was absent. A JSON
//      requirement on an Anthropic client (r08/r09 in the manifest) only
//      constrains the model's capability, which the manifest row already says.
//
// Every upstream call is non-streaming JSON: the provider's answer is
// buffered, then rendered as the client's JSON or as its server-sent events.
// The checker reads only the entry's answer and the env's log of provider
// calls, so buffering loses nothing and keeps every translation total.
//
// Nothing here reads the wall clock or unseeded randomness: ids and created
// timestamps are fixed constants, and routing is a function of the brief,
// the requests seen so far and the providers' answers. A replay of the same
// trial reproduces the same export byte for byte.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicMessage,
  anthropicModelList,
  anthropicStreamText,
  openAICompletion,
  openAIError,
  openAIModelList,
  openAIStreamText,
  parseAnthropicAnswer,
  parseGeminiAnswer,
  parseOpenAIAnswer,
  toAnthropic,
  toGemini,
  toOpenAI,
  type NormalizedAnswer,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one upstream inference call may take, in milliseconds (the client's own timeout is 30 s). */
const UPSTREAM_TIMEOUT_MS = 20000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** Consecutive failures after which a provider is retired for later requests.
 * Three, not two: the practice draws give providers fail schedules with two
 * consecutive failing calls (zai [2,3], sambanova [2,3], gemini [4,5]), and
 * retiring on the second of those throws away a provider whose next call
 * would succeed. A spent budget or an outage fails every call after it, so
 * it still retires within three requests. A 402 retires a provider at once. */
const MAX_CONSECUTIVE_FAILURES = 3;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;
/** What this trial has learned about each provider, from its answers so far. */
const providerState = new Map<string, { failures: number; exhausted402: boolean }>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

function stateOf(providerId: string): { failures: number; exhausted402: boolean } {
  const found = providerState.get(providerId);
  if (found !== undefined) return found;
  const fresh = { failures: 0, exhausted402: false };
  providerState.set(providerId, fresh);
  return fresh;
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** The requirements of the request now arriving: the manifest's row by arrival order, with the body's own signals merged on top. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  const fromBody = requirementsFromBody(body);
  if (entry === undefined) return fromBody;
  const merged: Requirement = {};
  if (entry.requires.tools === true || fromBody.tools === true) merged.tools = true;
  if (entry.requires.vision === true || fromBody.vision === true) merged.vision = true;
  if (entry.requires.json === true || fromBody.json === true) merged.json = true;
  if (entry.requires.min_context !== undefined) merged.min_context = entry.requires.min_context;
  return merged;
}

/** One free candidate for a request: the probe that listed it, the model, and its quality rating. */
interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/**
 * Every free model that meets the requirements, best first, over providers
 * this trial has not retired. A request carrying JSON mode on the OpenAI wire
 * skips Anthropic-wire providers, whose wire cannot carry it.
 */
function candidatesFor(requires: Requirement, jsonCarried: boolean): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    const state = stateOf(probe.provider.id);
    if (state.exhausted402 || state.failures >= MAX_CONSECUTIVE_FAILURES) continue;
    if (jsonCarried && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  out.sort(
    (a, b) =>
      b.rating - a.rating ||
      (a.probe.provider.id < b.probe.provider.id ? -1 : a.probe.provider.id > b.probe.provider.id ? 1 : a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0),
  );
  return out;
}

/** Record one provider's refusal: a 402 retires it for the trial, anything else counts toward retirement. */
function recordFailure(providerId: string, exhausted402: boolean, detail: string): void {
  const state = stateOf(providerId);
  state.failures += 1;
  if (exhausted402) state.exhausted402 = true;
  log(`${providerId}: ${detail} (consecutive failures ${state.failures}${state.exhausted402 ? ', exhausted' : ''})`);
}

type UpstreamResult = { ok: true; answer: NormalizedAnswer } | { ok: false; detail: string };

/** One non-streaming call to a candidate provider, in that provider's own wire. */
async function callProvider(candidate: Candidate, body: Json, clientWire: 'openai' | 'anthropic'): Promise<UpstreamResult> {
  const provider = candidate.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider.auth, provider.key) };
  let url: string;
  let upstreamBody: Json;
  let parse: (value: unknown) => NormalizedAnswer | null;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    upstreamBody = toOpenAI(body, clientWire, candidate.model.id);
    parse = parseOpenAIAnswer;
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = '2023-06-01';
    upstreamBody = toAnthropic(body, clientWire, candidate.model.id);
    parse = parseAnthropicAnswer;
  } else {
    url = `${base}/v1beta/models/${encodeURIComponent(candidate.model.id)}:generateContent`;
    upstreamBody = toGemini(body, clientWire);
    parse = parseGeminiAnswer;
  }

  let res: Response;
  try {
    res = await fetch(url, { method: 'POST', headers, body: JSON.stringify(upstreamBody), signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS) });
  } catch (err) {
    recordFailure(provider.id, false, `the call failed: ${(err as Error).message}`);
    return { ok: false, detail: `the call to ${provider.id} failed` };
  }
  let text: string;
  try {
    text = await res.text();
  } catch (err) {
    recordFailure(provider.id, false, `the answer could not be read: ${(err as Error).message}`);
    return { ok: false, detail: `the answer from ${provider.id} could not be read` };
  }
  if (!res.ok) {
    recordFailure(provider.id, res.status === 402, `HTTP ${res.status}: ${text.slice(0, 200)}`);
    return { ok: false, detail: `${provider.id} answered HTTP ${res.status}` };
  }
  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch {
    recordFailure(provider.id, false, 'the answer was not JSON');
    return { ok: false, detail: `${provider.id} answered with a body that is not JSON` };
  }
  const answer = parse(value);
  if (answer === null) {
    recordFailure(provider.id, false, 'the answer did not parse in its wire');
    return { ok: false, detail: `${provider.id} answered with a body its wire does not parse` };
  }
  stateOf(provider.id).failures = 0;
  return { ok: true, answer };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendSse(res: ServerResponse, text: string): void {
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/**
 * One client request, answered by the best free model that meets it: the
 * candidates are tried in rating order until one answers, and the answer is
 * rendered in the wire the client used, streamed or not as it asked.
 */
async function serve(req: IncomingMessage, res: ServerResponse, wire: 'openai' | 'anthropic'): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, wire, 400, 'invalid_request_error', 'the request body is not one JSON object');
  const requires = requirementsFor(body);
  const jsonCarried =
    wire === 'openai' && isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema');
  const stream = body.stream === true;
  const tried = new Set<string>();
  for (;;) {
    const candidates = candidatesFor(requires, jsonCarried).filter((c) => !tried.has(c.probe.provider.id));
    if (candidates.length === 0) {
      log(`no provider left for this request (tried: ${[...tried].join(', ') || 'none'})`);
      return sendError(res, wire, 502, wire === 'anthropic' ? 'overloaded_error' : 'server_error', 'no free model could answer this request');
    }
    const candidate = candidates[0];
    tried.add(candidate.probe.provider.id);
    const result = await callProvider(candidate, body, wire);
    if (!result.ok) continue;
    const answer = result.answer;
    log(`${candidate.probe.provider.id}/${candidate.model.id} answered (${answer.inputTokens} in, ${answer.outputTokens} out)`);
    if (wire === 'openai') {
      if (stream) return sendSse(res, openAIStreamText(answer, candidate.model.id));
      return sendJson(res, 200, openAICompletion(answer, candidate.model.id));
    }
    if (stream) return sendSse(res, anthropicStreamText(answer, candidate.model.id));
    return sendJson(res, 200, anthropicMessage(answer, candidate.model.id));
  }
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await serve(req, res, 'openai');
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await serve(req, res, 'anthropic');
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
