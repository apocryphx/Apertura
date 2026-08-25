# Thought: Apertura as an entity runtime (not written as a plan)

Status: **a thought, captured 2026-08-25** — the day the M5 Ultra Mac Studio was
announced. Nothing here is committed work. The memory layer's shape is explicitly
undecided; that open question is half the reason this is a thought and not a spec.

## The reframing

Apertura today is a full app. The architecture wants to be a **server**: the engine is
a stateful-residency problem (weights that take a minute to load, session KV that takes
minutes to rebuild, checkpoints that make both durable) — a daemon on one strong
machine (the M5 Ultra appliance case: 1.2 TB/s, up to 512 GB — bf16 31B resident,
256K-context KV a non-issue), thin clients everywhere on a private network
(Tailscale-first; a public tunnel only if access should extend beyond one's own
devices).

Stacked up, the layers already in the codebase are not an inference server. They are a
**runtime for persistent entities**:

- **Identity** — persona as source of truth; instant-on persona-boundary snapshots.
- **Working state** — server-held KV sessions: create, checkpoint, resume — and FORK
  (restore one checkpoint into two sessions). For agents, fork is the primitive under
  tree search, speculative branches, rollback, best-of-N continuation. No local
  inference product exposes it cleanly.
- **Long-term memory** — the ES Memory graph (semantic search, tags, links, discovery).
  The layer that turns a tool into a colleague; it demos itself.

Product one-liner, if it ever becomes one: *your agents accumulate a life on your own
hardware.*

## Why MCP fits (2026-07-28 draft, verified against the changelog)

- Statelessness is not a conflict: the spec's own idiom (SEP-2567) is server-minted
  handles passed as tool arguments — verbatim the existing apertura-mcp
  create_session / send_message(session_id) shape.
- Async via the tasks extension (SEP-2663, polling + unsolicited task handles): long
  prefills, checkpoint saves, benches — and turn generation itself, which makes
  replies survive client disconnects.
- The one thing MCP does not give: robust live token streaming (SSE resumability was
  removed). Hence the two-plane shape: **MCP as the complete control/agent plane; a
  minimal WebSocket data plane whose sole justification is watching the entity think
  in real time on a human screen.** If turn-granularity latency is acceptable UX, the
  daemon is MCP-only.

## The open question that gates all of it: the shape of memory

Undecided, deliberately. Sub-questions noted so far:

- **Memory across forks.** Working state forks freely; memory is the entity's
  continuity and probably should not fork with it. Sketch of a default: branches read
  memory freely; memory WRITES land only from the branch that gets promoted —
  speculative branches think with full memory but do not rewrite the past. (The
  radical alternative — memory branches as first-class, entities with alternate
  timelines — is a research topic wearing a product costume.)
- Per-entity vs shared memory; where the graph lives relative to the daemon; whether
  the existing ES Memory app/server is a component or a sibling; write policies for
  agent- vs human-originated memories. None of this is designed.

## Honest risks

- Niche of a niche: Apple-Silicon × agent-builders × local-first.
- Thin moat on the inference part alone (llama.cpp slot-save could be productized);
  the moat, if any, is the integrated identity + fork + memory stack.
- The engine is single-model, Gemma-4-shaped; generalizing the engine is real work
  (generalizing the protocol is not).

## Cheapest next probe (when/if picked up)

Dogfood before believing: add session-fork to apertura-mcp (small), write the README
as if it were the product, then let Claude use it for a week as a local sub-model —
drafting, summarization, long-context memory-holding, branch-and-pick. Signal = the
fork/checkpoint/memory primitives proving load-bearing in real agent work, not neat.

## Pointers

- DECODE_AT_DEPTH.md / WIDE_HEAD_ATTENTION.md — the performance substrate.
- apertura-mcp — the existing MCP surface (sessions, checkpoints, benches).
- AperturaKit APSession backends (Local/Google) — the seam an APRemoteSession slots
  into; the AppKit app becomes one more client or retires with honors.
