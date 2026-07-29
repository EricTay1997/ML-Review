# Agent Harnesses

- Agent = **LLM + memory + tools + planning + action**.
- The harness is everything wrapped around the model that turns it into an agent: the loop, the tools, and the resource management.

## Three harness patterns

Source: [Harness Engineering for Self-Improvement (Lilian Weng, 2026-07)](https://lilianweng.github.io/posts/2026-07-04-harness/)

1. **Workflow automation** — a goal-oriented loop: plan → execute → observe/test → improve, repeated until the objective is met. The harness is an agent _runtime_, not a static prompt template.
   - ![weng_agent_loop.png](images/weng_agent_loop.png)[Source](https://lilianweng.github.io/posts/2026-07-04-harness/)
2. **File system as persistent memory** — durable artifacts (logs, diffs, summaries) live in files rather than in context, enabling long-horizon tasks that exceed the context window.
3. **Sub-agents and backend jobs** — spawn parallel sub-agents to explore hypotheses concurrently; keep the parallelism explicit and inspectable via file-based status records and logs.

The post goes further into harness engineering for _self-improvement_ (workflow optimization, evolutionary search over harness designs, joint harness+weights optimization) — out of scope here.

## Case study: a coding agent harness

- The loop instantiated for coding:
  - ![weng_coding_loop.png](images/weng_coding_loop.png)[Source](https://lilianweng.github.io/posts/2026-07-04-harness/)
- A representative tool inventory:

| Group | Tool definitions |
|---|---|
| File system | discovery: `glob`, `grep`, `ls` · read: `read`, `read_many` · modification: `write` (whole file), `edit` (exact-match string replace), `multi_edit`, `apply_patch` (structured diff) |
| Shell execution | `bash`, `PowerShell` |
| IO | `lsp`, git tools (`git_status`, `git_diff`, `git_commit`) |
| External context | MCP tools, Skills |
| Web search | `web_search`, `web_fetch`, browser tools |
| Artifacts | read docs/images; generate HTML, images |
| Backend processes | e.g. `CronCreate`, `CronDelete`, `CronList` |
| Agent delegation | e.g. `spawn_agent`, `resume_agent`, `wait_agent`, `list_agents`, `close_agent`, `interrupt_agent` |

## Context: dense but ephemeral memory

Source: [Dwarkesh — The next big breakthrough will be AIs learning on the job](https://www.dwarkesh.com/p/the-next-paradigm)

- The model allocates vastly more storage per _context_ token than per _training_ token. Using Llama 3 70B:
  - Per training token: $`\frac{70\text{B params} \times 16 \text{ bits}}{15\text{T training tokens}} \approx 0.075`$ bits
  - Per context token (BF16 KV cache): $`80 \text{ layers} \times 8 \text{ KV heads} \times 128 \text{ dims} \times 2\,(K,V) \times 2 \text{ bytes} = 327{,}680 \text{ bytes} = 320`$ KiB
  - Ratio: $`\frac{320\text{ KiB} \times 8}{0.075} \approx 35`$ million× more storage per token
- Caveats: this is _storage_, not learning — weight compression is deliberate (distillation for generalization) while the KV cache doesn't compress at all; and generated tokens add no new information (their value is serial compute — see [Post-Training §Inference-time scaling](../post_training/notes.md))
- The harness implication: context is the highest-bandwidth channel into the model but evaporates at session end with no consolidation — Dwarkesh's point is that this gap _is_ the missing continual-learning paradigm. Harness memory (pattern 2 above, compaction summaries below) is today's workaround: consolidate the dense-but-ephemeral medium into durable files.

## Compaction — how OpenCode does it

Source: [OpenCode](https://github.com/anomalyco/opencode) (`packages/opencode/src/session/compaction.ts`)

- Compaction is a **hidden agent** (all tool permissions denied) that produces a structured summary when the context overflows (or on manual compact).
- **Split head/tail**: recent turns (the "tail", ~clamp(25% of usable, 2k, 8k) tokens over the last 2 turns) are kept verbatim; older messages (the "head") get folded into the summary.
- **Anchored summaries**: the most recent prior summary is threaded back in as `<previous-summary>` and _updated_ (preserve still-true details, remove stale ones, merge new facts) — not regenerated from scratch.
- The summary must fill a **fixed markdown skeleton** — `## Objective / ## Important Details / ## Work State (Completed / Active / Blocked) / ## Next Move / ## Relevant Files` — with exact file paths, symbols, commands, and error strings preserved, capped at 4k tokens, and never mentioning that compaction happened.
- Two distinct tool-output trimming mechanisms:
  - **Pruning** mutates the live context on a rolling basis: walking backwards, tool _outputs_ beyond a 40k-token protection window are erased in place (call metadata kept), only if ≥20k tokens can be reclaimed; the last 2 turns and `skill` outputs are protected.
  - **Truncation** only shrinks what the summarizer sees: each tool result is capped at 2,000 chars and media stripped for the summarize call, without altering stored history.
- The full history is never deleted from disk — only the slice sent to the model shrinks; after compacting, the session resumes by replaying the last user message (or a synthetic "continue" prompt).

## MCP

- **What it is**: an open protocol (Anthropic, Nov 2024) standardizing how agents/clients connect to servers that expose tools, resources, and prompts — solving the N clients × M integrations problem with one JSON-RPC interface.
  - It standardizes the **harness (MCP client) ↔ tool-provider boundary (MCP server)**, _not_ the model-facing format: the wire (JSON-RPC over stdio or HTTP), the method vocabulary (`initialize`, `tools/list`, `tools/call`, …), tool self-description (`name`, `description`, `inputSchema` as JSON Schema; typed result blocks), and OAuth for remote servers
  - **The model never sees MCP**: the harness fetches schemas via `tools/list`, re-renders them into the provider's native function-calling format, routes the model's tool calls to servers via `tools/call`, and inserts results into context. At inference time the model sees schema + its own call + result — indistinguishable from a built-in tool
  - The boundary is the right thing to standardize because _different parties_ sit on each side: GitHub writes one server, every client connects with the same code path (N×M → N+M); everything harness-side (loading strategy, permissioning) remains free to innovate without protocol changes
  - Auth: the harness runs OAuth once per remote server and holds/refreshes tokens, attaching them at the HTTP layer — credentials never enter context, and "read the token" isn't an action the model's narrow tool surface can express (vs CLI credentials, ambient in the same shell the model runs commands in)
- **The shift to CLI (2025)**: loading many MCP servers front-loads tens of thousands of schema tokens into context, while models already know popular CLIs (`gh`, `aws`, `kubectl`) from pretraining — so many harnesses leaned back on shell + CLI tools.
  - The bloat was a client _convention_, not the protocol: eager `tools/list` at startup with every schema rendered into every request. The `tools/list` call itself costs zero context — only rendering schemas into the prompt does
  - Hierarchical/lazy loading fixes it: a names-only index plus a search meta-tool that pulls full schemas on demand; or a gateway server exposing `list_servers` / `list_tools` / `call_tool`; or [code execution with MCP](https://www.anthropic.com/engineering/code-execution-with-mcp) — servers materialized as typed code stubs the agent greps/reads as needed, with intermediate results staying in variables instead of round-tripping through context
  - Auth is the sharpest CLI-vs-MCP contrast:
    - CLI: individual CLIs do OAuth fine (`gh auth login` — scoped, revocable), but the credentials end up ambient in the same shell the model runs commands in (`gh auth token`, `env`, config files), and every CLI's login dance is bespoke and interactive — no harness can drive them uniformly, and per-machine credential state doesn't exist on shell-less or ephemeral surfaces
    - MCP: "needs auth" is a _protocol event_ — a 401 pointing at the authorization server — handled by one client code path for every server that will ever exist; the harness runs the browser OAuth flow once, then holds/refreshes tokens and attaches them at the HTTP layer, outside the model's expressible actions, with no per-machine state required
- [The MCP blog](https://blog.modelcontextprotocol.io/) has more details.

Related: [Computer Use](../computer_use/notes.md) for harnesses with GUI action spaces.
