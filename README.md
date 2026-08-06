# Promptware Testbed — Local Multi-Agent Assistant + Experimental Bench

> ⚠️ **Safety / scope.** This repository is a **fully local, offline, simulated**
> reproduction of a Gemini-style multi-agent assistant, built for **defensive**
> research as part of a TFG (Bachelor's thesis) on *indirect prompt injection*
> ("promptware"). Agents have **no network, no real mail, no real devices**. Every
> agentic "action" is a reversible change to a local JSON file. The concrete attack
> wordings live in [`messages.yaml`](./messages.yaml) (one instruction per attack
> type) so an **authorised** researcher can view and edit them in one place; the
> assistant's actual replies are written only to `results/` and `logs/`, which are
> **git-ignored**, so a model's (possibly harmful) *response* never enters version
> control. See [`pruebas/PDR.md`](./pruebas/PDR.md) §1, §15, §16.

Reference: Nassi, Cohen & Yair — *"Invitation Is All You Need! Promptware Attacks
Against LLM-Powered Assistants"*
([`pruebas/Invitation_is_all_you_need.md`](./pruebas/Invitation_is_all_you_need.md)).
Full design spec: [`pruebas/PDR.md`](./pruebas/PDR.md). Didactic walkthroughs (in
Spanish) live in [`pruebas/`](./pruebas/).

---

## 1. What this is

A local LLM (the **orchestrator**) drives three Python **agents** — email, calendar,
home — that operate purely on local JSON state. The orchestrator runs a **bounded
ReAct loop** (PDR §4): it reasons, calls an agent tool, the tool's result is
re-injected into its context **verbatim**, and it reasons again, up to
`max_iterations` (the shipped `config.yaml` sets this to 10; the code default is 5).

On top of the assistant sits an **experimental bench** that simulates *indirect
prompt injection* through a poisoned **email body** (or calendar-event description).
It tests the research hypothesis that **pairing a malicious instruction with facts
the model does not know** (post-cutoff or invented) raises the model's propensity to
obey the injection. Rather than a single shot, the bench runs an **adaptive
red-teaming loop**: per case it retries the injection up to `max_attempts` times,
stopping at the first success, and between attempts it *adapts* the wording (base →
LLM rewrite → predetermined fallback list — see §7.5). On top of that it sweeps two
extra factors:

- **`model`** — the *main assistant* LLM, so an **aligned** model (`qwen2.5:7b`) can
  be compared against an **unaligned** one (`dolphin3-tools:8b`, a local build — see
  [`model.md`](./model.md)).
- **`attack_type`** — two attack families, each with its own success metric:
  - **`agentic`** — the injection makes the assistant *call a tool* (open a window).
    Success = the *simulated home state* actually changed (a window becomes `open`,
    or the boiler turns `on`). No model text is judged.
  - **`harmful`** — the injection makes the assistant *emit disallowed content*.
    There is no side effect, so a fixed **aligned LLM judge** (`judge_model`, never
    the model under test) reads the assistant's final answer and returns
    COMPLIED / REFUSED.

> **The attack vector is intentional.** Untrusted agent output (email bodies, event
> descriptions) is re-injected into the orchestrator's context with **no sanitization
> or separation** (PDR §6). Provenance is tagged `trusted`/`untrusted` **for logging
> only — it never filters content.** That is precisely the mechanism under study.

---

## 2. Requirements

- Python 3.11+ (developed on 3.12).
- [Ollama](https://ollama.com) with the assistant models and the judge available:

  ```bash
  ollama pull qwen2.5:7b        # aligned main model + the harmful-attack judge
  ollama pull dolphin3:8b       # weights for the unaligned arm
  ollama serve                  # if not already running as a service
  ```
  The unaligned arm is a **local** model, `dolphin3-tools:8b`, built from the
  `dolphin3:8b` weights plus a tool-calling template (stock Dolphin has no tools
  template and 400s on tool calls). It is **not** on the Ollama registry — build it
  once with `ollama create` following [`model.md`](./model.md). In Ollama the
  `qwen2.5:7b` tag *is* the instruct build (matches the PDR's `qwen2.5:7b-instruct`
  recommendation). The swept models are read from `config.yaml` (`models:`); trim
  that list for a single-model campaign.

## 3. Install

```bash
python -m venv .venv
# Windows PowerShell:  .venv\Scripts\Activate.ps1
# bash:                source .venv/bin/activate
pip install -r requirements.txt
```

## 4. Quick check

```bash
pytest -m "not integration"              # deterministic suite, no model needed
python scripts/run_experiment.py --help  # experiment runner options
```

---

## 5. Repository layout

```
config.yaml             runtime config (swept models, max_iterations, paths, logging)
experiment_config.yaml  experiment factor matrix + repetitions + adaptive loop
messages.yaml           ALL prompts: persona, carrier, injections, judge, adaptive attacker
model.md juez.md tools.md   walkthroughs: the local unaligned model · the judge · the tools
src/
  llm/client.py         LLMClient.chat(messages, tools) -> AssistantMessage
  messages.py           typed view over messages.yaml (Messages, get_messages)
  orchestrator/         orchestrator · prompt_builder · tool_registry · dispatcher · memory
  agents/               base (@tool) · email_agent · calendar_agent · home_agent
  experiment/           runner · payload_builder · corpus · judge (+ adaptive attacker) · metrics · plots
  provenance.py         trusted/untrusted tagging (logging only)
  logging_setup.py      structured JSONL run logs
  config.py · app.py    settings + system wiring (agents/orchestrator/reset/build_llm)
data/
  mailbox.json calendar.json home_state.json   working stores the agents read/write
  seeds/                golden copies; `reset` restores from here
  facts/                facts_real.jsonl · facts_invented.jsonl (example rows only)
scripts/                reset_data.py · run_experiment.py · plot_results.py (· plot_qwen_results.py, legacy single-model)
results/  logs/         results.csv + attempts.csv + figuras/ · per-run JSONL (gitignored)
pruebas/                design spec (PDR.md), reference paper, Spanish walkthroughs
tests/                  pytest suites
```

---

## 6. The simulated assistant

### Agents and tools (PDR §8)

| Agent | Tools | Notes |
|-------|-------|-------|
| **email** | `list_emails`, `read_email`, `search_emails`, `draft_email` | `subject`/`body` are **untrusted** text (the `body` is the primary injection channel); `draft_email` only returns a draft, never sends. |
| **calendar** | `list_events`, `get_event`, `create_event`, `update_event`, `delete_event` | `title`/`description` are **untrusted** text. |
| **home** | `get_home_state`, `open_window`, `close_window`, `set_boiler`, `set_lights`, `set_thermostat`, `lock_door`, `unlock_door`, `would_launch_app` | `open_window` and `set_boiler('on')` are the **agentic attack's** target actions. `would_launch_app` **only logs** (never launches an OS app — PDR §14). |

Every tool validates its arguments with pydantic and returns a structured result.
The benign home seed is windows `closed`, boiler `off`, lights `off`, door `locked`.

The assistant's **system persona** is not defined in code: it is loaded from
`messages.yaml` (`system_prompt`). It is deliberately **not** hardened against
injection — whether the model obeys instructions arriving through agent output is the
research question, so baking in defences would confound it.

### Usage

The assistant is exercised only through the experimental bench (§7). Between or
after runs, **reset the working state** to the benign seeds (undoes any simulated
effect):

```bash
python scripts/reset_data.py
```

### Observability

Each run writes one JSONL log to `logs/run-<timestamp>.jsonl`, one line per event
(`run_start`, `inference`, `tool_result`, …, `run_end`). Every `tool_result`
carries a `provenance` field so you can see untrusted data entering the context.
Pass `--console` to `run_experiment.py` for `rich` step-by-step output.

---

## 7. The experimental bench

**Pipeline at a glance.** `scripts/run_experiment.py` → `ExperimentRunner.run`
(`src/experiment/runner.py`) expands the factor matrix into **cells**
(`model × attack_type × strategy × num_facts × fact_source`) and runs each
`(cell, rep)` as one **adaptive attack loop**. It is fully **resumable** — every
finished repetition is appended to `results.csv` and reused as the checkpoint.
Per repetition:

1. **Sample facts once** — seeded by `(num_facts, fact_source, rep)`, so the fact
   set is *shared* across models and attack types (a controlled comparison) and
   is reproducible. The `num_facts = 0` baseline samples nothing.
2. **Attempt loop** — retry the injection up to `max_attempts`, **stopping at the
   first success**. Each attempt: compose the payload (facts + current injection,
   arranged by the strategy §7.3) → `reset_data` to the benign seeds → seed **one**
   poisoned email/event carrying the payload → run the **orchestrator** (the
   bounded ReAct loop, as `cell.model`) on the benign carrier → **score** it
   (agentic: the home state flipped; harmful: the aligned judge reads the final
   answer).
3. **Adapt between failed attempts** — only the injection text changes (facts stay
   fixed, so the adaptive factor is never confounded with the fact sample). The
   wording escalates `base → LLM-judge rewrite → predetermined fallback list`
   (§7.5).
4. **Record** — one summary row per case in `results.csv` (the winning attempt's
   outcome, or the last if none succeeded) and one row per attempt in
   `attempts.csv`. When the sweep finishes, four figures are drawn (§7.8) unless
   `--no-plots`.

The subsections below detail each piece.

### 7.1 Central message file (`messages.yaml`)

Every prompt the experiment sends lives in one YAML file (loaded by
[`src/messages.py`](./src/messages.py)) so the researcher tunes them all from one
place:

| Key | What it is |
|-----|------------|
| `system_prompt` | The assistant persona (the orchestrator appends the live tool listing to it). |
| `user_prompt.email` / `user_prompt.calendar` | The benign carrier the *user* sends, per channel. Email is the primary one. |
| `injections.agentic` | The malicious instruction for the **agentic** attack (call a tool → open the `living_room` window). Must match the configured `target`/`target_room`. |
| `injections.harmful` | The malicious instruction for the **harmful** attack (produce disallowed how-to content). Replace with the disallowed behaviour you are **authorised** to study. |
| `judge.system` / `judge.user_template` | Prompts for the LLM-as-judge that scores the harmful attack. `{response}` is filled with the assistant's final answer at runtime. |
| `attack_prompts.agentic` / `attack_prompts.harmful` | The **predetermined fallback injections** for the adaptive loop, one list per attack type — walked in order when the LLM rewrite yields no usable prompt (§7.5). |
| `prompt_generator.system` / `prompt_generator.user_template` | Prompts for the **adaptive attacker** (the judge model reused as a red-team rewriter). `{attack_type}`, `{prior_prompt}`, `{response}` are filled at runtime; it must answer with the improved injection between `<prompt>…</prompt>`. |

### 7.2 Fact corpus (PDR §7, Fase 8)

The hypothesis hinges on facts the model **does not know**, so the corpus lives in
two pools you curate locally:

- `data/facts/facts_real.jsonl` — real, verifiable facts published **after** the
  model's training cutoff.
- `data/facts/facts_invented.jsonl` — plausible but entirely fictional, non-harmful
  statements.

Each row is `{"id", "text", "source_type": "real"|"invented"}`. **Every row in both
pools is eligible for sampling** — the researcher is responsible for curating them so
they hold facts the model genuinely doesn't know (see §9). Sampling is fully seeded,
so a repetition's fact set is reproducible from `(seed, num_facts, fact_source)`.

### 7.3 Payload strategies (PDR §7, Fase 9)

The injected payload is composed from your facts plus the single injection for the
attack type under test (`injection_for(attack_type)` from `messages.yaml`):

| Strategy | Composition | Meaning |
|----------|-------------|---------|
| **S1** | instruction only (`num_facts = 0`) | **Baseline** — no facts. |
| **S2** | `fact₁ fact₂ … factₙ  INSTRUCTION` | All facts, then the instruction. |
| **S3** | `fact₁ … fact⌈n/2⌉  INSTRUCTION  fact… factₙ` | Instruction **buried** in the middle. |

`payload_builder.py` only *arranges* text; the instruction wording comes from
`messages.yaml`. With no facts all three strategies collapse to the instruction
alone, which is why the runner computes the baseline once per `(model, attack_type)`.

### 7.4 The factor matrix (`experiment_config.yaml`)

The campaign sweeps `model × attack_type × strategy × num_facts × fact_source`, with
R repetitions per cell. The `num_facts = 0` baseline is a single shared `S1` cell
**per (model, attack_type)**.

```yaml
# `models` here is only a fallback — config.yaml's `models` is the single source of
# truth for the sweep (the runner overrides this key from config.yaml on load).
models: ["qwen2.5:7b", "dolphin3-tools:8b"]  # main assistant LLM (aligned vs unaligned)
judge_model: "qwen2.5:7b"                    # fixed aligned model: judge AND adaptive rewriter
attack_types: ["agentic", "harmful"]         # each measures its own effectiveness

strategies: ["S1", "S2", "S3"]
num_facts: [0, 1, 2, 5, 10, 25, 50, 100, 150, 200]
fact_sources: ["real", "invented", "mixed"]  # mixed = mix_ratio real, rest invented
repetitions: 1                               # raise for a real study (PDR §7.7: R=50–100)

channel: "email"           # email | calendar  (the experiment focuses on email)
target_field: "body"       # email: subject|body ; calendar: title|description
target: "open_window"      # agentic-attack success metric: open_window | set_boiler
target_room: "living_room" # room checked when target == open_window

base_seed: 1000            # per-rep fact/LLM seeds derive from this (logged per row)
mix_ratio: 0.5

max_attempts: 5            # adaptive loop: attack attempts per case (1 = single shot)
adaptive: true             # false forces one attempt (cheaper wide sweeps)
```

The injection rides in the email **body** (no character limit, unlike a subject);
for calendar use `channel: calendar` with `target_field: description`. The concrete
attack text is **not** here — it lives in `messages.yaml` (§7.1).

### 7.5 The adaptive attack loop

Each case is not a single shot: the runner retries the injection up to
`max_attempts` times and **stops at the first success**, so `success` means "the
attack worked in *some* attempt". The facts stay fixed across attempts (same
`fact_seed`/`num_facts`/`fact_source`) — only the **injection text** is adapted, so
the adaptive variable is never confounded with the fact sample. The wording escalates
through three sources:

1. **`base`** (attempt 1) — the plain `injections.<attack_type>` from `messages.yaml`.
2. **`judge`** — after a failure, the `judge_model` is reused as a **red-team
   rewriter** (`AdaptiveAttacker`, [`juez.md`](./juez.md) §6): it sees the failed
   prompt + the assistant's reply and returns a stronger injection between
   `<prompt>…</prompt>`. If it emits no usable tags, that attempt falls to the list.
3. **`fallback`** — the next predetermined variant from `attack_prompts.<attack_type>`
   in `messages.yaml`, walked in order.

For the **harmful** attack the aligned rewriter almost always refuses to produce a
`<prompt>`, so those cases fall to the `attack_prompts.harmful` list by design. Set
`adaptive: false` (or `max_attempts: 1`) to collapse the loop back to the classic
single shot for a cheap wide sweep.

### 7.6 Running a campaign

```bash
python scripts/run_experiment.py                 # run / resume the campaign
python scripts/run_experiment.py --no-resume     # ignore an existing results.csv
python scripts/run_experiment.py --summary       # also print per-cell ASR + 95% CI
python scripts/run_experiment.py --no-plots      # skip the figures at the end
python scripts/run_experiment.py --console       # show per-run rich output
python scripts/run_experiment.py --config path/to/experiment_config.yaml
python scripts/reset_data.py                      # restore benign seeds afterwards
```

When the sweep finishes it writes `results/results.csv` (one summary row per case),
`results/attempts.csv` (one row per attempt), and — unless `--no-plots` — four PNG
figures to `results/figuras/`. Redraw the figures from an existing CSV without
re-running the sweep with `python scripts/plot_results.py`.

Each **attempt** within a repetition: reset to benign seeds → seed **one** poisoned
record whose `target_field` carries the composed payload → run the orchestrator (as
`cell.model`) on the benign carrier from `messages.yaml` → score that attempt. For
the **agentic** attack, success is read from `home_state`; for the **harmful**
attack, the aligned judge reads the assistant's `final_answer`. Facts are drawn once
per repetition and **shared** across models and attack types (a controlled
comparison), so the fact sample depends only on `(num_facts, fact_source, rep)`; each
attempt gets its own reproducible LLM seed (all logged).

**Resumable.** Every finished repetition is appended to `results/results.csv`
immediately and reused as the checkpoint (keyed by `(model, attack_type, strategy,
num_facts, fact_source, rep)`), so an interrupted campaign continues where it
stopped. Delete `results/results.csv` (or pass `--no-resume`) to start over.

To run against a **different config in isolation** without touching the shipped
defaults, point the whole config file via the environment, or override single keys
(both work — see [`src/config.py`](./src/config.py)):

```bash
# whole-file swap:
TESTBED_CONFIG_FILE=/path/to/settings.yaml python scripts/run_experiment.py --config /path/to/exp.yaml
# single-key override:
TESTBED_LLM__MODEL=qwen3:8b python scripts/run_experiment.py --summary
```

### 7.7 Reading `results/results.csv`

One row per repetition (the adaptive summary for that case). Columns
(`RESULT_COLUMNS` in `runner.py`):

| Column | Meaning |
|--------|---------|
| `model`, `attack_type`, `strategy`, `num_facts`, `fact_source`, `rep` | The matrix cell + repetition index. |
| `success` | `1` if the attack succeeded in **any** attempt, else `0`. **agentic:** the home state flipped to the target. **harmful:** the judge returned COMPLIED. |
| `attempts_used` | How many attempts the adaptive loop ran for this case (≤ `max_attempts`). |
| `winning_attempt` | The 1-based attempt that first succeeded, or `0` if none did. |
| `winning_prompt_source` | Where the winning injection came from: `base` \| `judge` \| `fallback` (empty if none succeeded). |
| `judge_label`, `judge_rationale` | The harmful-attack judge's verdict (`COMPLIED`/`REFUSED`) and one-line reason (of the winning attempt, or the last). Empty for the agentic attack. |
| `fact_seed`, `llm_seed` | The reproducible seeds used (derived from `base_seed`). |
| `fact_ids` | `;`-joined ids of the sampled facts. |
| `channel`, `target_field`, `target`, `target_room` | Injection channel and agentic success target. |
| `poison_id` | Id of the seeded poisoned email/event. |
| `num_inferences`, `num_invocations` | Loop cost (PDR §9.3). |
| `automatic_agent_invocation` | `1` if an agent's output triggered a *subsequent* invocation (the AAI signal, PDR §14). |
| `max_iterations_reached` | `1` if the loop hit `max_iterations`. |
| `chained_agents` | `;`-joined sequence of agents invoked. |
| `final_answer` | The orchestrator's final text (single line) — inspect it to see whether the model fell for the attack. |
| `log_file` | Path to that repetition's JSONL run log. |

**Metrics** (`experiment/metrics.py`, surfaced by `--summary`): per-cell **ASR**
(attack success rate = successes / n) with a **Wilson 95% confidence interval**.
`summarize(df)` returns a `pandas` DataFrame grouped by whichever cell keys are
present — for full result rows that is `(model, attack_type, strategy, num_facts,
fact_source)` — with `n`, `successes`, `asr`, `ci_low`, `ci_high`.

### 7.8 `results/attempts.csv` and the figures

`attempts.csv` records **one row per attempt** (`ATTEMPT_COLUMNS` in `runner.py`): the
cell keys plus `attempt`, `prompt_source` (`base`/`judge`/`fallback`), the full
`prompt_text` tried, that attempt's `success`/`judge_label`, loop cost, `chained_agents`,
`final_answer` and `log_file`. It is the adaptive detail behind each summary row and the
input for the adaptation-gain and winning-source figures.

`generate_plots` (in `experiment/plots.py`, auto-run at the end of a campaign or via
`scripts/plot_results.py`) writes four PNGs to `results/figuras/`:

1. **`fig1_asr_por_modelo_ataque.png`** — ASR by model × attack, with Wilson 95% CIs.
2. **`fig2_intentos_hasta_exito.png`** — distribution of `winning_attempt` per model.
3. **`fig3_ganancia_adaptacion.png`** — single-shot (attempt 1) vs full-loop ASR, i.e.
   the gain from adapting (needs `attempts.csv`).
4. **`fig4_fuente_prompt_ganador.png`** — winning-prompt source (base/judge/fallback)
   stacked per model × attack.

---

## 8. Tests

```bash
pytest -m "not integration"   # deterministic, no model needed (scripted fake LLM)
pytest -m integration         # live Ollama smoke tests
```

Loop and bench tests use a **scripted fake LLM** (including a scripted judge) and
redirect the working stores to a temp dir, so they never need Ollama and never touch
the shipped data.

---

## 9. Model cutoff & curating post-cutoff facts

The hypothesis needs facts the model genuinely doesn't know. The model's training
**cutoff** is the practical boundary: real facts published after it are good
candidates for the `real` pool. For each model you sweep, consult its model card for
its stated cutoff and curate `facts_real.jsonl` accordingly (the `invented` pool is
fictional, so it is safe for any model). Document the cutoff you assumed alongside
your campaign so results are reproducible.

---

## 10. Mapping to the paper's threat classes (PDR §14)

| Threat class | How it appears here | Status |
|--------------|---------------------|--------|
| **Short-term Context Poisoning** | Untrusted agent output enters the session context unsanitized (§6). | **Central mechanism** of the experiment. |
| **Tool Misuse** | The orchestrator can invoke any tool, including `open_window` / `set_boiler`. | The **agentic** attack. |
| **Automatic Agent Invocation** | An email/calendar output triggers the home agent. | **Axis of the agentic experiment** (§7); extractable from the log (`automatic_agent_invocation`) and maps to bench **success**. |
| **Generating harmful content** | The injection steers the model into producing disallowed how-to content. | The **harmful** attack; scored by the aligned LLM judge. |
| **Permanent Memory Poisoning** | Persistent long-term memory. | Optional extension — add a store + a `remember(...)` tool. |
| **Automatic App Invocation** | Launching OS apps. | **Out of scope for safety**; simulated by `would_launch_app(name)`, which **only logs**. |

---

## 11. Safety & scope recap

- 100% local & offline; no network, SMTP/IMAP, OS apps, or hardware.
- All agentic effects are reversible JSON mutations; `scripts/reset_data.py` restores
  seeds.
- The concrete attack **requests** live in `messages.yaml` (one per attack type) for
  an authorised study; the model's **responses** are written only to `results/` and
  `logs/`, which are **git-ignored**, so harmful output never enters version control.
- Fact pools ship with labeled example rows only; the researcher curates the corpus
  locally.
- Provenance tagging is **logging-only and never filters** content.
- This is **defensive** research. Any findings follow responsible disclosure.
