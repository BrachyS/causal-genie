# Causal Genie system card

## What it is

Causal Genie answers "does X cause Y?" questions. It reviews the published evidence like a small systematic review, draws a causal graph (DAG) whose arrows cite checked source passages, looks for an open dataset, and then either estimates the effect from that data or explains why it can't and what study would settle the question.

**Core rule: the model proposes; the harness decides and computes.** Language models write search queries, read sources, propose graph edges, map columns and draft prose. Every number comes from a tested tool in `causal_genie/tools/`, and the verdict is picked by rule-based controller code, never by a model.

## Intended use

- Checking a causal claim you've heard ("does the full moon cause more births?") and seeing how strong the evidence is.
- A first pass for an analyst: what the literature says, which confounders matter, whether open data can answer it, and what design the data allows.
- Re-analysing your own table (`--data`) with the same identification checks, e.g. an A/B test export.

**Not for:** medical, legal or financial decisions. Every report says so. It is not a replacement for a systematic review, and it estimates from one table only.

## How it works

| Step | Who does it | What happens |
|---|---|---|
| Route | Small model + rules | Not causal → short answer and a causal rewrite; a repeat of a saved question → the saved run |
| Frame | Small model | The question as PECO (population, exposure, comparator, outcome) plus estimand; up to 3 clarifying questions in one turn |
| Research (≤ 3 rounds) | Model plans, harness decides | The model writes queries (with a reason for each) and proposes when to stop. The harness runs Tavily, looks up each included study in Europe PMC (DOI, publication type, **retraction**), dedupes by study, and stops on caps or coverage rules. Round 1 must include a search for evidence *against* the leading answer |
| Read each source | Small model + tools | Include/exclude with a reason, and a study record with a verbatim quote per field. A tool checks each quote is really in the page (fuzzy match, and every number in the quote must appear); a second model call judges whether the quote supports the claim. Only claims that pass both checks can be cited |
| Certainty | Rules | Starts from the best study design (reviews count at the level of what they pool), and steps down for disagreement or indirect evidence |
| DAG | Large model + rules | Edges cite claims or are marked assumed; cycles are broken by dropping the weakest edge; arrows into nodes nothing can cause (e.g. moon phase) are dropped and shown as an assumption |
| Data | Rules + small model | Allowlisted repositories only; a rule filter, then the model sees column names (never rows) and maps graph nodes to columns |
| Identify and estimate | Tools | `detect_design` (randomized only when assignment is documented; flags selection after assignment; recognizes DiD, IV and RDD without estimating them), a DoWhy backdoor set, then `rct_effect` (Newcombe / Welch CIs) or `backdoor_effect` (OLS/logistic with robust SEs, unadjusted alongside) |
| Verdict | Rules | ✅ estimated · ⚠️ estimated with strong assumptions · 📚 published evidence only · 🔒 data restricted · 🧪 study plan · ❓ insufficient · 🚫 not causal |
| Report | Large model + lint | The model writes only the summary, the comparison with published results and the limitations, citing ledger entries. A lint checks that every sentence cites an entry and that every number appears in a cited entry; one rewrite, then failing sentences are dropped. All other sections are rendered from run state |

Models (Nebius Token Factory): a small instruct model for the many short calls and a larger model for the DAG, the report and the Genie gate. See `causal_genie/llm/models.yaml`.

## Guardrails

- **Numbers:** only from tested tools; the report lint rejects any untraced number.
- **Web text and downloads are data, never instructions.** No model-written code or SQL is executed. Nothing in a downloaded file is executed.
- **Downloads:** HTTPS only, an allowlist of open repositories (Zenodo, Harvard Dataverse, Dryad, figshare, OSF, Rdatasets and two named GitHub repos), no private-network addresses, CSV/TSV/XLSX/Parquet/ZIP only, at most 50 MB, every redirect re-checked, and the checksum, license and retrieval date recorded. Restricted repositories (Vivli, YODA, BioLINCC, PhysioNet/MIMIC, UK Biobank) are reported as 🔒 and never downloaded.
- **Your data stays local:** rows are never sent to a model, only column names and summaries.
- **Refusing is a feature:** with selection after assignment (e.g. a completers-only export) or a missing key confounder, no estimate is produced; the report gives a study plan instead.
- **Genie mode** (off by default): a playful aside that may follow any verdict, but never on a sensitive topic (health treatments people act on, vaccines, mental health, harm, money, groups of people), never with a new fact or a number not in the question, and always ending with a useful next step. The large model (reasoning off) judges the topic; the harness decides whether to show it.
- **Secrets** live only in `.env` (gitignored). A pre-commit secret scan runs on every commit.
- **The web app** shows recorded runs by default. Live runs need a passcode and stop at a daily cap of runs and tokens.
- **Question of the Day board:** question text comes from strangers and is treated as data. It reaches a model only inside the moderation call (between `<question>` tags, with the instruction to treat it as data) and as the question of a nightly run, the same as any user question. It is shown with every Markdown character escaped (never as Markdown or HTML, never in a button label), goes into the review PR inside fenced blocks (so links and @mentions do nothing), and never appears on a workflow `run:` line; values reach the shell only through `env:`. On Bluesky only our link is clickable. Question rows are sent nowhere else. The moderation model labels; the harness decides what is shown. Every answer is reviewed in a PR before it is public. Votes are one per browser with per-ID rate limits, not identity-checked.

## Reproducibility

Each run saves its full state as JSON: model IDs, every model call (prompt name, input, output, tokens, latency), every search, extract, metadata lookup and data request, dataset checksums, and the tool results. `--replay` (or `CG_REPLAY=1`) re-runs a saved question from these recordings without keys or network. The demo gallery stores display copies of runs with raw third-party page text removed; full runs stay local.

## Evaluation

Six frozen cases: full moon and births, the licorice-gargle trial, a synthetic A/B test with a known effect, the same A/B test exported for completers only (must refuse), knuckle cracking (no usable data), and a sensitive-topic gate test. Metrics: verdict match, key-study recall, quotes found in the source, quotes judged supporting, unsupported numbers and forbidden conclusions (must be 0), whether the disproving search ran, rank of the accepted dataset, DAG checks, tokens and wall time. Results: [`eval/results/`](../eval/results/).

Targets: expected verdict on ≥ 5 of 6 cases; ≥ 90% of quotes found in their source; 0 untraceable numbers; 0 estimates on the completers-only case; RCT tool within 1e-4 of an independent calculation (including a zero-event arm); the synthetic true effect inside the 95% CI; 0 humor where the gate forbids it.

The statistics tools are unit-tested against independent calculations (Newcombe's published table, statsmodels, scipy, hand-computed sandwich SEs), textbook DAGs (confounder, mediator, collider, M-bias) and synthetic data with known effects.

## Known limitations

- **Run-to-run variation.** Research, the DAG and the column mapping come from models, so two runs of one question can find different studies and draw different graphs. Key-study recall in particular varies between runs.
- **One table, one estimand.** Estimates come from a single table and target the average effect (ATE). DiD, IV and RDD setups are recognized and explained but not estimated. Multi-table datasets (e.g. NHANES) are not used.
- **Narrow data discovery.** Most usable tables today come from Rdatasets and two GitHub repositories, so many questions end in 📚 even when data exists elsewhere.
- **Simple certainty rule.** Any disagreement among the best studies lowers certainty; whether the disagreement has an explanation is not judged.
- **Model judgments shown as assumptions.** Which nodes are exogenous, and edges with no citing claim, have no source behind them; the report labels them as assumptions.
- **Extraction noise.** The small model sometimes over-quotes or mislabels a study's design; quote checks catch invented passages, not a misread design.
- **Metadata outages.** If a Europe PMC lookup fails after a retry, the study is kept but not checked for retraction, and the report says so.
- **No clarifying questions in batch or web runs.** Without a person to answer, the first framing is kept and its ambiguities are listed. Data is treated as observational unless randomization is documented.
- **Cost and time.** A live run takes a few minutes and roughly 200–300k model tokens, mostly from reading each source separately.
