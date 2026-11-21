# 🧞 Causal Genie

**Try it: [causalgenie.streamlit.app](https://causalgenie.streamlit.app)**. Read answered questions in the gallery, or post your own and vote.

**▶️ [Watch the 3-minute demo](https://youtu.be/0SusEmi7FcQ)**: a live run on the site, then answers from the gallery.

Built for the ODSC AI hackathon (November 2026). The version submitted on Nov 19 is the [`hackathon-submission`](https://github.com/BrachyS/causal-genie/releases/tag/hackathon-submission) release; work since then continues on `main`.

A research agent for "does X cause Y?" questions (*"Does the full moon cause more births?"*). It researches the question like a small systematic review, draws a causal graph (DAG) whose arrows cite checked source passages, looks for a public dataset with the treatment, outcome and confounders, and estimates the effect, or declines and says what is missing. Every number in a report traces to a tool call, and every claim to a quoted source. Models run on Nebius Token Factory and search runs on Tavily.

**Team:** Sabrina Liu ([@BrachyS](https://github.com/BrachyS))

## The problem

"Does X cause Y?" is one of the most common questions people take to a search engine or a chatbot, about health, habits or a product change. The usual answers either repeat a headline or give a confident summary with no way to check it. Answering properly means finding the studies, telling correlation from causation, naming the confounders and, where data exists, estimating the effect: hours of work for an analyst, and out of reach for most people.

Causal Genie is for curious readers who want a checked answer, and for analysts who want a sourced first pass, including on their own A/B test table. Its answer says how strong the evidence is, and it declines to estimate when the data cannot support one.

## Run it

```sh
uv sync
uv run python -m causal_genie.agent.controller "Does the full moon cause more births?"
uv run python -m causal_genie.agent.controller "Does the full moon cause more births?" --replay   # replay that run: no keys, no network
uv run --env-file .env streamlit run causal_genie/ui/app.py   # the web app
uv run pytest -q                                              # tests (no keys or network)
```

Keys go in `.env` (`NEBIUS_API_KEY`, `NEBIUS_BASE_URL`, `TAVILY_API_KEY`; the app's settings are listed below). The app needs no keys to show the saved runs in `gallery/`.

## How it works

The model proposes; the harness decides and computes.

1. **Frame** the question as population, exposure, comparator and outcome.
2. **Research** it: the model plans searches, the harness runs them (Tavily), looks up each study in Europe PMC, drops retracted papers and merges reports of the same study. A quote counts only if it is found in the page.
3. **Draw the causal graph.** Each arrow cites checked claims or is marked as an assumption.
4. **Find data** in allowlisted public repositories, or take your own table (`--data FILE`). The model sees column names, never rows.
5. **Identify and estimate** with tested statistics tools. No number comes from a model.
6. **Pick the verdict by rule**: estimated, estimated with strong assumptions, published evidence only, data restricted, study plan, insufficient evidence or not causal.
7. **Write the report.** A check rejects any sentence without a citation and any number not found in a cited source or tool result.

A live run takes about 2 to 11 minutes and 200k to 450k model tokens. Every run is saved to `runs/` with all its model calls, searches and downloads, so `--replay` reproduces it offline. The gallery holds display copies of ten recorded runs.

- **Query planning:** the model plans searches for the effect itself, for confounders and for datasets. The harness stops research after the first round if every part of the question is covered, and later rounds stop when fewer than 5 new studies turn up. An arrow in the causal graph with no source that could change the adjustment set sends research back for one more round.
- **Evidence handling:** each source is looked up in Europe PMC, retracted papers are dropped, and reports of the same study are merged (same DOI or PubMed id, or a close title with the same first author and year), so ten news stories about one paper count once. A small model copies a quote for each claim; the harness keeps the claim only if that quote is found on the source page and another check judges that it supports the claim. A claim that found no effect is never used to support an arrow.
- **Safety:** web pages, papers and datasets are treated as data, never as instructions. Model output is validated against Pydantic schemas (one repair retry). No code or SQL written by a model is ever run, downloads come only from allowlisted hosts and formats, and an uploaded table's rows never reach a model. A sensitive question (for example vaccines and autism) turns the Genie's playful aside off.
- **Trade-offs:** about 95% of the tokens go to the small model, which does the many short steps (routing, extracting quotes, mapping columns); the large model does the few hard ones (the causal graph and the report). Every model call, search and lookup is recorded, so a run can be replayed for free; a failed API call gets one retry, then the error is logged and the run carries on without that source where it can.

## Required services

| Requirement | How we use it |
|---|---|
| Nebius Token Factory (inference) | Small model `Qwen/Qwen3-30B-A3B-Instruct-2507`: routing, framing, planning searches, extracting quotes, judging support, mapping columns. Large model `deepseek-ai/DeepSeek-V4-Pro-0813`: the causal graph, the report, board moderation and the sensitive-topic check. Set in [causal_genie/llm/models.yaml](causal_genie/llm/models.yaml). |
| Tavily (real-time web search) | Every research query and dataset search. Each result's URL becomes a study's source; the page text is where the quote must be found, and the report links each claim to that URL. |

## Evaluation

Seven frozen cases in [eval/cases.yaml](eval/cases.yaml), scored by `uv run python eval/run_eval.py`. Latest [scorecard](eval/results/scorecard.md):

| Check | Result |
|---|---|
| Expected verdict, live runs | 7/7 |
| Expected verdict, offline replay of the recordings | 7/7 |
| Quotes found in their source | 547/560 (97.7%) |
| Numbers in a report not traceable to a source or tool result | 0 |
| Conclusions the case forbids | 0 |
| Estimates produced when the design does not support one | 0 |
| Synthetic A/B test with a true effect of 5 points | 5.5 (95% CI 3.1 to 7.9) |

The cases cover an estimate from public data (full moon and births), a trial's own data (licorice gargle and sore throat), an uploaded A/B test, the same test with only the users who finished onboarding (the agent must decline), a question with no usable data (knuckle cracking), a spurious link explained by a common cause (ice cream sales and shark attacks), and a sensitive topic where the playful aside must stay off (vaccines and autism).

## Example

**Question:** Does licorice gargling reduce post-op sore throat? ([full answer](https://causalgenie.streamlit.app/?run=20261115-212135-098533))

**Answer (excerpt):** ✅ Estimated from data. "The evidence suggests licorice gargling before surgery likely reduces sore throat in the first 24 hours after a breathing tube is removed. Our analysis of trial data estimated a 11.7 percentage point lower risk of sore throat with licorice, though the confidence interval included no difference. [tool result] Published trials and reviews also found fewer sore throats with licorice, including a meta-analysis reporting a risk ratio of 0.44. [1, 2, 3] Confidence is moderate because the best studies point in different directions."

**Sources and the quotes found on them:**
- [tool result] `backdoor_effect`: risk difference −0.117 (95% CI −0.244 to 0.010), OLS with robust standard errors, 233 patients from the trial's own data.
- [1] [Meta-analysis, 2019](https://www.ncbi.nlm.nih.gov/pubmed/30391446): "topical licorice was associated with a reduced incidence (risk ratio, 0.44; 95% confidence interval (CI), 0.28-0.69; P < 0.001)…"
- [2] [RCT, 2013](https://pubmed.ncbi.nlm.nih.gov/23921656): "19% and 36% at 30 minutes, 10% and 35% at 1.5 hours, and 21% and 45% at 4 hours, respectively."
- [3] [Secondary analysis of an RCT](https://healthinnovationpress.com/index.php/jnes/article/view/postoperative-sore-throat-double-lumen-licorice-gargle): "Licorice reduced the odds of sore throat by about two thirds across assessments (adjusted OR 0.33, 95% CI 0.20–0.52; p < .001)…"

## Failure / insufficient-evidence case

**Question:** Did the new onboarding flow raise 7-day activation? The uploaded A/B table holds only the users who finished onboarding. ([full answer](https://causalgenie.streamlit.app/?run=20261116-042447-932f43))

**Answer:** 🧪 Not estimated: study plan. "The available data could not support an estimate because the analysis kept only users who completed onboarding, and the treated and control groups were no longer balanced. … the key next step is to get the full randomized roster, including units that left early." The same test with the full table gives 5.5 points (95% CI 3.1 to 7.9) against a true effect of 5.

Other ways the agent stops short of an answer, each its own verdict: published evidence only (no usable dataset, as for knuckle cracking), data restricted, insufficient evidence, a common cause instead of an effect (ice cream sales and shark attacks), and not causal. When a tool fails, it gets one retry; then the error is logged and recorded, and the run goes on without that source or stops with a partial report that names the failed step.

## The web app

One page, no tabs.

- **Ask the Genie 🧞** (top of the page): the one place to type a question. **Post to the board** makes it public: anyone can vote, and the top-voted question is answered each night (needs `BOARD_DB_URL`). Under **Have a passcode?**, **Run privately now** answers it live instead (about 5 to 10 minutes, with a daily spend cap). A question that is already in the gallery gets a link to its answer.
- **📚 Answered questions:** cards in topic bands (Daily life, Health, Silly & fun, Business (A/B tests), Miscellaneous), newest first, each with the question, the date it was answered, the verdict, the first sentence of the answer and the Genie's aside. Search the questions or filter by verdict. **Read full answer →** opens the answer on its own page, with the causal graph, the studies behind each arrow linked to their source cards, and **Download report (PDF)**. Answers are dated: they reflect the evidence found then.
- **Sidebar:** the board (**🗳️ Vote on what's next**: today's question, the one being answered, and what's up next), **How to use it**, and **What the verdicts mean**.
- **Links:** `?run=<run_id>` opens a saved run and `?q=<question>` opens a saved question, so shared links and Bluesky posts land on the answer.

### Deploy on Streamlit Community Cloud

1. **New app:** this repo, branch `main`, main file `causal_genie/ui/app.py`, Python 3.11 or later. Dependencies come from `pyproject.toml` and `uv.lock`; `packages.txt` installs Graphviz, which live runs need to draw the DAG.
2. **Secrets** (app settings → Secrets, TOML). Streamlit also sets top-level secrets as environment variables, which is how the app reads them:

   ```toml
   BOARD_DB_URL = "postgresql://…"     # the board's database; leave out to hide the board
   NEBIUS_API_KEY = "…"                # board moderation and live runs
   NEBIUS_BASE_URL = "…"
   TAVILY_API_KEY = "…"                # live runs
   CG_LIVE_PASSCODE = "…"              # turns on live runs and the board's admin panel
   # Optional:
   # CG_LIVE_RUNS_PER_DAY = "5"        # live-run cap per day
   # CG_LIVE_TOKENS_PER_DAY = "1500000"
   # GENIE_MODE = "1"                  # the Genie's one-line aside on live runs
   ```

3. **After the first deploy:** put the app's address (here `https://causalgenie.streamlit.app`) in the GitHub Actions variable `CG_APP_URL`, so Bluesky posts link back to the app.

Without any secrets the app still serves the gallery.

## Ask the Genie (Question of the Day)

In the app's **Ask the Genie 🧞** box, anyone can post a public "does X cause Y?" question and vote (one vote per browser, no account) in the sidebar. Each night at 10:00 UTC a GitHub Action answers the top-voted question with the full pipeline and opens a PR. Once the PR is merged, the answer joins the gallery and is posted on Bluesky ([@causalgenie.bsky.social](https://bsky.app/profile/causalgenie.bsky.social)) with a link back. Each new question gets one moderation check: abusive, personal, spam or non-causal questions are hidden, and the asker is told why.

Setup:
- Actions secrets: `BOARD_DB_URL`, `NEBIUS_API_KEY`, `NEBIUS_BASE_URL`, `TAVILY_API_KEY`, `BLUESKY_HANDLE`, `BLUESKY_APP_PASSWORD`.
- Actions variable: `CG_APP_URL`.
- Repo setting: "Allow GitHub Actions to create and approve pull requests".
- Streamlit secret: `BOARD_DB_URL`.

## Where to look

| To check | Where to look |
|---|---|
| Sourced and verifiable answers | [Example](#example), [Evaluation](#evaluation) (547/560 quotes found on their source page, 0 untraceable numbers), the demo |
| Works end to end | [Run it](#run-it), the [live app](https://causalgenie.streamlit.app), the live run in the demo |
| Usefulness | [The problem](#the-problem), [Failure case](#failure--insufficient-evidence-case), [Ask the Genie](#ask-the-genie-question-of-the-day) |
| Agent design | [How it works](#how-it-works), [docs/DECISIONS.md](docs/DECISIONS.md) |
| Use of Token Factory and Tavily | [Required services](#required-services) |

## More

- [PRD.md](PRD.md): design, verdicts, data contracts, evaluation.
- [docs/DECISIONS.md](docs/DECISIONS.md): design decisions and why.
- [docs/SYSTEM_CARD.md](docs/SYSTEM_CARD.md): guardrails, limits and known failure modes.

## License

MIT, see [LICENSE](LICENSE).
