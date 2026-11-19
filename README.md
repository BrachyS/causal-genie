# 🧞 Causal Genie

**Try it: [causalgenie.streamlit.app](https://causalgenie.streamlit.app)**. Read answered questions in the gallery, or post your own and vote.

**▶️ [Watch the 3-minute demo](https://youtu.be/0SusEmi7FcQ)**: a live run on the site, then three answers from the gallery.

A research agent for "does X cause Y?" questions (*"Does the full moon cause more births?"*). It researches the question like a small systematic review, draws a causal graph (DAG) whose arrows cite checked source passages, looks for a public dataset with the treatment, outcome and confounders, and estimates the effect, or declines and says what is missing. Every number in a report traces to a tool call, and every claim to a quoted source. Models run on Nebius Token Factory and search runs on Tavily.

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

## More

- [PRD.md](PRD.md): design, verdicts, data contracts, evaluation.
- [docs/DECISIONS.md](docs/DECISIONS.md): design decisions and why.
- [docs/SYSTEM_CARD.md](docs/SYSTEM_CARD.md): guardrails, limits and known failure modes.

## License

MIT, see [LICENSE](LICENSE).
