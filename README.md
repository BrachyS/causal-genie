# 🧞 Causal Genie

A research agent for "does X cause Y?" questions (*"Does the full moon cause more births?"*). It researches the question like a small systematic review, draws a causal graph (DAG) whose arrows cite checked source passages, looks for a public dataset with the treatment, outcome and confounders, and estimates the effect, or declines and says what is missing. Every number in a report traces to a tool call, and every claim to a quoted source. Models run on Nebius Token Factory and search runs on Tavily.

## Run it

```sh
uv sync
uv run python -m causal_genie.agent.controller "Does the full moon cause more births?"
uv run --env-file .env streamlit run causal_genie/ui/app.py   # the web app
uv run pytest -q                                              # tests (no keys or network)
```

Keys go in `.env` (`NEBIUS_API_KEY`, `NEBIUS_BASE_URL`, `TAVILY_API_KEY`; the app's settings are listed below). The app needs no keys to show the saved runs in `gallery/`.

## The web app

- **Ask the Genie:** pick a saved run from the gallery, or type a question. A question with no saved run can run live (about 5 minutes), behind a passcode and a daily spend cap.
- **Questions for the Genie:** the public Question of the Day board (shown only when `BOARD_DB_URL` is set).
- **Links:** `?run=<run_id>` opens a saved run and `?q=<question>` opens a saved question, so shared links and Bluesky posts land on the answer.

### Deploy on Streamlit Community Cloud

1. **New app:** this repo, branch `main`, main file `causal_genie/ui/app.py`, Python 3.11 or later. Dependencies come from `pyproject.toml` and `uv.lock`; `packages.txt` installs Graphviz, which live runs need to draw the DAG.
2. **Secrets** (app settings → Secrets, TOML). Streamlit also sets top-level secrets as environment variables, which is how the app reads them:

   ```toml
   BOARD_DB_URL = "postgresql://…"     # the board's database; leave out to hide the board tab
   NEBIUS_API_KEY = "…"                # board moderation and live runs
   NEBIUS_BASE_URL = "…"
   TAVILY_API_KEY = "…"                # live runs
   CG_LIVE_PASSCODE = "…"              # turns on live runs and the board's admin panel
   # Optional:
   # CG_LIVE_RUNS_PER_DAY = "5"        # live-run cap per day
   # CG_LIVE_TOKENS_PER_DAY = "1500000"
   # GENIE_MODE = "1"                  # the Genie's one-line aside on live runs
   ```

3. **After the first deploy:** put the app's address (for example `https://<name>.streamlit.app`) in the GitHub Actions variable `CG_APP_URL`, so Bluesky posts link back to the app.

Without any secrets the app still serves the gallery.

## Ask the Genie (Question of the Day)

In the app's "Questions for the Genie" tab, anyone can post a public "does X cause Y?" question and vote (one vote per browser, no account). Each night at 10:00 UTC a GitHub Action answers the top-voted question with the full pipeline and opens a PR. Once the PR is merged, the answer joins the gallery and is posted on Bluesky ([@causalgenie.bsky.social](https://bsky.app/profile/causalgenie.bsky.social)) with a link back. Each new question gets one moderation check: abusive, personal, spam or non-causal questions are hidden, and the asker is told why.

Setup:
- Actions secrets: `BOARD_DB_URL`, `NEBIUS_API_KEY`, `NEBIUS_BASE_URL`, `TAVILY_API_KEY`, `BLUESKY_HANDLE`, `BLUESKY_APP_PASSWORD`.
- Actions variable: `CG_APP_URL`.
- Repo setting: "Allow GitHub Actions to create and approve pull requests".
- Streamlit secret: `BOARD_DB_URL`.

## More

- [PRD.md](PRD.md): design, verdicts, data contracts, evaluation.
- [docs/DECISIONS.md](docs/DECISIONS.md): design decisions and why.
- [docs/SYSTEM_CARD.md](docs/SYSTEM_CARD.md): guardrails, limits and known failure modes.
