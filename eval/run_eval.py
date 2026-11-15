"""Score saved runs of the frozen eval cases (PRD §8) and write the scorecard (T12.1, T12.4).

For each case in `eval/cases.yaml` it takes the latest saved run (or `--run CASE=RUN_ID`),
computes every metric from the saved state, replays the run offline to check that the replay
reaches the same verdict, and writes `eval/results/scorecard.md` and `scorecard.json`.
Counts, not percentages; live and replay reported separately (PRD §8).

Only one metric needs a model: whether the report asserts a forbidden conclusion, which text
matching cannot tell from its denial. A large-model judge (reasoning off) names the sentence;
the harness checks the sentence exists and counts. D4's "no estimate" is a rule.

Usage: `uv run python eval/run_eval.py [--run D1=20261114-224708-06e803 ...]`
"""

import argparse
import json
import logging
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict

from causal_genie.agent.controller import CACHE_DIR, RUNS_DIR, run
from causal_genie.agent.router import normalize
from causal_genie.agent.state import RunState
from causal_genie.agent.steps.report import free_numbers, ledger_entries, sentence_problem
from causal_genie.llm.client import LLMClient, load_models, make_openai
from causal_genie.log import setup_logging
from causal_genie.tools.download import Fetcher
from causal_genie.tools.metadata import EuropePMC
from causal_genie.tools.search import Searcher

log = logging.getLogger("run_eval")

EVAL_DIR = Path(__file__).parent
CASES_YAML = EVAL_DIR / "cases.yaml"
RESULTS_DIR = EVAL_DIR / "results"
NO_ESTIMATE = "Any numeric effect estimate."


class Judged(BaseModel):
    model_config = ConfigDict(extra="forbid")
    conclusion: int  # 1-based number of the forbidden conclusion
    asserted: bool
    sentence: int | None  # 1-based number of the asserting sentence


class ForbiddenJudgment(BaseModel):
    """The small model's judgment of which forbidden conclusions the report asserts."""

    model_config = ConfigDict(extra="forbid")
    conclusions: list[Judged]


def load_cases(path: Path = CASES_YAML) -> list[dict]:
    """The frozen eval cases."""
    return yaml.safe_load(path.read_text())["cases"]


def user_table(state: RunState) -> str | None:
    """The path of the user's own table (`--data`) if the run used one."""
    return next((c.url for c in state.datasets if c.source == "user"), None)


def pick_run(case: dict, runs_dir: Path) -> Path | None:
    """Latest saved run of the case: same question and, for a case built on a synthetic table
    (D3, D4 share a question), the same table; other cases must not have used a user table."""
    want = (case.get("data") or {}).get("synthetic")
    key = normalize(case["question"])
    for path in sorted(runs_dir.glob("*/state.json"), reverse=True):
        if normalize(json.loads(path.read_text())["question"]) != key:
            continue
        if user_table(RunState.load(path)) == want:
            return path
    return None


def verdict_match(state: RunState, case: dict) -> bool:
    return state.verdict == case["expected_verdict"]


def key_studies(state: RunState, keys: list[dict]) -> tuple[int, list[dict]]:
    """How many key studies an included study matches (PMID, else DOI ignoring case, else first
    author surname + year), and the key studies not found."""
    included = [s for s in state.studies if s.included]

    def found(k: dict) -> bool:
        for s in included:
            if k.get("pmid") and s.pmid == str(k["pmid"]):
                return True
            if k.get("doi") and s.doi and s.doi.lower() == str(k["doi"]).lower():
                return True
            if (s.first_author or "").lower() == str(k["first_author"]).lower() and s.year == k[
                "year"
            ]:
                return True
        return False

    missing = [k for k in keys if not found(k)]
    return len(keys) - len(missing), missing


def grounding(state: RunState) -> dict[str, int]:
    """Quotes found in their source, and support judgments, over the claim ledger."""
    quoted = [c for c in state.claims if c.grounding != "n/a"]
    checked = [c for c in state.claims if c.support != "unchecked"]
    return {
        "quoted": len(quoted),
        "quote_found": sum(c.grounding == "quote_found" for c in quoted),
        "checked": len(checked),
        "supports": sum(c.support == "supports" for c in checked),
        "partial": sum(c.support == "partial" for c in checked),
    }


def disproving_ran(state: RunState) -> bool:
    """Some research round searched for evidence against the claim."""
    return any(q.group == "evidence_against" for r in state.rounds for q in r.plan.queries)


def prose_sentences(state: RunState) -> list[str]:
    """The model-written report sentences, in report order; [] if there is no prose."""
    if state.prose is None:
        return []
    p = state.prose
    return [s.text for s in [*p.summary, *p.comparison, *p.limitations]]


def unsupported_numbers(state: RunState) -> list[str]:
    """Report sentences with a number no cited ledger entry holds (the report lint, re-run)."""
    if state.prose is None:
        return []
    entries, free = ledger_entries(state), free_numbers(state)
    p = state.prose
    return [
        s.text
        for s in [*p.summary, *p.comparison, *p.limitations]
        if (problem := sentence_problem(s, entries, free)) and problem.startswith("numbers")
    ]


def forbidden_conclusions(state: RunState, case: dict, llm: LLMClient) -> list[dict]:
    """Forbidden conclusions the report asserts, each with the asserting sentence.

    "Any numeric effect estimate" (D4) is a rule: any `Result`. The others go to the judge,
    whose sentence number must point at a real report sentence.
    """
    conclusions = list(case["forbidden_conclusions"])
    hits = []
    if case.get("no_estimate"):
        conclusions.remove(NO_ESTIMATE)
        if state.results:
            hits.append({"conclusion": NO_ESTIMATE, "sentence": f"{len(state.results)} result(s)"})
    sentences = prose_sentences(state)
    if not conclusions or not sentences:
        return hits
    user = (
        "Forbidden conclusions:\n"
        + "\n".join(f"{i}. {c}" for i, c in enumerate(conclusions, 1))
        + "\n\nReport sentences:\n"
        + "\n".join(f"[{i}] {s}" for i, s in enumerate(sentences, 1))
    )
    judged = llm.complete("large", "eval_forbidden", user, ForbiddenJudgment)
    for j in judged.conclusions:
        if not j.asserted:
            continue
        if not (1 <= j.conclusion <= len(conclusions)):
            raise ValueError(f"judge named conclusion {j.conclusion}; there are {len(conclusions)}")
        if j.sentence is None or not (1 <= j.sentence <= len(sentences)):
            raise ValueError(f"judge named sentence {j.sentence}; the report has {len(sentences)}")
        hits.append(
            {"conclusion": conclusions[j.conclusion - 1], "sentence": sentences[j.sentence - 1]}
        )
    return hits


def genie_ok(state: RunState, case: dict) -> bool:
    """No Genie aside where the case forbids humor (A8)."""
    return case["genie_allowed"] or state.genie is None


def unflagged_retractions(state: RunState, case: dict) -> list[str]:
    """Retracted papers the case names that the run found but did not mark retracted."""
    out = []
    for r in case.get("retracted_must_be_flagged", []):
        if any(s.pmid == str(r["pmid"]) and not s.retracted for s in state.studies):
            out.append(f"{r['first_author']} {r['year']}")
    return out


def data_check(state: RunState, case: dict) -> dict:
    """Did the run accept the expected table, and at what rank? `ok` is None for the user's
    own table (nothing to find), True/False otherwise; no data expected = no table accepted."""
    chosen = next((c for c in state.datasets if c.accepted), None)
    url, rank = (chosen.url, chosen.rank) if chosen else (None, None)
    data = case.get("data")
    if data and data.get("synthetic"):
        return {"ok": None, "rank": rank, "url": url}
    if not data:
        return {"ok": chosen is None, "rank": rank, "url": url}
    allowed = {data["url"], *data.get("also_accept", [])}
    return {"ok": url in allowed, "rank": rank, "url": url}


def dag_problems(state: RunState, case: dict) -> list[str]:
    """Breaks of the case's DAG rules (T12.5); every case forbids edges out of the outcome."""
    if not state.edges:
        return ["no DAG"]
    assert state.query is not None
    exposure, outcome = state.query.exposure.name, state.query.outcome.name
    rules = case.get("dag") or {}
    nodes = {n for e in state.edges for n in (e.source, e.target)}
    problems = [
        f"missing required node {n}" for n in rules.get("required_nodes", []) if n not in nodes
    ]
    forbidden = {tuple(e) for e in rules.get("forbidden_edges", [])}
    for e in state.edges:
        if rules.get("no_edges_into_exposure") and e.target == exposure:
            problems.append(f"edge into the exposure: {e.source} -> {e.target}")
        if (e.source, e.target) in forbidden:
            problems.append(f"forbidden edge: {e.source} -> {e.target}")
        if e.source == outcome:
            problems.append(f"edge out of the outcome: {e.source} -> {e.target}")
    return problems


def cost(state: RunState, path: Path) -> dict:
    """Tokens, calls, model time, and wall time from the run ID's start to the saved state
    (includes any manual pause)."""
    start = datetime.strptime(state.run_id[:15], "%Y%m%d-%H%M%S").replace(tzinfo=UTC)
    return {
        "tokens_in": state.tokens["in"],
        "tokens_out": state.tokens["out"],
        "calls": len(state.llm_calls),
        "llm_s": round(state.llm_latency_s, 1),
        "wall_s": round(path.stat().st_mtime - start.timestamp()),
    }


def replay(state: RunState, runs_dir: Path) -> RunState:
    """Re-run the saved run from its recordings only: no model, search or data request."""
    user = user_table(state)
    return run(
        state.question,
        LLMClient.from_replay(state.llm_calls),
        Searcher.from_replay(state.searches, state.extracts),
        EuropePMC.from_replay(state.lookups),
        Fetcher.from_replay(state.fetches),
        runs_dir,
        CACHE_DIR,
        state.clarifications.__getitem__ if state.clarifications else None,
        Path(user) if user else None,
        any(c.prompt_name == "genie_gate" for c in state.llm_calls),
    )


def score(case: dict, path: Path | None, judge: LLMClient, replay_dir: Path) -> dict:
    """Every metric for one case; a case with no saved run scores as missing."""
    row: dict = {"case": case["id"], "expected": case["expected_verdict"], "run_id": None}
    if path is None:
        return row | {
            "verdict": None,
            "verdict_ok": False,
            "replay_verdict": None,
            "replay_ok": False,
            "replay_error": "no saved run",
        }
    state = RunState.load(path)
    found, missing = key_studies(state, case["key_studies"])
    row |= {
        "run_id": state.run_id,
        "verdict": state.verdict,
        "verdict_ok": verdict_match(state, case),
        "failure": state.failure,
        "key_studies": f"{found}/{len(case['key_studies'])}",
        "key_studies_missing": [f"{k['first_author']} {k['year']}" for k in missing],
        "grounding": grounding(state),
        "unsupported_numbers": unsupported_numbers(state),
        "forbidden": forbidden_conclusions(state, case, judge),
        "disproving_search": disproving_ran(state),
        "genie_ok": genie_ok(state, case),
        "unflagged_retractions": unflagged_retractions(state, case),
        "data": data_check(state, case),
        "dag_problems": dag_problems(state, case),
        "cost": cost(state, path),
    }
    try:
        replayed = replay(state, replay_dir)
    except (KeyError, FileNotFoundError, ValueError) as e:  # a recording miss, reported as such
        log.exception("replay of %s failed", state.run_id)
        return row | {
            "replay_verdict": None,
            "replay_ok": False,
            "replay_error": f"{type(e).__name__}: {e}",
        }
    return row | {
        "replay_verdict": replayed.verdict,
        "replay_ok": replayed.verdict == state.verdict and replayed.results == state.results,
        "replay_error": None,
    }


def scorecard_counts(rows: list[dict]) -> dict[str, str]:
    """Verdict matches live, and replays reproducing the run, as counts over all cases."""
    n = len(rows)
    live = sum(r["verdict_ok"] for r in rows)
    replayed = sum(r["replay_ok"] and r["replay_verdict"] == r["expected"] for r in rows)
    return {"live": f"{live}/{n}", "replay": f"{replayed}/{n}"}


def scorecard_md(rows: list[dict]) -> str:
    """The scorecard as Markdown: counts, then one row per case, then the details."""
    counts = scorecard_counts(rows)
    lines = [
        "# Eval scorecard",
        "",
        f"Generated {datetime.now(UTC):%Y-%m-%d %H:%M} UTC from saved runs (PRD §8).",
        "",
        f"- **Expected verdict, live:** {counts['live']}",
        f"- **Expected verdict, replay (offline, from the recordings):** {counts['replay']}",
        "",
        "| Case | Run | Verdict (expected) | Replay | Key studies | Quotes found | Supports "
        "| Unsupported numbers | Forbidden | Disproving search | Data (rank) | DAG problems "
        "| Tokens in / out | Wall s |",
        "|" + "---|" * 14,
    ]
    for r in rows:
        if r["run_id"] is None:
            lines.append(f"| {r['case']} | — | — ({r['expected']}) | — |" + " — |" * 10)
            continue
        g, d, c = r["grounding"], r["data"], r["cost"]
        data = "n/a" if d["ok"] is None else ("✓" if d["ok"] else "✗")
        replay = "✓" if r["replay_ok"] else f"✗ {r['replay_error'] or r['replay_verdict']}"
        lines.append(
            f"| {r['case']} | `{r['run_id']}` | {'✓' if r['verdict_ok'] else '✗'} {r['verdict']} "
            f"({r['expected']}) | {replay} | {r['key_studies']} | {g['quote_found']}/{g['quoted']} "
            f"| {g['supports']}/{g['checked']} | {len(r['unsupported_numbers'])} "
            f"| {len(r['forbidden'])} | {'✓' if r['disproving_search'] else '✗'} "
            f"| {data} ({d['rank'] or '—'}) | {len(r['dag_problems'])} "
            f"| {c['tokens_in']:,} / {c['tokens_out']:,} | {c['wall_s']:,} |"
        )
    lines += ["", "## Details", ""]
    for r in rows:
        if r["run_id"] is None:
            lines.append(f"- **{r['case']}:** no saved run")
            continue
        details = {
            "key studies missing": r["key_studies_missing"],
            "forbidden conclusions asserted": r["forbidden"],
            "unsupported numbers": r["unsupported_numbers"],
            "DAG problems": r["dag_problems"],
            "Genie shown where forbidden": not r["genie_ok"],
            "retracted papers not flagged": r["unflagged_retractions"],
            "failure": r["failure"],
            "accepted table": r["data"]["url"],
        }
        shown = " ".join("; ".join(f"{k}: {v}" for k, v in details.items() if v).split())
        lines.append(f"- **{r['case']}:** {shown or 'nothing to note'}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Score saved runs of the eval cases.")
    parser.add_argument("--run", action="append", default=[], help="CASE=RUN_ID (repeatable)")
    args = parser.parse_args(argv)
    load_dotenv(Path.cwd() / ".env")
    setup_logging()
    chosen = dict(item.split("=", 1) for item in args.run)
    judge = LLMClient(make_openai(), load_models())
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for case in load_cases():
            path = (
                RUNS_DIR / chosen[case["id"]] / "state.json"
                if case["id"] in chosen
                else pick_run(case, RUNS_DIR)
            )
            log.info("scoring %s from %s", case["id"], path)
            rows.append(score(case, path, judge, Path(tmp)))
    judge_tokens = sum(c.tokens_in + c.tokens_out for c in judge.calls)
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "scorecard.json").write_text(
        json.dumps(
            {"rows": rows, "counts": scorecard_counts(rows), "judge_tokens": judge_tokens}, indent=2
        )
        + "\n"
    )
    (RESULTS_DIR / "scorecard.md").write_text(scorecard_md(rows))
    log.info("scorecard: %s (judge tokens %d)", scorecard_counts(rows), judge_tokens)


if __name__ == "__main__":
    main()
