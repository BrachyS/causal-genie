# Design decisions

Each entry gives the decision, why, and what it costs. Entries marked *open* are still being measured.

## Agent design

**The model proposes; the harness decides and computes.** Models write queries, read sources, propose edges and draft prose. The controller picks the verdict and certainty by rule, and every number comes from a tested tool. *Why:* a model's number or verdict can't be checked or reproduced; a tool's can. *Cost:* more code, and verdict rules have to be written out for every case.

**Hybrid research loop.** Each round the model plans the queries (with a reason for each) and proposes when to stop; the controller enforces caps (3 rounds, ≤ 12 queries in round 1, ≤ 6 later) and coverage (a search for evidence against the leading answer, and a primary source behind each key claim). A round-1 plan missing a query group is rejected; if the model says "done" early, the next prompt names what's missing. *Why:* the model adapts queries to what it has found; the controller makes sure the disproving search always runs. *Open:* fixed preset queries vs model-planned rounds is measured on key-study recall in the eval.

**Stop rule.** Stop after 3 rounds, when a round adds fewer than 2 new studies, or when the model says done and coverage is met. A later round whose plan says done with no gaps is skipped. *Open:* the "fewer than 2" threshold rarely fires; a higher one is being measured against key-study recall.

**Failures give partial reports, not silence.** After one retry (API) or one repair (schema), a failed step ends the run with a partial report that names the step and says it is not a finding that no evidence exists. A single source whose extraction fails is excluded (`extraction_failed`) and research goes on. A research plan that fails falls back to preset query groups.

**Up to 3 clarifying questions in one turn, then one reframe.** Asking one at a time took several rounds; asking everything at once is bounded (one user turn, at most two framing calls). Answered ambiguities are never listed as unresolved.

**The Genie's aside may follow any verdict; only the topic gates it.** The aside is shown unless the model labels the topic sensitive (health treatments people act on, vaccines, mental health, harm, money, groups of people) or the aside has a number that isn't in the question; it never states facts and always ends with a next step. *Why:* the earlier rule (only after 📚/❓/🚫, or for a whimsical question) left most demo runs without one, and an aside with no facts or numbers is no riskier next to an estimate or a study plan than next to "published evidence only". The sensitive-topic rule is what keeps humor off vaccines and medicines, and it stays. Because that label is now the whole gate, it moved to the large model with reasoning off: the small model called licorice gargling before surgery "not sensitive" and wrote an aside implying it soothes; the large one blocked it and the vaccines question in 3 of 3 tries each, and allowed the full moon and onboarding questions (about 500 tokens a call). Numbers from the question are allowed ("7-day activation" blocked every onboarding aside). *Cost:* a joke can sit beside a serious estimate; the gate rests on one model label (prompt errs to "sensitive" when unsure), still scored in the eval (humor where forbidden must be 0); asides can still hint at the finding ("babies prefer their own schedule"), which the prompt forbids but the harness can't check.

## Research quality

**Prefer scholarly domains, don't restrict to them.** Research searches use Tavily `include_domains_mode="prefer"` with PubMed, Europe PMC, Cochrane, NIH/CDC, NBER and medRxiv. Restricting would cut leads, the disproving search and journal sites. Raising results per search from 5 to 10 (same credits) improved key-study recall at the cost of more tokens.

**Dedupe by study, not by page.** DOI/PMID first, then fuzzy title + first author + year, comparing main titles before a colon. News about a paper matches on first author + year and merges into the paper. *Why:* ten news stories about one paper are one study.

**A quote counts only if it is in the page.** Fuzzy match ≥ 90 per `...` segment, short segments exact, and every number in the quote must appear in the page. Quotes that fail are sent back once for a verbatim passage or null. Then a second model call judges support. *Why:* the small model copied the question's wording into quotes; asking once more for a verbatim passage fixed nearly all of them.

**A review counts at the level of what it pools.** A systematic review of observational studies is observational evidence for certainty, not trial-level.

## Causal graph

**The question edge is the hypothesis, not a cited fact.** Exposure → outcome is always drawn, never cites claims, and never triggers extra research. Its evidence, for and against, is in the Evidence section. It is drawn bold, purple and marked "?" so it does not read as a finding; one graph rather than a question graph plus a result graph, since with no estimate there is nothing honest to draw on the second (removing the arrow would claim no effect).

**Exogenous nodes.** The model lists nodes nothing in the graph can cause (moon phase, calendar date); the harness drops arrows into them and the report shows both as the model's judgment. *Why:* "season → full moon" turned calendar factors into fake confounders.

**Outcome causes as precision variables.** Observed direct causes of the outcome that the exposure doesn't affect (weekday, holiday, season, year for births) are adjusted for if the set still passes the backdoor criterion. They are reported apart from confounders. *Why:* with no confounders, the full-moon estimate was the unadjusted one with a CI about three times wider.

**Extra research only for edges out of a cause of the exposure.** An uncited edge sends research back for one round only if its source is an ancestor of the exposure, where a missing citation could hide a confounder. Uncited edges elsewhere stay assumed.

## Data and identification

**Allowlisted open repositories only, single table.** See the system card for the list. Restricted repositories are reported, never downloaded. The model sees column names, never rows.

**Calendar nodes come from the date.** A graph node named like a derive rule (`full_moon`, `day_of_week`, `holiday`, `season`, `weekend`) is covered whenever the table has a date column or year + month + day, whatever the coverage model says. *Why:* the coverage model marked moon phase "not in the data" in some runs and not others.

**Randomized only when assignment is documented.** Either the table is an included trial's own data with a documented design, or the user says yes. Selection after assignment is flagged when rows are under 95% of the stated roster, or the arm sizes fail a sample-ratio test (p < 0.001). *Why:* the licorice trial lost one patient of 236 and must not be refused; a completers-only export must be.

**Standard errors.** Heteroskedasticity-robust (HC1) by default; cluster-robust by time when several rows share a time value (e.g. state-by-day births).

**A time-window outcome uses "any above the cutoff".** For "sore throat within 24 h" measured at four time points, the outcome is 1 if any is above the cutoff, 0 only if all are present and none is, else missing (row excluded and counted).

## Models

**Two tiers.** A small instruct model for routing, extraction, quote checks, mapping (many short calls); a large model for the DAG, the report and the Genie gate (its sensitive-topic label is the whole humor gate, and the small model missed a clinical treatment question). Chosen from a JSON-schema probe of several Token Factory models; `response_format=json_schema` was needed, as JSON mode alone returned wrong keys.

**Reasoning off for the DAG and the report.** The large model's reasoning length is unbounded and the provider ignores `reasoning_effort`: runs spent the whole 16k-token budget reasoning with empty output, or timed out. With reasoning off, recorded DAG inputs returned schema-valid output in under 5 s with the same structural checks. Graph content changes from run to run either way and is scored in the eval.

## Serving

**Replay by recording every external call.** Model calls, Tavily searches and extracts, Europe PMC lookups and data-step HTTP requests are saved with the run; downloads are read back from the cache by checksum. A replay needs no keys and no network.

**The demo shows recorded runs.** A live run takes minutes, so the web app shows a gallery by default. Gallery runs are display copies: raw page text, search snippets and model inputs/outputs are removed before they are committed; the verbatim quotes the report cites are kept. Live runs need a passcode and stop at a daily cap of runs and tokens, checked before each run starts (one run can go past it). The cap file resets when the hosted app restarts, so provider-side limits remain the hard stop.
