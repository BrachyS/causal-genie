# Eval scorecard

Generated 2026-11-15 14:57 UTC from saved runs (PRD §8).

- **Expected verdict, live:** 6/6
- **Expected verdict, replay (offline, from the recordings):** 5/6

| Case | Run | Verdict (expected) | Replay | Key studies | Quotes found | Supports | Unsupported numbers | Forbidden | Disproving search | Data (rank) | DAG problems | Tokens in / out | Wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| D0 | `20261115-052047-115743` | ✓ estimated (estimated) | ✓ | 2/4 | 86/87 | 60/86 | 0 | 0 | ✓ | ✓ (1) | 0 | 177,623 / 36,517 | 1 |
| D1 | `20261114-224708-06e803` | ✓ estimated (estimated) | ✗ KeyError: "no recorded call for prompt 'check_support' with this input" | 1/1 | 71/74 | 55/71 | 0 | 0 | ✓ | ✓ (1) | 0 | 190,994 / 40,429 | 23,605 |
| D3 | `20261114-234555-c5e141` | ✓ estimated (estimated) | ✓ | 0/0 | 34/38 | 22/34 | 0 | 0 | ✓ | n/a (1) | 0 | 237,521 / 45,784 | 5,777 |
| D4 | `20261115-014452-a6e5b4` | ✓ study_plan (study_plan) | ✓ | 0/0 | 32/33 | 16/32 | 0 | 0 | ✓ | n/a (1) | 0 | 271,603 / 30,654 | 12,942 |
| D5 | `20261115-014121-7b78a7` | ✓ published_only (published_only) | ✓ | 2/3 | 102/102 | 77/102 | 0 | 0 | ✓ | ✓ (—) | 0 | 171,672 / 29,780 | 209 |
| G1 | `20261115-144947-108e7c` | ✓ published_only (published_only) | ✓ | 2/2 | 148/153 | 110/148 | 0 | 0 | ✓ | ✓ (—) | 0 | 234,569 / 48,102 | 458 |

## Details

- **D0:** key studies missing: ['Morton-Pradhan 2005', 'Kuss 2008']; accepted table: https://vincentarelbundock.github.io/Rdatasets/csv/fivethirtyeight/US_births_1994_2003.csv
- **D1:** accepted table: https://vincentarelbundock.github.io/Rdatasets/csv/medicaldata/licorice_gargle.csv
- **D3:** accepted table: tests/fixtures/d3_onboarding_ab.csv
- **D4:** accepted table: tests/fixtures/d4_onboarding_completers.csv
- **D5:** key studies missing: ['Unger 1998']
- **G1:** nothing to note
