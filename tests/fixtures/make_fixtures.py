"""Build the synthetic CSVs in this folder (T9.6, T9.4.1). Fixed seeds, so the files are
reproducible: `uv run python tests/fixtures/make_fixtures.py`.

Each table has a known true effect and a known design, so the estimate and design tests check
the tools against the truth:

- D3 `d3_onboarding_ab.csv`: 1:1 A/B test, every assigned user; activation 30% → 35% (RD 0.05).
- D4 `d4_onboarding_completers.csv`: the same test exported for completers only (selection
  after assignment: the new flow lets more users finish).
- `did_staggered_launch.csv`: city × month panel, launches at months 8, 12 and 16, 5 cities
  never launch; launch adds 5 orders (ATT 5).
- `rdd_scholarship.csv`: scholarship given at score ≥ 60; it adds 0.3 to GPA at the cutoff.
- `iv_distance.csv`: distance to college shifts attendance; an unobserved ability confounds
  college and wage and is left out of the file; college adds 2 to wage.
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent


def onboarding(rng: np.random.Generator) -> pd.DataFrame:
    """The full A/B roster, with whether each user finished onboarding (post-assignment)."""
    n = 6000
    new_flow = rng.permutation(np.repeat([0, 1], n // 2))
    completed = rng.random(n) < np.where(new_flow == 1, 0.70, 0.55)
    activated = rng.random(n) < 0.30 + 0.05 * new_flow
    days = rng.integers(0, 28, n)
    return pd.DataFrame(
        {
            "user_id": np.arange(100001, 100001 + n),
            "signup_date": pd.DatetimeIndex(
                pd.Timestamp("2026-03-02") + pd.to_timedelta(days, "D")
            ).strftime("%Y-%m-%d"),
            "platform": rng.choice(["ios", "android", "web"], n, p=[0.4, 0.4, 0.2]),
            "new_flow": new_flow,
            "activated_7d": activated.astype(int),
            "completed_onboarding": completed.astype(int),
        }
    )


def staggered_launch(rng: np.random.Generator) -> pd.DataFrame:
    """20 cities × 24 months; 15 cities launch at month 8, 12 or 16, 5 never do."""
    cities, months = 20, 24
    start = {c: [8, 12, 16][c % 3] for c in range(15)}
    rows = []
    for c in range(cities):
        level = rng.normal(100, 10)
        for m in range(1, months + 1):
            launched = int(c in start and m >= start[c])
            orders = level + 0.5 * m + 5 * launched + rng.normal(0, 2)
            rows.append(
                {
                    "city": f"city_{c:02d}",
                    "month": m,
                    "launched": launched,
                    "orders": round(orders, 2),
                }
            )
    return pd.DataFrame(rows)


def scholarship(rng: np.random.Generator) -> pd.DataFrame:
    """Students with a test score; the scholarship goes to everyone scoring 60 or more."""
    n = 2000
    score = rng.uniform(0, 100, n).round(1)
    got = (score >= 60).astype(int)
    gpa = 2 + 0.01 * score + 0.3 * got + rng.normal(0, 0.3, n)
    return pd.DataFrame(
        {"student_id": np.arange(1, n + 1), "score": score, "scholarship": got, "gpa": gpa.round(2)}
    )


def distance(rng: np.random.Generator) -> pd.DataFrame:
    """Distance to the nearest college shifts attendance; ability (not in the file) confounds."""
    n = 3000
    km = rng.uniform(0, 100, n).round(1)
    ability = rng.normal(0, 1, n)
    college = (ability - 0.03 * km + 1.5 + rng.normal(0, 1, n) > 0).astype(int)
    wage = 10 + 2 * college + 3 * ability + rng.normal(0, 1, n)
    return pd.DataFrame(
        {
            "person_id": np.arange(1, n + 1),
            "distance_km": km,
            "college": college,
            "wage": wage.round(2),
        }
    )


def main() -> None:
    """Write every fixture CSV next to this script."""
    roster = onboarding(np.random.default_rng(3))
    roster.drop(columns="completed_onboarding").to_csv(HERE / "d3_onboarding_ab.csv", index=False)
    completers = roster[roster["completed_onboarding"] == 1]
    completers.to_csv(HERE / "d4_onboarding_completers.csv", index=False)
    staggered_launch(np.random.default_rng(4)).to_csv(
        HERE / "did_staggered_launch.csv", index=False
    )
    scholarship(np.random.default_rng(5)).to_csv(HERE / "rdd_scholarship.csv", index=False)
    distance(np.random.default_rng(6)).to_csv(HERE / "iv_distance.csv", index=False)


if __name__ == "__main__":
    main()
