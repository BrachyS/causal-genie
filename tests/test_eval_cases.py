from pathlib import Path

import yaml

CASES_YAML = Path(__file__).parents[1] / "eval" / "cases.yaml"
VERDICTS = {  # PRD §5.2, in order: ✅ ⚠️ 📚 🔒 🧪 🔀 ❓ 🚫
    "estimated",
    "estimated_strong_assumptions",
    "published_only",
    "restricted",
    "study_plan",
    "common_cause",
    "insufficient",
    "not_causal",
}


def test_eval_set_has_the_seven_frozen_cases() -> None:
    cases = yaml.safe_load(CASES_YAML.read_text())["cases"]

    assert [c["id"] for c in cases] == ["D0", "D1", "D3", "D4", "D5", "D6", "G1"]
    for case in cases:
        assert case["question"].endswith("?")
        assert case["expected_verdict"] in VERDICTS
        assert isinstance(case["genie_allowed"], bool)
        assert case["forbidden_conclusions"]
        for study in case["key_studies"]:
            assert study["pmid"] or study["doi"]
            # a comma in an unquoted inline-YAML value silently splits it into extra keys
            assert set(study) == {"pmid", "doi", "first_author", "year", "finding"}
