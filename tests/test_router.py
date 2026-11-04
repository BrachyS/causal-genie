import json

from causal_genie.agent.router import route

CACHED = {"Does the full moon cause more births?": "run-moon"}
GALLERY = {"Does sugar cause hyperactivity in kids?": "gallery-sugar"}


def decision(causal: bool, rewrite: str | None = None) -> str:
    """Router model output saying whether the question is causal."""
    return json.dumps({"causal": causal, "reason": "test", "causal_rewrite": rewrite})


def test_exact_repeat_hits_cache_without_model(make_llm) -> None:
    llm, fake = make_llm([])

    r = route("does the full moon cause more births", llm, CACHED, GALLERY)

    assert (r.kind, r.run_id) == ("cache", "run-moon")
    assert fake.requests == []


def test_gallery_question_replays_without_model(make_llm) -> None:
    llm, fake = make_llm([])

    r = route("Does sugar cause hyperactivity in kids", llm, CACHED, GALLERY)

    assert (r.kind, r.run_id) == ("gallery", "gallery-sugar")
    assert fake.requests == []


def test_one_word_change_is_not_a_gallery_hit(make_llm) -> None:
    # fuzzy text scores rate this 93/100 against the sugar question; it is a different question
    llm, _ = make_llm([decision(True)])

    r = route("Does salt cause hyperactivity in kids?", llm, CACHED, GALLERY)

    assert r.kind == "causal"


def test_descriptive_question_is_not_causal(make_llm) -> None:
    rewrite = "Does living in a city cause higher blood pressure?"
    llm, fake = make_llm([decision(False, rewrite)])

    r = route("What is the average blood pressure in US cities?", llm, CACHED, GALLERY)

    assert (r.kind, r.causal_rewrite) == ("not_causal", rewrite)
    assert fake.requests[0]["model"] == "test/small"


def test_causal_question_goes_to_research(make_llm) -> None:
    llm, _ = make_llm([decision(True)])

    r = route("Does cracking your knuckles cause arthritis?", llm, CACHED, GALLERY)

    assert (r.kind, r.run_id, r.causal_rewrite) == ("causal", None, None)
