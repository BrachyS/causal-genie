"""After a pick's PR is merged (U18): mark the question answered and post it to Bluesky."""

import json
from pathlib import Path

import httpx
import pytest
from sqlalchemy import Engine

from causal_genie.ui import bluesky, board_db
from causal_genie.ui.picks import Pick

APP = "https://causal-genie.streamlit.app"
LINE = "Brewing up an answer tonight!"


def pick(question: str = "Does tea cause calm?", genie: bool = True, run_id: str = "r1") -> Pick:
    return Pick(
        question_id=1,
        question=question,
        votes=4,
        run_id=run_id,
        verdict="published_only",
        announcement=LINE if genie else f"Tonight's question: {question}",
        genie=genie,
    )


def test_post_has_the_question_verdict_genie_line_and_a_clickable_link() -> None:
    text, facets = bluesky.compose(pick(), APP)

    link = f"{APP}/?run=r1"
    assert "Does tea cause calm?" in text and "4 votes" in text
    assert "📚 Published evidence only" in text
    assert LINE in text
    assert text.endswith(link)
    (facet,) = facets
    start, end = facet["index"]["byteStart"], facet["index"]["byteEnd"]
    assert text.encode()[start:end] == link.encode()
    assert facet["features"] == [{"$type": "app.bsky.richtext.facet#link", "uri": link}]


def test_a_long_question_with_emoji_is_cut_but_the_link_never_is() -> None:
    question = "Does 🍵 green tea, " + "taken every single morning, " * 12 + "cause calm?"

    text, facets = bluesky.compose(pick(question), APP)

    assert len(text) <= bluesky.MAX_CHARS
    assert "…" in text
    assert text.endswith(f"{APP}/?run=r1")
    start, end = facets[0]["index"]["byteStart"], facets[0]["index"]["byteEnd"]
    assert text.encode()[start:end].decode() == f"{APP}/?run=r1"
    assert start == len(text.encode()) - len(f"{APP}/?run=r1".encode())


def test_no_genie_line_on_a_sensitive_topic() -> None:
    text, _ = bluesky.compose(pick("Do vaccines cause autism?", genie=False), APP)

    assert "Tonight's question:" not in text
    assert text.count("Do vaccines cause autism?") == 1


def test_stranger_text_gets_no_facets() -> None:
    question = "Does @someone.bsky.social https://spam.example cause X?"
    text, facets = bluesky.compose(pick(question), APP)

    assert [f["features"][0]["uri"] for f in facets] == [f"{APP}/?run=r1"]


# --- posting through the Bluesky API (fake HTTP) -----------------------------------------


class FakeBluesky:
    """httpx.MockTransport handler for createSession and createRecord."""

    def __init__(self, record_status: int = 200) -> None:
        self.record_status = record_status
        self.requests: list[tuple[str, dict]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append((request.url.path, body))
        if request.url.path.endswith("com.atproto.server.createSession"):
            return httpx.Response(200, json={"accessJwt": "jwt", "did": "did:plc:genie"})
        assert request.headers["Authorization"] == "Bearer jwt"
        if self.record_status != 200:
            return httpx.Response(self.record_status, json={"error": "RateLimitExceeded"})
        return httpx.Response(200, json={"uri": "at://did:plc:genie/post/1", "cid": "c"})


def test_post_logs_in_with_the_app_password_then_creates_the_record() -> None:
    fake = FakeBluesky()
    with httpx.Client(transport=httpx.MockTransport(fake)) as http:
        uri = bluesky.post(http, "genie.bsky.social", "app-pass", "hello", [])

    assert uri == "at://did:plc:genie/post/1"
    (login_path, login), (record_path, record) = fake.requests
    assert login_path.endswith("createSession")
    assert login == {"identifier": "genie.bsky.social", "password": "app-pass"}
    assert record_path.endswith("createRecord")
    assert record["repo"] == "did:plc:genie"
    assert record["collection"] == "app.bsky.feed.post"
    assert record["record"]["text"] == "hello" and record["record"]["$type"] == "app.bsky.feed.post"


# --- publish: what the after-merge workflow runs -----------------------------------------


@pytest.fixture
def db() -> Engine:
    return board_db.connect("sqlite://")


def _running(db: Engine, gallery: Path, text: str, run_id: str, genie: bool = True) -> int:
    qid = board_db.add_question(db, text, f"asker of {text}")
    board_db.set_status(db, qid, "running", run_id=run_id)
    p = pick(text, genie=genie, run_id=run_id).model_copy(update={"question_id": qid})
    (gallery / run_id).mkdir(parents=True)
    (gallery / run_id / "pick.json").write_text(p.model_dump_json())
    return qid


def test_publish_marks_merged_picks_answered_and_posts_each_once(db, tmp_path) -> None:
    qid = _running(db, tmp_path, "Does tea cause calm?", "r1")
    fake = FakeBluesky()

    with httpx.Client(transport=httpx.MockTransport(fake)) as http:
        failed = bluesky.publish(db, tmp_path, http, "genie.bsky.social", "pw", APP)
        again = bluesky.publish(db, tmp_path, http, "genie.bsky.social", "pw", APP)

    q = board_db.get_question(db, qid)
    assert (q.status, q.run_id, q.announcement) == ("answered", "r1", LINE)
    assert q.answered_at is not None
    assert failed == again == []
    assert sum(path.endswith("createRecord") for path, _ in fake.requests) == 1


def test_a_post_error_fails_without_undoing_answered(db, tmp_path) -> None:
    qid = _running(db, tmp_path, "Does tea cause calm?", "r1")

    with httpx.Client(transport=httpx.MockTransport(FakeBluesky(record_status=429))) as http:
        failed = bluesky.publish(db, tmp_path, http, "genie.bsky.social", "pw", APP)

    assert failed == ["r1"]
    assert board_db.get_question(db, qid).status == "answered"


def test_a_sensitive_pick_is_answered_without_the_genie_line(db, tmp_path) -> None:
    qid = _running(db, tmp_path, "Do vaccines cause autism?", "r1", genie=False)
    fake = FakeBluesky()

    with httpx.Client(transport=httpx.MockTransport(fake)) as http:
        bluesky.publish(db, tmp_path, http, "genie.bsky.social", "pw", APP)

    assert board_db.get_question(db, qid).announcement is None
    assert "Tonight's question" not in fake.requests[-1][1]["record"]["text"]


def test_main_exits_1_when_a_post_fails(db, tmp_path, monkeypatch) -> None:
    url = f"sqlite:///{tmp_path / 'board.db'}"
    board = board_db.connect(url)
    _running(board, tmp_path / "g", "Does tea cause calm?", "r1")
    for name, value in {
        "BOARD_DB_URL": url,
        "BLUESKY_HANDLE": "genie.bsky.social",
        "BLUESKY_APP_PASSWORD": "pw",
        "CG_APP_URL": APP,
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(bluesky, "transport", lambda: httpx.MockTransport(FakeBluesky(500)))

    with pytest.raises(SystemExit) as exit_:
        bluesky.main(["publish", "--gallery", str(tmp_path / "g")])

    assert exit_.value.code == 1
