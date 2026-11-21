"""After a Question of the Day PR is merged (U18): mark the question answered, post to Bluesky.

A daily workflow runs `python -m causal_genie.ui.bluesky publish`. It reads every
`gallery/*/pick.json`; a pick whose question is still `running` with that run is merged but not
yet posted. Only the oldest such pick (by run id, which starts with its date) is published per
run, so the feed gets one question a day and a backlog drains on nights with no new answer.
It is marked `answered` first, then posted, so a failed post leaves the answer published on
the board and fails the job for a retry by hand; a re-run never posts twice.

The post (at most 300 characters) has the question (cut if needed, never the link), the vote
count (left out at 0), the verdict label and a deep link to the run. The pick's announcement
is not posted: it was written before the answer ("Tonight the Genie…") and reads backwards
next to a verdict. Only our link gets a facet, so links or @mentions inside a stranger's
question stay plain text. Plain httpx: log in with the app password, then create the record.
"""

import argparse
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import httpx
from sqlalchemy import Engine

from causal_genie.agent.steps.report import VERDICT_LABELS
from causal_genie.log import setup_logging
from causal_genie.ui import board_db
from causal_genie.ui.gallery import GALLERY_DIR
from causal_genie.ui.picks import Pick

log = logging.getLogger(__name__)

XRPC = "https://bsky.social/xrpc"
# Bluesky counts graphemes; code points are never fewer, so this count is on the safe side.
MAX_CHARS = 300


def run_link(app_url: str, run_id: str) -> str:
    """The app's deep link to a saved run (U14)."""
    return f"{app_url.rstrip('/')}/?run={run_id}"


def compose(pick: Pick, app_url: str) -> tuple[str, list[dict]]:
    """The post text and its facets (the link's UTF-8 byte range, so it is clickable)."""
    link = run_link(app_url, pick.run_id)
    votes = f" ({pick.votes} vote" + ("" if pick.votes == 1 else "s") + ")" if pick.votes else ""
    head = f"Question of the Day{votes}: "
    verdict = f"\n{VERDICT_LABELS[pick.verdict]}" if pick.verdict else ""
    tail = f"{verdict}\n{link}"
    room = MAX_CHARS - len(head) - len(tail)
    question = pick.question
    if len(question) > room:
        question = question[: room - 1].rstrip() + "…"
    text = head + question + tail
    start = len(text.encode()) - len(link.encode())
    facet = {
        "index": {"byteStart": start, "byteEnd": start + len(link.encode())},
        "features": [{"$type": "app.bsky.richtext.facet#link", "uri": link}],
    }
    return text, [facet]


def post(http: httpx.Client, handle: str, app_password: str, text: str, facets: list[dict]) -> str:
    """Log in and create the post; returns its at:// URI. HTTP errors are raised."""
    session = http.post(
        f"{XRPC}/com.atproto.server.createSession",
        json={"identifier": handle, "password": app_password},
    )
    session.raise_for_status()
    auth = session.json()
    record = {
        "$type": "app.bsky.feed.post",
        "text": text,
        "facets": facets,
        "langs": ["en"],
        "createdAt": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }
    created = http.post(
        f"{XRPC}/com.atproto.repo.createRecord",
        headers={"Authorization": f"Bearer {auth['accessJwt']}"},
        json={"repo": auth["did"], "collection": "app.bsky.feed.post", "record": record},
    )
    created.raise_for_status()
    return created.json()["uri"]


def publish(
    db: Engine, gallery: Path, http: httpx.Client, handle: str, app_password: str, app_url: str
) -> list[str]:
    """Mark the oldest merged, unposted pick answered and post it; returns its run id if the
    post failed."""
    failed: list[str] = []
    for path in sorted(gallery.glob("*/pick.json"), key=lambda p: p.parent.name):
        pick = Pick.model_validate_json(path.read_text())
        q = board_db.get_question(db, pick.question_id)
        if q.status != "running" or q.run_id != pick.run_id:
            continue
        board_db.set_status(
            db,
            q.id,
            "answered",
            run_id=pick.run_id,
            announcement=pick.announcement if pick.genie else None,
        )
        text, facets = compose(pick, app_url)
        try:
            uri = post(http, handle, app_password, text, facets)
        except httpx.HTTPError:
            log.exception("bluesky: post for run %s failed; the question stays answered", q.run_id)
            failed.append(pick.run_id)
            return failed
        log.info("bluesky: posted run %s as %s", pick.run_id, uri)
        return failed
    log.info("bluesky: no merged pick waiting to post")
    return failed


def transport() -> httpx.BaseTransport:
    return httpx.HTTPTransport(retries=1)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Publish merged Question of the Day answers.")
    sub = parser.add_subparsers(dest="command", required=True)
    pub = sub.add_parser("publish", help="post the oldest merged pick to Bluesky")
    pub.add_argument("--gallery", type=Path, default=GALLERY_DIR)
    args = parser.parse_args(argv)
    setup_logging()
    app_url = os.environ.get("CG_APP_URL", "")
    if not app_url:
        # Checked before anything is marked answered, so the pick waits for a re-run.
        log.error("bluesky: CG_APP_URL is not set; nothing published")
        sys.exit(1)
    db = board_db.connect(os.environ["BOARD_DB_URL"])
    with httpx.Client(transport=transport(), timeout=30) as http:
        failed = publish(
            db,
            args.gallery,
            http,
            os.environ["BLUESKY_HANDLE"],
            os.environ["BLUESKY_APP_PASSWORD"],
            app_url,
        )
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
