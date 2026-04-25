#!/usr/bin/env python3
"""Post a corpus-monitor card to a Feishu custom-bot webhook.

The Feishu bot on this repo enforces a keyword filter — outgoing
messages must contain the word ``播报`` or they're rejected. Every
card template below includes that keyword in the title by design.

Usage (command line)::

    python3 scripts/notify_feishu.py --kind A \
        --title "corpus 播报 @ 1485/1066 OK" \
        --body "$(cat body.md)"

Or via stdin::

    cat body.md | python3 scripts/notify_feishu.py --kind B \
        --title "corpus B 播报 — SUBSETS snapshot"

The webhook URL comes from $FEISHU_WEBHOOK, falling back to the
known project-bot URL embedded below. Exit 0 on HTTP success,
non-zero on any error — kept simple so it's OK to call from a loop
without killing the monitor.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request

DEFAULT_WEBHOOK = (
    "https://open.feishu.cn/open-apis/bot/v2/hook/"
    "cb57e4cb-6ac3-4d81-a1e4-68f0ff1cb04d"
)
KEYWORD = "播报"


def _card(title: str, body_md: str, kind: str) -> dict:
    """Build a Feishu interactive card.

    Header colour is chosen by ``kind``:
    - A (per-paper broadcast) → wathet (light blue), low visual weight
    - B (10-min snapshot)     → blue,   medium visual weight
    - ALERT / CRITICAL        → red,    high visual weight
    """
    template = {
        "A": "wathet",
        "B": "blue",
        "ALERT": "red",
        "CRITICAL": "red",
        "INFO": "grey",
    }.get(kind, "wathet")
    if KEYWORD not in title and KEYWORD not in body_md:
        # Keyword filter: prepend to title so the bot accepts the card
        # without polluting the body.
        title = f"【{KEYWORD}】{title}"
    return {
        "msg_type": "interactive",
        "card": {
            "config": {"wide_screen_mode": True},
            "header": {
                "title": {"tag": "plain_text", "content": title},
                "template": template,
            },
            "elements": [
                {"tag": "markdown", "content": body_md},
            ],
        },
    }


def send(payload: dict, webhook: str, timeout: float = 10.0) -> tuple[int, str]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        webhook,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return resp.status, body
    except Exception as exc:  # noqa: BLE001
        return -1, f"error: {exc}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kind",
        choices=("A", "B", "ALERT", "CRITICAL", "INFO"),
        default="A",
    )
    parser.add_argument("--title", required=True)
    parser.add_argument(
        "--body",
        default=None,
        help="Markdown body. If omitted, read from stdin.",
    )
    parser.add_argument(
        "--webhook",
        default=os.environ.get("FEISHU_WEBHOOK", DEFAULT_WEBHOOK),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Render the card JSON to stdout without POSTing.",
    )
    args = parser.parse_args()

    body = args.body if args.body is not None else sys.stdin.read()
    payload = _card(args.title, body, args.kind)

    if args.dry_run:
        json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
        return 0

    status, resp_body = send(payload, args.webhook)
    # Feishu returns HTTP 200 + JSON {"code":0,"msg":"success",...} on ok.
    try:
        resp_obj = json.loads(resp_body)
        code = resp_obj.get("code")
    except Exception:  # noqa: BLE001
        code = None
    if status == 200 and code == 0:
        print(f"[feishu ok] {args.title}")
        return 0
    print(f"[feishu FAIL status={status} code={code}] {resp_body[:200]}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
