#!/usr/bin/env python3
"""Post a Feishu (Lark) interactive card to the project's custom-bot
webhook, using card JSON schema 2.0 so we can mix native components
(table, divider, markdown, ...) instead of stuffing everything into
one big markdown blob.

The Feishu bot on this repo enforces a content-keyword filter — the
outgoing message must contain ``播报`` or it's rejected. Every card
title is auto-prefixed with the keyword if the caller forgot.

Usage:

    # full custom: pass elements JSON via stdin
    cat <<'EOF' | python3 scripts/notify_feishu.py \
        --kind B --title "corpus 播报 @ 1488 papers"
    [
      {"tag": "markdown", "content": "## SNAPSHOT\n- papers: 1488"},
      {"tag": "hr"},
      {"tag": "table", "columns": [...], "rows": [...]}
    ]
    EOF

    # OR use --elements-file
    python3 scripts/notify_feishu.py --kind B --title "..." \
        --elements-file body.json

The webhook URL comes from $FEISHU_WEBHOOK, falling back to the
known project-bot URL embedded below. Exit 0 on HTTP success,
non-zero on any error — kept simple so callers can decide whether
to retry.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

DEFAULT_WEBHOOK = (
    "https://open.feishu.cn/open-apis/bot/v2/hook/"
    "cb57e4cb-6ac3-4d81-a1e4-68f0ff1cb04d"
)
KEYWORD = "播报"

HEADER_TEMPLATES = {
    "A": "wathet",
    "B": "blue",
    "ALERT": "orange",
    "CRITICAL": "red",
    "INFO": "grey",
    "OK": "green",
}


def build_card(
    title: str,
    elements: list[dict],
    kind: str = "B",
    subtitle: str | None = None,
) -> dict:
    """Assemble a Feishu schema-2.0 interactive card envelope.

    ``elements`` is the body content as a list of native v2 components
    (``markdown`` / ``table`` / ``hr`` / ``column_set`` / ...).  We
    splice them into the card unchanged so the caller has full
    control over rendering.
    """
    template = HEADER_TEMPLATES.get(kind, "blue")
    if KEYWORD not in title:
        title = f"【{KEYWORD}】{title}"
    header = {
        "title": {"tag": "plain_text", "content": title},
        "template": template,
    }
    if subtitle:
        header["subtitle"] = {"tag": "plain_text", "content": subtitle}
    return {
        "msg_type": "interactive",
        "card": {
            "schema": "2.0",
            "config": {"wide_screen_mode": True},
            "header": header,
            "body": {"elements": elements},
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
            return resp.status, resp.read().decode("utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001
        return -1, f"error: {exc}"


def _read_elements(args: argparse.Namespace) -> list[dict]:
    if args.elements_file:
        raw = Path(args.elements_file).read_text(encoding="utf-8")
    else:
        raw = sys.stdin.read()
    try:
        elements = json.loads(raw)
    except json.JSONDecodeError as exc:
        sys.stderr.write(f"elements JSON parse error: {exc}\n")
        sys.exit(2)
    if not isinstance(elements, list):
        sys.stderr.write("elements must be a JSON array\n")
        sys.exit(2)
    return elements


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", required=True)
    parser.add_argument(
        "--subtitle",
        default=None,
        help="optional subtitle line under the header",
    )
    parser.add_argument(
        "--kind",
        choices=tuple(HEADER_TEMPLATES.keys()),
        default="B",
    )
    parser.add_argument(
        "--elements-file",
        default=None,
        help="path to a JSON file containing the body.elements array. "
        "If omitted, the array is read from stdin.",
    )
    parser.add_argument(
        "--webhook",
        default=os.environ.get("FEISHU_WEBHOOK", DEFAULT_WEBHOOK),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the assembled card JSON to stdout without POSTing",
    )
    args = parser.parse_args()

    elements = _read_elements(args)
    payload = build_card(args.title, elements, args.kind, args.subtitle)

    if args.dry_run:
        json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
        return 0

    status, resp_body = send(payload, args.webhook)
    try:
        code = json.loads(resp_body).get("code")
    except Exception:  # noqa: BLE001
        code = None
    if status == 200 and code == 0:
        print(f"[feishu ok] {args.title}")
        return 0
    print(f"[feishu FAIL status={status} code={code}] {resp_body[:200]}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
