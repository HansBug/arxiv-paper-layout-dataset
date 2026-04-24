"""Inject bounding-box anchors into arXiv LaTeX source.

Strategy:
- Wrap the *atomic* visual payload (``\\includegraphics``, ``tabular`` body,
  algorithm/listing body) in a TikZ ``remember picture`` node so we can read
  the 2-D corner coordinates precisely.
- Place zref-savepos line markers at the top and bottom of each float so the
  "_cap" variant can span from the first pixel of the float to the last pixel
  of the caption.
- Emit a machine-readable ``ARXIVLAYOUT-ANCHOR`` typeout per anchor and a
  ``ARXIVLAYOUT-LABEL`` typeout that carries the semantic class + logical id
  so the downstream extractor can stitch anchors into labeled boxes without
  relying on name conventions alone.

The injector is deliberately conservative: if we cannot find an anchorable
element inside a float we skip it silently and leave the surrounding TeX
untouched.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path


PREAMBLE_INJECTION = r"""
% ========== arxiv-paper-layout-dataset injection start ==========
\errorcontextlines=0
\makeatletter
\@ifpackageloaded{zref-savepos}{}{\usepackage{zref-savepos}}
\@ifpackageloaded{zref-abspage}{}{\usepackage{zref-abspage}}

% Register abspage on the savepos label type so every \zsavepos stores abspage
% alongside posx/posy. zref-abspage only *defines* the property; savepos does
% not pick it up automatically.
\zref@addprop{savepos}{abspage}

% Line anchor: records (posx, posy, abspage) at the current typesetting point
% AND reports the enclosing typesetting width (\hsize).
%
% Intentionally does NOT call \leavevmode. In vertical mode (e.g., at the top
% of a float vbox), \leavevmode would force an empty first line and push
% \zsavepos down by \topskip, so the mark would land *below* the actual vbox
% top instead of ON it. A bare \zsavepos is a whatsit that TeX happily accepts
% in both H and V mode, and it captures the current pos in whichever mode we
% happen to be in -- which is the tight boundary we want.
\newcommand{\alxmark}[1]{%
  \zsavepos{#1}%
  \typeout{ARXIVLAYOUT-MARK id=#1 hsize=\the\hsize\space linewidth=\the\linewidth\space columnwidth=\the\columnwidth}%
}

% Anchor-based 2-D bbox for a display block.
%
% Strategy: measure the natural width/height/depth of the payload in a box,
% then emit one \zsavepos right *before* placing the box and another right
% *after*. The box's left edge is at the pre-anchor's x; right edge at the
% post-anchor's x; top at (pre.y + height); bottom at (pre.y - depth).
\newsavebox\alx@box
\newcommand{\alxwrap}[2]{%
  \begingroup
    \setbox\alx@box=\hbox{#2}%
    \typeout{ARXIVLAYOUT-BOX id=#1 width=\the\wd\alx@box\space height=\the\ht\alx@box\space depth=\the\dp\alx@box\space hsize=\the\hsize}%
    \leavevmode
    \zsavepos{#1-preL}%
    \copy\alx@box%
    \zsavepos{#1-postR}%
  \endgroup
}

% Emit page geometry at \AtBeginDocument rather than \AtEndDocument -- some
% journal classes (A&A's aa.cls, for instance) clean up document-end hooks
% before we can run. Emitting at begin-of-document is safe since \paperwidth
% etc. are fixed by the time \begin{document} runs.
% Track the "current float id" so hook-emitted anchors can be tied back to
% the manifest. The injector rewrites \gdef\alx@current@alg@id{<cap_id>}
% just before each \begin{algorithm}/\begin{listing}.
\def\alx@current@alg@id{unset}%

% Hook bodies defined OUTSIDE \AtBeginDocument so the # param token survives
% a single \def round-trip. \AtBeginDocument otherwise doubles each ##,
% turning #1 into ##1 in the stored body and breaking the final expansion.
\def\alx@install@algorithm2e@hook{%
  \@ifpackageloaded{algorithm2e}{%
    \@ifundefined{algocf@makethealgo}{}{%
      \let\alx@orig@a2e@makethealgo\algocf@makethealgo
      \def\algocf@makethealgo{%
        \vtop{%
          \alxmark{alx@flt@top@\alx@current@alg@id}%
          \ifthenelse{\equal{\csname @algocf@capt@\algocf@style\endcsname}{above}}%
            {\csname algocf@caption@\algocf@style\endcsname}{}%
          \csname @algocf@pre@\algocf@style\endcsname%
          \ifthenelse{\equal{\csname @algocf@capt@\algocf@style\endcsname}{top}}%
            {\csname algocf@caption@\algocf@style\endcsname}{}%
          \alxmark{alx@flt@bodytop@\alx@current@alg@id}%
          \box\algocf@algobox%
          \alxmark{alx@flt@bodybot@\alx@current@alg@id}%
          \ifthenelse{\equal{\csname @algocf@capt@\algocf@style\endcsname}{bottom}}%
            {\csname algocf@caption@\algocf@style\endcsname}{}%
          \csname @algocf@post@\algocf@style\endcsname%
          \ifthenelse{\equal{\csname @algocf@capt@\algocf@style\endcsname}{under}}%
            {\csname algocf@caption@\algocf@style\endcsname}{}%
          \alxmark{alx@flt@bot@\alx@current@alg@id}%
        }%
      }%
    }%
  }{}%
}

% Hook \@makecaption so we see where a caption's text actually ends, without
% having to rely on the in-source \alxmark span landing past trailing
% \vspace*{-...} / \label tokens after the user's \caption{...}. Stores the
% coords using \alx@current@alg@id (set by the injector right before every
% figure/table/algorithm/listing opening) so the right cap can be found by
% anchor name alone.
\def\alx@install@makecaption@hook{%
  \@ifundefined{@makecaption}{}{%
    \let\alx@orig@makecaption\@makecaption
    \long\def\@makecaption##1##2{%
      \alxmark{alx@cap@top@\alx@current@alg@id}%
      \alx@orig@makecaption{##1}{##2}%
      \alxmark{alx@cap@bot@\alx@current@alg@id}%
    }%
  }%
}

\def\alx@install@float@hook{%
  \@ifpackageloaded{float}{%
    \@ifundefined{float@makebox}{}{%
      \let\alx@orig@float@makebox\float@makebox
      \renewcommand{\float@makebox}[1]{%
        \vbox{\hsize=##1 \@parboxrestore
          \alxmark{alx@flt@top@\alx@current@alg@id}%
          \@fs@pre
          \@fs@iftopcapt
            \ifvoid\@floatcapt\else\unvbox\@floatcapt\par\@fs@mid\fi
            \alxmark{alx@flt@bodytop@\alx@current@alg@id}%
            \unvbox\@currbox
            \alxmark{alx@flt@bodybot@\alx@current@alg@id}%
          \else
            \alxmark{alx@flt@bodytop@\alx@current@alg@id}%
            \unvbox\@currbox
            \alxmark{alx@flt@bodybot@\alx@current@alg@id}%
            \ifvoid\@floatcapt\else\par\@fs@mid\unvbox\@floatcapt\fi
          \fi
          \par\@fs@post
          \alxmark{alx@flt@bot@\alx@current@alg@id}%
          \vskip\z@
        }%
      }%
    }%
  }{}%
}

\AtBeginDocument{%
  \typeout{ARXIVLAYOUT-PAGEINFO paperwidth=\the\paperwidth\space paperheight=\the\paperheight\space textwidth=\the\textwidth\space textheight=\the\textheight\space oddsidemargin=\the\oddsidemargin\space evensidemargin=\the\evensidemargin\space topmargin=\the\topmargin\space headheight=\the\headheight\space headsep=\the\headsep\space columnwidth=\the\columnwidth\space columnsep=\the\columnsep}%
  \alx@install@algorithm2e@hook%
  \alx@install@float@hook%
  \alx@install@makecaption@hook%
}
\makeatother
% ========== arxiv-paper-layout-dataset injection end ==========
"""


# Labels we emit. kind is the semantic class; scope is "body" or "cap".
# cap variants are always keyed under the float id, body variants under the
# inner content id.
LABEL_CLASSES = {
    "fig",
    "fig_cap",
    "table",
    "table_cap",
    "algorithm",
    "algorithm_cap",
    "listing",
    "listing_cap",
}


@dataclass
class LabeledAnchor:
    """A logical label backed by one or more raw zref anchors."""

    label_id: str            # eg fig_3
    kind: str                # eg fig
    float_id: str | None     # eg figure_0 (None for freestanding listings)
    anchor_names: list[str]  # zref anchor names we read coordinates from
    method: str              # "wrap" (2D corners) or "span" (top/bot pair, x from textwidth)

    def as_dict(self) -> dict:
        return {
            "label_id": self.label_id,
            "kind": self.kind,
            "float_id": self.float_id,
            "anchor_names": self.anchor_names,
            "method": self.method,
        }


@dataclass
class InjectionManifest:
    labels: list[LabeledAnchor] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"labels": [label.as_dict() for label in self.labels]}

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.as_dict(), indent=2), encoding="utf-8")


# --- Regexes ----------------------------------------------------------------

# Matches \begin{env}[opts] body \end{env}. Requires matching same env name.
# Non-greedy body; handles nested same-name envs poorly (we only care about
# common paper patterns where figures/tables are not nested).
def _env_regex(envname: str) -> re.Pattern:
    return re.compile(
        r"(?P<begin>\\begin\{" + re.escape(envname) + r"\*?\}(?:\[[^\]]*\])?)"
        r"(?P<body>.*?)"
        r"(?P<end>\\end\{" + re.escape(envname) + r"\*?\})",
        re.DOTALL,
    )


_INCLUDE_GRAPHICS = re.compile(
    r"\\includegraphics\*?(?:\s*\[[^\]]*\])?\s*\{[^{}]+\}",
    re.DOTALL,
)

_TABULAR = re.compile(
    r"\\begin\{tabular[*xy]?\}(?:\[[^\]]*\])?(?:\{[^{}]*\})?"
    r".*?\\end\{tabular[*xy]?\}",
    re.DOTALL,
)

_TIKZPICTURE = re.compile(
    r"\\begin\{tikzpicture\}(?:\[[^\]]*\])?"
    r".*?\\end\{tikzpicture\}",
    re.DOTALL,
)


def _strip_nested_floats(body: str) -> str:
    """Remove the *body* of nested floats from ``body`` so detection of
    top-level patterns (captions, minipages) doesn't get confused by what
    happens inside an inner algorithm/listing/figure/table block.
    """
    cleaned = body
    for env in ("figure", "table", "algorithm", "algorithm2e", "listing"):
        pattern = re.compile(
            r"\\begin\{" + env + r"\*?\}.*?\\end\{" + env + r"\*?\}",
            re.DOTALL,
        )
        cleaned = pattern.sub("", cleaned)
    # also strip comment lines
    cleaned = "\n".join(
        line for line in cleaned.splitlines() if not line.lstrip().startswith("%")
    )
    return cleaned


def _find_top_level_minipages_with_caption(body: str) -> list[tuple[int, int]]:
    """Find every ``\\begin{minipage}...\\end{minipage}`` block in ``body``
    that itself contains a top-level ``\\caption{…}``.

    Returns a list of (start, end) source offsets in *body*. The minipages
    are the ones we treat as sub-floats so a side-by-side
    ``\\begin{table}{minipage}{minipage}\\end{table}`` pattern emits one
    cap per minipage instead of one merged cap for the outer table.
    """
    out: list[tuple[int, int]] = []
    pattern = re.compile(
        r"\\begin\{minipage\}(?:\[[^\]]*\])?(?:\{[^{}]*\})?",
        re.DOTALL,
    )
    end_pattern = re.compile(r"\\end\{minipage\}")
    pos = 0
    while True:
        m = pattern.search(body, pos)
        if m is None:
            break
        # find matching \end{minipage} via depth counting
        depth = 1
        scan = m.end()
        while depth > 0:
            next_begin = pattern.search(body, scan)
            next_end = end_pattern.search(body, scan)
            if next_end is None:
                # malformed -- bail
                return out
            if next_begin is not None and next_begin.start() < next_end.start():
                depth += 1
                scan = next_begin.end()
            else:
                depth -= 1
                scan = next_end.end()
        inner = body[m.end(): scan - len("\\end{minipage}")]
        inner_clean = _strip_nested_floats(inner)
        # also strip any nested minipages from the inner so we don't double-count
        inner_no_minipage = re.sub(
            r"\\begin\{minipage\}.*?\\end\{minipage\}",
            "",
            inner_clean,
            flags=re.DOTALL,
        )
        if re.search(r"(?<!\\)\\caption\b", inner_no_minipage):
            out.append((m.start(), scan))
        pos = scan
    return out


def _has_caption_call(body: str) -> bool:
    """True if ``body`` calls ``\\caption{…}`` that belongs to its own
    enclosing float.
    """
    cleaned = _strip_nested_floats(body)
    return re.search(r"(?<!\\)\\caption\b", cleaned) is not None


def _find_caption_span(body: str) -> tuple[int, int] | None:
    """Return (start, end) indices of a ``\\caption{...}`` occurrence in body.

    Handles balanced braces in the caption argument. Returns None when no
    caption is found. Only the first caption is located.
    """
    m = re.search(r"\\caption\b", body)
    if m is None:
        return None
    # find the opening { of the mandatory arg (skip optional [...])
    i = m.end()
    while i < len(body) and body[i] in " \t":
        i += 1
    if i < len(body) and body[i] == "[":
        depth = 1
        i += 1
        while i < len(body) and depth > 0:
            if body[i] == "[":
                depth += 1
            elif body[i] == "]":
                depth -= 1
            i += 1
    while i < len(body) and body[i] in " \t\n":
        i += 1
    if i >= len(body) or body[i] != "{":
        return None
    depth = 1
    j = i + 1
    while j < len(body) and depth > 0:
        ch = body[j]
        if ch == "\\" and j + 1 < len(body):
            j += 2
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        j += 1
    if depth != 0:
        return None
    return m.start(), j


class LatexBBoxInjector:
    """Inject bbox anchors into a LaTeX document.

    Operates on an already-concatenated TeX string (caller is responsible for
    resolving \\input / \\include if they want those handled; for arxiv sources
    the main file typically contains the bulk of the floats).
    """

    def __init__(self) -> None:
        self.manifest = InjectionManifest()
        self._counters: dict[str, int] = {}

    # -- public API ---------------------------------------------------------

    def inject(self, tex: str) -> str:
        # Scan + rewrite floats FIRST so the preamble's comment text (which
        # mentions \begin{algorithm} etc. only as documentation) doesn't get
        # picked up by the env regex.
        tex = self._inject_floats(tex, env="figure", kind_body="fig")
        tex = self._inject_floats(tex, env="table", kind_body="table")
        tex = self._inject_floats(tex, env="algorithm", kind_body="algorithm")
        tex = self._inject_floats(tex, env="algorithm2e", kind_body="algorithm")
        tex = self._inject_floats(tex, env="listing", kind_body="listing")
        tex = self._inject_standalone_lstlisting(tex)
        tex = self._inject_preamble(tex)
        return tex

    # -- helpers ------------------------------------------------------------

    def _next_id(self, kind: str) -> str:
        n = self._counters.get(kind, 0) + 1
        self._counters[kind] = n
        return f"{kind}_{n}"

    def _inject_preamble(self, tex: str) -> str:
        if "\\begin{document}" not in tex:
            return tex
        return tex.replace(
            "\\begin{document}",
            PREAMBLE_INJECTION + "\n\\begin{document}",
            1,
        )

    def _inject_floats(self, tex: str, env: str, kind_body: str) -> str:
        """For floats like figure / table / algorithm:

        - Wrap every \\includegraphics / tabular / tikzpicture we find in the
          body with ``\\alxwrap`` so we get its exact 2-D corners.
        - Emit FOUR span anchors around the float for the ``*_cap`` label:
          two *outer* anchors before / after ``\\begin{env}...\\end{env}`` and
          two *inner* anchors right inside the env. Outer anchors give tight
          bounds for non-floating placements (``[H]``); inner anchors are the
          fallback used whenever the float migrates to a different page.
        """

        pattern = _env_regex(env)
        kind_cap = f"{kind_body}_cap"

        def sub(match: re.Match) -> str:
            begin = match.group("begin")
            body = match.group("body")
            end = match.group("end")

            float_id = self._next_id(f"{env}_float")

            # Atomic-body wrapping + cap-id allocation.
            #
            # For every figure/table/algorithm/listing float we ALWAYS set
            # \alx@current@alg@id (the shared "current float id" the LaTeX
            # hooks read), so that downstream \caption calls emit hook
            # anchors tied to THIS float's cap id. Otherwise the previous
            # float's cap id silently carries over, and a later caption in
            # an un-instrumented figure overwrites the stored coords of an
            # earlier cap anchor.
            #
            # We emit fig_cap / table_cap ONLY when either (a) we wrapped an
            # atomic body (includegraphics / tikzpicture / tabular), or (b)
            # the body contains a ``\caption{...}`` -- the caption hook will
            # give us real coords for the cap. Otherwise the figure is just
            # a wrapper around some other instrumented float (DDPM's two
            # side-by-side algorithm-bearing minipages) and adding a cap
            # label would just mis-paint the whole wrapper red.
            # Multi-cap subdivision: if a figure/table contains multiple
            # top-level minipages and each carries its own \caption (cs page
            # 10, llava table 4 left/right pattern), treat each minipage as
            # a sub-float with its own cap_id. Otherwise the outer float
            # would emit one cap merging both side-by-side panels.
            if env in ("figure", "table"):
                minipages = _find_top_level_minipages_with_caption(body)
                if len(minipages) >= 2:
                    new_body = self._wrap_multicap_minipages(
                        body, env, kind_body, kind_cap, float_id, minipages
                    )
                    return f"{begin}%\n{new_body}{end}%\n"

            wrapped = False
            has_caption = _has_caption_call(body)
            if env == "figure":
                trial = self._wrap_graphics(body, kind_body, float_id)
                if trial == body:
                    trial = self._wrap_tikzpictures(body, kind_body, float_id)
                wrapped = trial != body
                new_body = trial
            elif env == "table":
                trial = self._wrap_tabulars(body, kind_body, float_id)
                wrapped = trial != body
                new_body = trial
            elif env in ("algorithm", "algorithm2e"):
                cap_id = self._next_id(kind_cap)
                new_body = self._wrap_algorithm_body(body, kind_body, float_id, cap_id)
                wrapped = True
            elif env == "listing":
                cap_id = self._next_id(kind_cap)
                new_body = self._wrap_listing_body(body, kind_body, float_id, cap_id)
                wrapped = True
            else:
                new_body = body

            emit_cap = wrapped or (env in ("figure", "table") and has_caption)
            if not emit_cap:
                # figure/table wrapper with nothing to instrument: still set
                # \alx@current@alg@id to a fresh sentinel so our caption
                # hook (if it fires) doesn't write to a previous float's id.
                sentinel = self._next_id(f"{env}_unused")
                return (
                    f"\\makeatletter\\gdef\\alx@current@alg@id{{{sentinel}}}\\makeatother%\n"
                    f"{match.group(0)}"
                )

            if env in ("figure", "table"):
                cap_id = self._next_id(kind_cap)

            outer_top = f"{cap_id}-outer-top"
            outer_bot = f"{cap_id}-outer-bot"
            inner_top = f"{cap_id}-inner-top"
            inner_bot = f"{cap_id}-inner-bot"
            # For algorithm floats we also pick up the algorithm2e-specific
            # anchors (see injector preamble): alx@a2e@captop / capbot. When
            # the paper uses a different algorithm package, these will be
            # missing and the resolver falls back to the inner-top/bot pair.
            # anchor_names is a flat list of (top, bot) pairs tried in order.
            # Primary pair: the zsavepos emitted by our hook on either
            # algorithm2e's \algocf@makethealgo or the `float` package's
            # \float@makebox -- each records the exact top rule / bottom rule
            # of the rendered block. Secondary: the in-source alxmark inner
            # pair, used when neither hook fires (e.g., a `figure` that has
            # no custom float style). Tertiary: outer alxmark pair (kept for
            # informational purposes; text-flow position, not placement).
            primary_top = f"alx@flt@top@{cap_id}"
            primary_bot = f"alx@flt@bot@{cap_id}"
            anchor_names = [primary_top, primary_bot, inner_top, inner_bot, outer_top, outer_bot]
            self.manifest.labels.append(
                LabeledAnchor(
                    label_id=cap_id,
                    kind=kind_cap,
                    float_id=float_id,
                    anchor_names=anchor_names,
                    method="span",
                )
            )

            # For algorithm / listing floats we set \alx@current@alg@id so
            # the preamble hook (\algocf@makethealgo or \float@makebox) knows
            # which manifest id to tag its zsavepos anchors with. Uses \gdef
            # so the value survives the float's own \bgroup/\egroup; wrapped
            # in \makeatletter because ``@`` is "other"-catcode in document
            # body and the control-sequence name would otherwise break.
            # Every instrumented float sets \alx@current@alg@id so our hooks
            # (algorithm2e, float@makebox, @makecaption) can tie their
            # zsavepos anchors back to this cap_id. The \gdef is wrapped in
            # \makeatletter because ``@`` is "other" cat in document body.
            preamble = (
                f"\\makeatletter\\gdef\\alx@current@alg@id{{{cap_id}}}\\makeatother%\n"
            )
            # After each instrumented float we reset \alx@current@alg@id to a
            # neutral sentinel. Otherwise a stale value from the previous
            # float survives and gets reused by LaTeX internals (e.g.
            # \float@makebox fired for a different deferred block on a
            # later page, clobbering the anchor coords of the last float
            # whose id we actually cared about).
            reset_id = self._next_id(f"{env}_done")
            return (
                preamble
                + f"\\alxmark{{{outer_top}}}%\n"
                f"{begin}%\n"
                f"\\alxmark{{{inner_top}}}%\n"
                f"{new_body}"
                f"\n\\alxmark{{{inner_bot}}}%\n"
                f"{end}%\n"
                f"\\alxmark{{{outer_bot}}}%\n"
                f"\\makeatletter\\gdef\\alx@current@alg@id{{{reset_id}}}\\makeatother%\n"
            )

        return pattern.sub(sub, tex)

    def _wrap_multicap_minipages(
        self,
        body: str,
        env: str,
        kind_body: str,
        kind_cap: str,
        float_id: str,
        minipages: list[tuple[int, int]],
    ) -> str:
        """For a float (figure/table) whose body contains multiple top-level
        minipages each holding their own ``\\caption``, treat each minipage
        as an independent sub-float -- one ``cap_id`` and one set of body
        wraps per minipage. Result: cs page 10 emits ``table_cap_5`` and
        ``table_cap_6`` for the two side-by-side tables instead of a single
        ``table_cap_5`` covering both.
        """
        out_parts: list[str] = []
        cursor = 0
        for mp_start, mp_end in minipages:
            # passthrough whatever's between the previous cursor and this
            # minipage (e.g. \centering, whitespace, \hfill)
            out_parts.append(body[cursor:mp_start])

            # locate the inner of this minipage
            inner_begin_match = re.match(
                r"\\begin\{minipage\}(?:\[[^\]]*\])?(?:\{[^{}]*\})?",
                body[mp_start:mp_end],
            )
            assert inner_begin_match is not None
            mp_open_end = mp_start + inner_begin_match.end()
            mp_inner_end = mp_end - len("\\end{minipage}")
            mp_inner = body[mp_open_end:mp_inner_end]

            cap_id = self._next_id(kind_cap)
            # Each minipage gets its OWN sub-float id so the cap-vs-body
            # union in render.py doesn't pull in the sibling minipage's
            # body box and stretch the cap across both columns.
            sub_float_id = self._next_id(f"{env}_minipage")
            # wrap the atomic body (includegraphics / tikzpicture / tabular)
            if env == "figure":
                inner_wrapped = self._wrap_graphics(mp_inner, kind_body, sub_float_id)
                if inner_wrapped == mp_inner:
                    inner_wrapped = self._wrap_tikzpictures(mp_inner, kind_body, sub_float_id)
            else:  # table
                inner_wrapped = self._wrap_tabulars(mp_inner, kind_body, sub_float_id)

            # rebuild the minipage with cap_id pinning + alxmark span
            preamble = (
                f"\\makeatletter\\gdef\\alx@current@alg@id{{{cap_id}}}\\makeatother%\n"
            )
            outer_top = f"{cap_id}-outer-top"
            outer_bot = f"{cap_id}-outer-bot"
            inner_top = f"{cap_id}-inner-top"
            inner_bot = f"{cap_id}-inner-bot"
            primary_top = f"alx@flt@top@{cap_id}"
            primary_bot = f"alx@flt@bot@{cap_id}"
            self.manifest.labels.append(
                LabeledAnchor(
                    label_id=cap_id,
                    kind=kind_cap,
                    float_id=sub_float_id,
                    anchor_names=[primary_top, primary_bot, inner_top, inner_bot, outer_top, outer_bot],
                    method="span",
                )
            )
            reset_id = self._next_id(f"{env}_done")
            out_parts.append(
                preamble
                + f"\\alxmark{{{outer_top}}}%\n"
                + body[mp_start:mp_open_end] + "%\n"
                + f"\\alxmark{{{inner_top}}}%\n"
                + inner_wrapped
                + f"\n\\alxmark{{{inner_bot}}}%\n"
                + "\\end{minipage}%\n"
                + f"\\alxmark{{{outer_bot}}}%\n"
                + f"\\makeatletter\\gdef\\alx@current@alg@id{{{reset_id}}}\\makeatother%\n"
            )
            cursor = mp_end
        out_parts.append(body[cursor:])
        return "".join(out_parts)

    def _wrap_graphics(self, body: str, kind: str, float_id: str) -> str:
        def sub(match: re.Match) -> str:
            inner = match.group(0)
            label_id = self._next_id(kind)
            self.manifest.labels.append(
                LabeledAnchor(
                    label_id=label_id,
                    kind=kind,
                    float_id=float_id,
                    anchor_names=[f"{label_id}-preL", f"{label_id}-postR"],
                    method="wrap",
                )
            )
            return f"\\alxwrap{{{label_id}}}{{{inner}}}"

        return _INCLUDE_GRAPHICS.sub(sub, body)

    def _wrap_tikzpictures(self, body: str, kind: str, float_id: str) -> str:
        def sub(match: re.Match) -> str:
            inner = match.group(0)
            label_id = self._next_id(kind)
            self.manifest.labels.append(
                LabeledAnchor(
                    label_id=label_id,
                    kind=kind,
                    float_id=float_id,
                    anchor_names=[f"{label_id}-preL", f"{label_id}-postR"],
                    method="wrap",
                )
            )
            return f"\\alxwrap{{{label_id}}}{{{inner}}}"

        return _TIKZPICTURE.sub(sub, body)

    def _wrap_tabulars(self, body: str, kind: str, float_id: str) -> str:
        def sub(match: re.Match) -> str:
            inner = match.group(0)
            label_id = self._next_id(kind)
            self.manifest.labels.append(
                LabeledAnchor(
                    label_id=label_id,
                    kind=kind,
                    float_id=float_id,
                    anchor_names=[f"{label_id}-preL", f"{label_id}-postR"],
                    method="wrap",
                )
            )
            return f"\\alxwrap{{{label_id}}}{{{inner}}}"

        return _TABULAR.sub(sub, body)

    def _wrap_algorithm_body(self, body: str, kind: str, float_id: str, cap_id: str) -> str:
        """For ``algorithm`` floats we emit a body label whose anchors come
        from the float-assembly hooks (``\\algocf@makethealgo`` for
        algorithm2e, ``\\float@makebox`` for the plain ``algorithm`` /
        ``float`` combo). Those anchors — ``alx@flt@bodytop/bodybot`` —
        bracket just the pseudocode box, excluding the title + separator
        rules. We also emit a pair of textual ``\\alxmark`` anchors as a
        fallback for oddball algorithm packages that don't load ``float``.
        """

        label_id = self._next_id(kind)
        body_top = f"alx@flt@bodytop@{cap_id}"
        body_bot = f"alx@flt@bodybot@{cap_id}"
        fallback_top = f"{label_id}-fallback-top"
        fallback_bot = f"{label_id}-fallback-bot"

        self.manifest.labels.append(
            LabeledAnchor(
                label_id=label_id,
                kind=kind,
                float_id=float_id,
                # primary: hook-assembled anchors (pseudocode body only)
                # fallback: \alxmark anchors placed in source
                anchor_names=[body_top, body_bot, fallback_top, fallback_bot],
                method="span",
            )
        )

        caption_end = _find_caption_span(body)
        if caption_end is None:
            return f"\n\\alxmark{{{fallback_top}}}%\n{body}\n\\alxmark{{{fallback_bot}}}%\n"

        cap_start, cap_end = caption_end
        pre = body[:cap_start].strip()
        post = body[cap_end:].strip()
        if len(pre) < len(post):
            # caption at top -> body is after caption
            return (
                body[:cap_end]
                + f"\n\\alxmark{{{fallback_top}}}%\n"
                + body[cap_end:]
                + f"\n\\alxmark{{{fallback_bot}}}%\n"
            )
        else:
            # caption at bottom -> body is before caption
            return (
                f"\n\\alxmark{{{fallback_top}}}%\n"
                + body[:cap_start]
                + f"\n\\alxmark{{{fallback_bot}}}%\n"
                + body[cap_start:]
            )

    def _wrap_listing_body(self, body: str, kind: str, float_id: str, cap_id: str = None) -> str:
        """Same strategy as the algorithm body: prefer float@makebox hook
        anchors, fall back to \\alxmark span around the caption."""

        label_id = self._next_id(kind)
        # Primary pair via \float@makebox hook (same naming scheme as
        # algorithms). cap_id is the enclosing listing_cap id.
        if cap_id is not None:
            primary_top = f"alx@flt@bodytop@{cap_id}"
            primary_bot = f"alx@flt@bodybot@{cap_id}"
        else:
            primary_top = primary_bot = "__missing__"
        fallback_top = f"{label_id}-fallback-top"
        fallback_bot = f"{label_id}-fallback-bot"
        self.manifest.labels.append(
            LabeledAnchor(
                label_id=label_id,
                kind=kind,
                float_id=float_id,
                anchor_names=[primary_top, primary_bot, fallback_top, fallback_bot],
                method="span",
            )
        )
        caption_end = _find_caption_span(body)
        if caption_end is None:
            return f"\n\\alxmark{{{fallback_top}}}%\n{body}\n\\alxmark{{{fallback_bot}}}%\n"
        cap_start, cap_end = caption_end
        pre = body[:cap_start].strip()
        post = body[cap_end:].strip()
        if len(pre) < len(post):
            return (
                body[:cap_end]
                + f"\n\\alxmark{{{fallback_top}}}%\n"
                + body[cap_end:]
                + f"\n\\alxmark{{{fallback_bot}}}%\n"
            )
        return (
            f"\n\\alxmark{{{fallback_top}}}%\n"
            + body[:cap_start]
            + f"\n\\alxmark{{{fallback_bot}}}%\n"
            + body[cap_start:]
        )

    def _inject_standalone_lstlisting(self, tex: str) -> str:
        """lstlisting / minted outside of a ``listing`` float. These get no
        ``_cap`` label."""

        def inject(env: str) -> str:
            nonlocal tex
            pattern = re.compile(
                r"(\\begin\{" + re.escape(env) + r"\}(?:\[[^\]]*\])?)"
                r"(.*?)"
                r"(\\end\{" + re.escape(env) + r"\})",
                re.DOTALL,
            )

            def sub(match: re.Match) -> str:
                # Skip lstlistings that live inside an already-instrumented
                # algorithm / listing float (perceiver wraps an lstlisting
                # inside an algorithm env; we want the algorithm label to
                # own the bbox, not a duplicate listing on top of it).
                context_before = tex[max(0, match.start() - 600): match.start()]
                if re.search(r"alx@flt@bodytop@(listing|algorithm)_", context_before):
                    return match.group(0)
                if re.search(r"\\alxmark\{(listing|algorithm)_\d+-fallback-top\}", context_before):
                    return match.group(0)
                label_id = self._next_id("listing")
                top = f"{label_id}-top"
                bot = f"{label_id}-bot"
                self.manifest.labels.append(
                    LabeledAnchor(
                        label_id=label_id,
                        kind="listing",
                        float_id=None,
                        anchor_names=[top, bot],
                        method="span",
                    )
                )
                # Force a paragraph break before the top mark so it lands in
                # v-mode at the upcoming column's left edge, not at the
                # *current* x-position of whatever inline text preceded the
                # \begin{lstlisting} (e.g. ``\textbf{Formal-to-formal:}``
                # immediately before the lstlisting in LLaVA's qualitative
                # appendix). Without the \par, the mark records an x in the
                # middle of that bold text and the bbox slides off the
                # listing's actual left edge.
                return (
                    f"\\par\\noindent\\alxmark{{{top}}}%\n{match.group(0)}"
                    f"\n\\par\\noindent\\alxmark{{{bot}}}%"
                )

            return pattern.sub(sub, tex)

        for env in ("lstlisting",):
            tex = inject(env)
        return tex


def inject_file(src: Path, dest: Path, manifest_path: Path | None = None) -> InjectionManifest:
    injector = LatexBBoxInjector()
    tex = src.read_text(encoding="utf-8", errors="ignore")
    out = injector.inject(tex)
    dest.write_text(out, encoding="utf-8")
    if manifest_path is not None:
        injector.manifest.save(manifest_path)
    return injector.manifest


class MultiFileInjector:
    """Apply the bbox injection to a whole source tree.

    Floats can be defined in \\input / \\include / \\subfile fragments that do
    not contain ``\\begin{document}`` themselves. We still need to inject the
    wrappers there so the corresponding anchors get saved; the preamble
    injection only goes into the file that owns ``\\begin{document}``.
    """

    def __init__(self) -> None:
        self.injector = LatexBBoxInjector()

    @property
    def manifest(self) -> InjectionManifest:
        return self.injector.manifest

    def inject_tree(self, src_root: Path, exts: tuple[str, ...] = (".tex",)) -> list[Path]:
        # Two-pass: rewrite every fragment first (so one shared injector
        # counter assigns sequential ids across the whole tree), then
        # prepend the preamble into the file that owns \begin{document}.
        paths = [p for p in src_root.rglob("*") if p.is_file() and p.suffix in exts]
        # stable ordering: main-doc first, then everything else by path
        paths.sort(key=lambda p: (not self._has_begin_document(p), p.as_posix()))
        modified: list[Path] = []
        for path in paths:
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            new = self._inject_floats_only(text)
            if "\\begin{document}" in new:
                new = self.injector._inject_preamble(new)
            if new != text:
                path.write_text(new, encoding="utf-8")
                modified.append(path)
        return modified

    def _has_begin_document(self, path: Path) -> bool:
        try:
            return "\\begin{document}" in path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return False

    def _inject_floats_only(self, tex: str) -> str:
        inj = self.injector
        tex = inj._inject_floats(tex, env="figure", kind_body="fig")
        tex = inj._inject_floats(tex, env="table", kind_body="table")
        tex = inj._inject_floats(tex, env="algorithm", kind_body="algorithm")
        tex = inj._inject_floats(tex, env="algorithm2e", kind_body="algorithm")
        tex = inj._inject_floats(tex, env="listing", kind_body="listing")
        tex = inj._inject_standalone_lstlisting(tex)
        return tex
