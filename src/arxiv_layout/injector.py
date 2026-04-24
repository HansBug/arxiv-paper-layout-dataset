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
\AtBeginDocument{%
  \typeout{ARXIVLAYOUT-PAGEINFO paperwidth=\the\paperwidth\space paperheight=\the\paperheight\space textwidth=\the\textwidth\space textheight=\the\textheight\space oddsidemargin=\the\oddsidemargin\space evensidemargin=\the\evensidemargin\space topmargin=\the\topmargin\space headheight=\the\headheight\space headsep=\the\headsep\space columnwidth=\the\columnwidth\space columnsep=\the\columnsep}%
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
        tex = self._inject_preamble(tex)
        tex = self._inject_floats(tex, env="figure", kind_body="fig")
        tex = self._inject_floats(tex, env="table", kind_body="table")
        tex = self._inject_floats(tex, env="algorithm", kind_body="algorithm")
        tex = self._inject_floats(tex, env="algorithm2e", kind_body="algorithm")
        tex = self._inject_floats(tex, env="listing", kind_body="listing")
        tex = self._inject_standalone_lstlisting(tex)
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

            # decide what the atomic "body" is for this env.
            if env in ("figure",):
                new_body = self._wrap_graphics(body, kind_body, float_id)
                # if figure contains no \includegraphics, fall back to wrapping
                # a standalone tikzpicture (common in math papers).
                if new_body == body:
                    new_body = self._wrap_tikzpictures(body, kind_body, float_id)
            elif env in ("table",):
                new_body = self._wrap_tabulars(body, kind_body, float_id)
            elif env in ("algorithm", "algorithm2e"):
                new_body = self._wrap_algorithm_body(body, kind_body, float_id)
            elif env in ("listing",):
                new_body = self._wrap_listing_body(body, kind_body, float_id)
            else:
                new_body = body

            cap_id = self._next_id(kind_cap)
            outer_top = f"{cap_id}-outer-top"
            outer_bot = f"{cap_id}-outer-bot"
            inner_top = f"{cap_id}-inner-top"
            inner_bot = f"{cap_id}-inner-bot"
            self.manifest.labels.append(
                LabeledAnchor(
                    label_id=cap_id,
                    kind=kind_cap,
                    float_id=float_id,
                    # order: outer-top, outer-bot, inner-top, inner-bot
                    anchor_names=[outer_top, outer_bot, inner_top, inner_bot],
                    method="span",
                )
            )

            return (
                f"\\alxmark{{{outer_top}}}%\n"
                f"{begin}%\n"
                f"\\alxmark{{{inner_top}}}%\n"
                f"{new_body}"
                f"\n\\alxmark{{{inner_bot}}}%\n"
                f"{end}%\n"
                f"\\alxmark{{{outer_bot}}}"
            )

        return pattern.sub(sub, tex)

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

    def _wrap_algorithm_body(self, body: str, kind: str, float_id: str) -> str:
        """For ``algorithm`` / ``listing`` floats we use top/bot span markers
        to bracket the pseudocode / code (i.e., the visual body), skipping
        over the caption which can appear at either end of the float.

        ``body`` here is the *inside* of ``\\begin{algorithm}...\\end{algorithm}``.
        We look for ``\\caption{...}``; if it appears, the code region is
        everything *outside* the caption. We put our markers accordingly:

        - caption at top: top mark goes AFTER the caption, bot mark at end
        - caption at bottom: top mark at start, bot mark BEFORE the caption
        - no caption: wrap the whole body
        """

        label_id = self._next_id(kind)
        top = f"{label_id}-top"
        bot = f"{label_id}-bot"
        self.manifest.labels.append(
            LabeledAnchor(
                label_id=label_id,
                kind=kind,
                float_id=float_id,
                anchor_names=[top, bot],
                method="span",
            )
        )

        caption_end = _find_caption_span(body)
        if caption_end is None:
            return f"\n\\alxmark{{{top}}}%\n{body}\n\\alxmark{{{bot}}}%\n"

        cap_start, cap_end = caption_end
        # Decide whether caption is "at top" or "at bottom" based on how much
        # content appears before vs after it.
        pre = body[:cap_start].strip()
        post = body[cap_end:].strip()
        if len(pre) < len(post):
            # caption at top -> body is after caption
            return (
                body[:cap_end]
                + f"\n\\alxmark{{{top}}}%\n"
                + body[cap_end:]
                + f"\n\\alxmark{{{bot}}}%\n"
            )
        else:
            # caption at bottom -> body is before caption
            return (
                f"\n\\alxmark{{{top}}}%\n"
                + body[:cap_start]
                + f"\n\\alxmark{{{bot}}}%\n"
                + body[cap_start:]
            )

    def _wrap_listing_body(self, body: str, kind: str, float_id: str) -> str:
        # Same strategy as algorithm: use a top/bot span excluding caption.
        return self._wrap_algorithm_body(body, kind, float_id)

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
                # Skip if inside a listing float -- crude but effective:
                # we already processed listing floats, so their contents got
                # top/bot markers; detect presence of ``\\alxmark{listing_``
                # surrounding the lstlisting and bail.
                context_before = tex[max(0, match.start() - 400): match.start()]
                if re.search(r"\\alxmark\{listing_\d+-top\}", context_before):
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
                return (
                    f"\\alxmark{{{top}}}%\n{match.group(0)}\n\\alxmark{{{bot}}}%"
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
        modified: list[Path] = []
        for path in src_root.rglob("*"):
            if not path.is_file() or path.suffix not in exts:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if "\\begin{document}" in text:
                new = self.injector.inject(text)
            else:
                new = self._inject_fragment(text)
            if new != text:
                path.write_text(new, encoding="utf-8")
                modified.append(path)
        return modified

    def _inject_fragment(self, tex: str) -> str:
        # reuse every inject step except the preamble addition
        tex = self.injector._inject_floats(tex, env="figure", kind_body="fig")
        tex = self.injector._inject_floats(tex, env="table", kind_body="table")
        tex = self.injector._inject_floats(tex, env="algorithm", kind_body="algorithm")
        tex = self.injector._inject_floats(tex, env="algorithm2e", kind_body="algorithm")
        tex = self.injector._inject_floats(tex, env="listing", kind_body="listing")
        tex = self.injector._inject_standalone_lstlisting(tex)
        return tex
