"""Regression tests for the YOLO-export split picker.

These pin the contract the user relies on when shipping this dataset:

- Same ``(paper_id, page_id)`` -> same filename stem.
- Same stem -> same split forever (a naïve re-run must not reshuffle).
- Weights of ``(8, 1, 1)`` land each sample in the ratio the user asked
  for (tested over a large random sample).
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from export_yolo import build_stem, pick_split  # noqa: E402


def test_build_stem_is_deterministic_and_identifying():
    stem = build_stem("2604.21931v1_cs_seeing_fast_and_slow", 10)
    assert stem == "2604.21931v1_cs_seeing_fast_and_slow__page_010"
    # stem changes if either component changes
    assert stem != build_stem("2604.21931v1_cs_seeing_fast_and_slow", 11)
    assert stem != build_stem("2604.21800v1_mathph_variance_geometry_codes", 10)


def test_pick_split_is_deterministic():
    stem = "2604.21931v1_cs_seeing_fast_and_slow__page_010"
    splits = {pick_split(stem, (8, 1, 1)) for _ in range(200)}
    assert splits == {pick_split(stem, (8, 1, 1))}, "same stem must always yield the same split"
    assert len(splits) == 1


def test_pick_split_approximate_ratio():
    # 10 000 random stems should respect 8:1:1 closely (within 1.5 pts).
    rng = random.Random(0)
    counts = {"train": 0, "val": 0, "test": 0}
    for _ in range(10000):
        stem = f"arxiv_{rng.randint(0, 10**10)}__page_{rng.randint(0, 500):03d}"
        counts[pick_split(stem, (8, 1, 1))] += 1
    total = sum(counts.values())
    assert 0.78 <= counts["train"] / total <= 0.82
    assert 0.08 <= counts["val"] / total <= 0.12
    assert 0.08 <= counts["test"] / total <= 0.12


def test_weight_change_reshuffles():
    # Different weight ratios can move a given stem; that's intentional.
    stem = "arxiv_example__page_007"
    weights_a = (8, 1, 1)
    weights_b = (1, 1, 8)  # extreme: heavy test side
    got_a = {pick_split(stem, weights_a) for _ in range(100)}
    got_b = {pick_split(stem, weights_b) for _ in range(100)}
    assert len(got_a) == 1 and len(got_b) == 1
    # not asserting equality - we only require individual determinism.


def test_split_counts_balance_over_real_paper_ids():
    # Smoke: run 100 synthetic papers x 20 pages each, confirm we get ~80% train.
    rng = random.Random(42)
    n_train = n_val = n_test = 0
    for _ in range(100):
        paper_id = f"2604.{rng.randint(1000, 99999)}v1_{rng.choice(['cs', 'math', 'physics'])}"
        for page in range(1, 21):
            stem = build_stem(paper_id, page)
            split = pick_split(stem, (8, 1, 1))
            if split == "train":
                n_train += 1
            elif split == "val":
                n_val += 1
            else:
                n_test += 1
    total = n_train + n_val + n_test
    assert 0.75 <= n_train / total <= 0.85
    assert 0.05 <= n_val / total <= 0.15
    assert 0.05 <= n_test / total <= 0.15
