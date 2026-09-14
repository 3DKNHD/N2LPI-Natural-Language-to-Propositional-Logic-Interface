#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate N2LPI train/test JSON: split leakage, schema, complexity rubric."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "training_dataset.json"
TEST_PATH = ROOT / "data" / "test_dataset.json"

FORMULA_OPS = ("¬", "∧", "∨", "→", "↔")


def structural_complexity(formula: str) -> str:
    """Map a formula to simple/medium/complex from structure, not sentence length."""
    f = formula.strip()
    atoms = len(re.findall(r"[A-Z]", f))
    n_bin = sum(f.count(op) for op in ("∧", "∨", "→", "↔"))
    wraps_impl = bool(re.search(r"¬\s*\(.*[→↔]", f))
    negated_compound = bool(re.search(r"¬\s*\(", f))
    impl_with_neg_atom = bool(re.search(r"¬[A-Z]\s*[→↔]|[→↔]\s*¬[A-Z]", f))

    if atoms >= 4 or n_bin >= 3 or wraps_impl:
        return "complex"
    if atoms >= 3 or n_bin >= 2 or negated_compound or impl_with_neg_atom:
        return "medium"
    return "simple"


def formula_symbols(formula: str) -> list[str]:
    found = []
    for op in FORMULA_OPS:
        if op in formula and op not in found:
            found.append(op)
    return found


def formula_vars(formula: str) -> set[str]:
    return set(re.findall(r"[A-Z]", formula))


def load_items(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"{path} must be a JSON list")
    return data


def check_item(item: dict, file_label: str, idx: int) -> list[str]:
    errors = []
    prefix = f"{file_label}[{idx}] id={item.get('id')}"
    for key in ("id", "sentence", "propositions", "formula", "complexity", "connectives"):
        if key not in item:
            errors.append(f"{prefix}: missing field {key}")
            return errors
    if not isinstance(item["propositions"], dict):
        errors.append(f"{prefix}: propositions must be an object")
        return errors
    fvars = formula_vars(item["formula"])
    pkeys = set(item["propositions"].keys())
    if fvars != pkeys:
        errors.append(f"{prefix}: formula vars {sorted(fvars)} != proposition keys {sorted(pkeys)}")
    for key, val in item["propositions"].items():
        if not isinstance(val, str):
            errors.append(f"{prefix}: proposition {key} is not a string")
        elif val != val.lower():
            errors.append(f"{prefix}: proposition {key} is not lowercase: {val!r}")
    expected_ops = formula_symbols(item["formula"])
    got_ops = list(item["connectives"])
    if set(got_ops) != set(expected_ops):
        errors.append(f"{prefix}: connectives {got_ops} != formula ops {expected_ops}")
    expected_cx = structural_complexity(item["formula"])
    if item["complexity"] != expected_cx:
        errors.append(
            f"{prefix}: complexity {item['complexity']!r} != structural {expected_cx!r} for {item['formula']}"
        )
    if not str(item["sentence"]).strip().endswith("."):
        errors.append(f"{prefix}: sentence should end with a period")
    return errors


def validate(train: list[dict], test: list[dict]) -> list[str]:
    errors: list[str] = []
    train_ids = [x["id"] for x in train]
    test_ids = [x["id"] for x in test]
    if len(train_ids) != len(set(train_ids)):
        errors.append("duplicate ids in training_dataset.json")
    if len(test_ids) != len(set(test_ids)):
        errors.append("duplicate ids in test_dataset.json")
    overlap_ids = set(train_ids) & set(test_ids)
    if overlap_ids:
        errors.append(f"train/test id overlap ({len(overlap_ids)}): {sorted(overlap_ids)[:20]}")
    train_sents = {x["sentence"].strip().lower() for x in train}
    test_sents = {x["sentence"].strip().lower() for x in test}
    overlap_sents = train_sents & test_sents
    if overlap_sents:
        sample = list(sorted(overlap_sents))[:5]
        errors.append(f"train/test sentence overlap ({len(overlap_sents)}): {sample}")
    for i, item in enumerate(train):
        errors.extend(check_item(item, "train", i))
    for i, item in enumerate(test):
        errors.extend(check_item(item, "test", i))
    return errors


def main() -> int:
    train = load_items(TRAIN_PATH)
    test = load_items(TEST_PATH)
    errors = validate(train, test)
    print(f"train={len(train)} test={len(test)}")
    if errors:
        print(f"FAIL ({len(errors)} issues)")
        for err in errors[:80]:
            print(" -", err)
        if len(errors) > 80:
            print(f" ... and {len(errors) - 80} more")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
