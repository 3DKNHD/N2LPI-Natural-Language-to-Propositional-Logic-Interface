#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Staged NL-to-logic pipeline: cues → boundary → precedence → AST, with beam."""

from __future__ import annotations

import contextvars
import re
from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from config import NLP_SPANISH, VERB_ENDINGS, logger
from logic_utils import son_formulas_equivalentes

from hybrid_parser import (
    AtomicNode,
    BinaryOpNode,
    FullRecursiveParser,
    LogicNode,
    PropositionRegistry,
    UnaryOpNode,
)

# High-precision cues are always treated as clause boundaries.
# Ambiguous y/o/e/u/pero are classified by the boundary model.
CUE_PATTERNS: Sequence[Tuple[str, str, str, bool]] = (
    (r"no es posible que", "NEG_BLOCK", "¬", True),
    (r"no es cierto que", "NEG_BLOCK", "¬", True),
    (r"no es verdad que", "NEG_BLOCK", "¬", True),
    (r"no ocurre que", "NEG_BLOCK", "¬", True),
    (r"no es el caso que", "NEG_BLOCK", "¬", True),
    (r"no se cumple que", "NEG_BLOCK", "¬", True),
    (r"es falso que", "NEG_BLOCK", "¬", True),
    (r"es mentira que", "NEG_BLOCK", "¬", True),
    (r"si y solo si", "IFF", "↔", True),
    (r"cuando y solamente cuando", "IFF", "↔", True),
    (r"cuando y solo cuando", "IFF", "↔", True),
    (r"es equivalente a", "IFF", "↔", True),
    (r"equivale a", "IFF", "↔", True),
    (r"no las dos cosas a la vez", "XOR_TAIL", "xor", True),
    (r"pero no ambos", "XOR_TAIL", "xor", True),
    (r"no ambas(?:\s+cosas)?", "XOR_TAIL", "xor", True),
    (r"no ambos", "XOR_TAIL", "xor", True),
    (r"no las dos", "XOR_TAIL", "xor", True),
    (r"a pesar de(?:\s+que)?", "AND", "∧", True),
    (r"sin embargo", "AND", "∧", True),
    (r"no obstante", "AND", "∧", True),
    (r"solo si", "ONLY_IF", "→", True),
    (r"siempre que", "IF", "→", True),
    (r"o bien", "OR", "∨", True),
    (r"no solo", "NO_SOLO", "∧", True),
    (r"sino(?:\s+también)?", "SINO", "∧", True),
    (r"aunque", "AND", "∧", True),
    (r"entonces", "THEN", "→", True),
    (r"cuando", "IF", "→", True),
    (r"\bni\b", "NOR", "∧", True),
    (r"\bpero\b", "AND", "∧", False),
    (r"\bsi\b", "IF", "→", True),
    (r"\by\b", "AND", "∧", False),
    (r"\be\b", "AND", "∧", False),
    (r"\bo\b", "OR", "∨", False),
    (r"\bu\b", "OR", "∨", False),
)

# Second reading when «y» and «o» are both clause boundaries.
# The default tree still splits on «o» first. This only builds the other grouping.
_AND_WIDER = contextvars.ContextVar("n2lpi_and_wider", default=False)

PREC = {
    "NEG_BLOCK": 100,
    "XOR_TAIL": 95,
    "IFF": 90,
    "ONLY_IF": 80,
    "NO_SOLO": 75,
    "NOR": 70,
    "IF": 65,
    "THEN": 65,
    "UNLESS": 82,
    "AUNQUE": 55,
    "SINO": 50,
    "OR": 40,
    "AND": 30,
    "NEG": 25,
}

KIND_KEYS = [
    "NEG_BLOCK", "IFF", "ONLY_IF", "UNLESS", "IF", "THEN", "OR", "AND", "NOR",
    "XOR_TAIL", "NO_SOLO", "SINO", "NEG", "AUNQUE",
]

KIND_TO_OP = {
    "NEG_BLOCK": "¬",
    "IFF": "↔",
    "ONLY_IF": "→",
    "UNLESS": "→",
    "IF": "→",
    "THEN": "→",
    "OR": "∨",
    "AND": "∧",
    "NOR": "∧",
    "XOR_TAIL": "xor",
    "NO_SOLO": "∧",
    "SINO": "∧",
    "NEG": "¬",
    "AUNQUE": "∧",
}

ML_FORCE_KINDS = {
    "NEG_BLOCK", "IFF", "ONLY_IF", "UNLESS", "IF", "THEN", "NOR", "XOR_TAIL",
    "NO_SOLO", "NEG", "AUNQUE",
}

_COMMON_VERBS = {
    "es", "son", "fue", "era", "está", "están", "hay", "tiene", "tienen",
    "hace", "hizo", "llueve", "nieva", "llegó", "llega", "sale", "entras",
    "sales", "apruebas", "suspendes", "estudia", "trabaja",
}


@dataclass
class Cue:
    start: int
    end: int
    kind: str
    op: str
    text: str
    force: bool
    prob: float = 1.0
    is_boundary: bool = True

    def key(self) -> Tuple[int, int, str]:
        return (self.start, self.end, self.kind)


def detect_cues(sentence: str) -> List[Cue]:
    """Stage 1: lexical connective mentions, longest match wins on overlaps."""
    s = sentence
    found: List[Cue] = []
    occupied = [False] * (len(s) + 1)
    for pattern, kind, op, force in CUE_PATTERNS:
        for match in re.finditer(pattern, s, flags=re.IGNORECASE):
            start, end = match.start(), match.end()
            if any(occupied[start:end]):
                continue
            for i in range(start, end):
                occupied[i] = True
            found.append(Cue(start, end, kind, op, match.group(0), force))
    found.sort(key=lambda c: (c.start, -(c.end - c.start)))
    return found


def iter_ngram_spans(sentence: str, max_n: int = 6) -> List[Tuple[int, int, str]]:
    """Word n-grams (1..max_n) as candidate connective spans."""
    tokens = list(re.finditer(r"\S+", sentence))
    spans = []
    for n in range(1, max_n + 1):
        for i in range(len(tokens) - n + 1):
            start = tokens[i].start()
            end = tokens[i + n - 1].end()
            text = sentence[start:end]
            if re.fullmatch(r"[\W_]+", text or ""):
                continue
            spans.append((start, end, text))
    return spans


def nms_cues(cues: Sequence[Cue]) -> List[Cue]:
    """Drop overlaps. Kind precedence first (IFF beats IF), then confidence, then length."""
    ordered = sorted(
        cues,
        key=lambda c: (-PREC.get(c.kind, 0), -c.prob, -(c.end - c.start), c.start),
    )
    if not ordered:
        return []
    occupied = [False] * (max(c.end for c in ordered) + 1)
    kept: List[Cue] = []
    for cue in ordered:
        if any(occupied[cue.start:cue.end]):
            continue
        for i in range(cue.start, cue.end):
            occupied[i] = True
        kept.append(cue)
    kept.sort(key=lambda c: (c.start, -(c.end - c.start)))
    return kept


_EXTRA_ENDINGS = ("e", "a", "an", "en", "ó", "ió", "aba", "ía")


def looks_verbal(text: str) -> bool:
    text = (text or "").strip()
    if not text:
        return False
    if NLP_SPANISH is not None:
        try:
            if any(t.pos_ in {"VERB", "AUX"} for t in NLP_SPANISH(text)):
                return True
        except Exception:
            pass
    words = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
    endings = tuple(VERB_ENDINGS) + _EXTRA_ENDINGS
    skip = {"para", "este", "esta", "modo", "casa", "tinta", "papel", "la", "el", "los", "las", "una", "uno"}
    for w in words:
        if w in _COMMON_VERBS:
            return True
        if w in skip:
            continue
        if len(w) > 3 and any(w.endswith(end) for end in endings):
            return True
    return False


def extract_boundary_features(sentence: str, cue: Cue, nlp=None) -> Dict[str, float]:
    """Window features for stage 2 (is this cue a clause boundary?)."""
    nlp = nlp if nlp is not None else NLP_SPANISH
    left = sentence[:cue.start]
    right = sentence[cue.end:]
    left_w = re.findall(r"\w+", left.lower(), flags=re.UNICODE)
    right_w = re.findall(r"\w+", right.lower(), flags=re.UNICODE)
    feats: Dict[str, float] = {
        "rel_pos": cue.start / max(len(sentence), 1),
        "span_len": (cue.end - cue.start) / max(len(sentence), 1),
        "left_words": float(len(left_w)),
        "right_words": float(len(right_w)),
        "left_verbal": float(looks_verbal(left)),
        "right_verbal": float(looks_verbal(right)),
        "both_verbal": float(looks_verbal(left) and looks_verbal(right)),
        "left_comma": float("," in left[-12:]),
        "right_comma": float("," in right[:12]),
        "force": float(cue.force),
        "at_start": float(not left.strip()),
        "near_end": float(len(right.strip()) < 8),
        "left_starts_det": float(bool(re.match(r"\s*(el|la|los|las|un|una)\b", left.strip()[-20:], re.I))),
        "right_starts_det": float(bool(re.match(r"\s*(el|la|los|las|un|una)\b", right, re.I))),
        "left_short": float(len(left_w) <= 3),
        "right_short": float(len(right_w) <= 3),
    }
    for kind in KIND_KEYS:
        feats[f"kind_{kind}"] = float(cue.kind == kind)
    if nlp is not None:
        try:
            doc = nlp(sentence)
            left_pos = {t.pos_ for t in doc if t.idx < cue.start}
            right_pos = {t.pos_ for t in doc if t.idx >= cue.end}
            feats["left_spacy_verb"] = float("VERB" in left_pos or "AUX" in left_pos)
            feats["right_spacy_verb"] = float("VERB" in right_pos or "AUX" in right_pos)
        except Exception:
            feats["left_spacy_verb"] = feats["left_verbal"]
            feats["right_spacy_verb"] = feats["right_verbal"]
    else:
        feats["left_spacy_verb"] = feats["left_verbal"]
        feats["right_spacy_verb"] = feats["right_verbal"]
    return feats


FEATURE_KEYS: List[str] = sorted(
    extract_boundary_features("el servidor responde y la app funciona.", Cue(22, 23, "AND", "∧", "y", False)).keys()
)


def vectorize_features(feats: Dict[str, float]) -> List[float]:
    return [float(feats.get(k, 0.0)) for k in FEATURE_KEYS]


def is_np_coordination(left: str, right: str) -> bool:
    """Subject/NP coordination: 'el taxi o el autobús llegaron'."""
    lw = re.findall(r"\w+", left.lower(), flags=re.UNICODE)
    rw = re.findall(r"\w+", right.lower(), flags=re.UNICODE)
    lv, rv = looks_verbal(left), looks_verbal(right)
    if (not lv) and rv and 1 <= len(lw) <= 4:
        return True
    if lv and (not rv) and 1 <= len(rw) <= 4:
        return True
    return False


def gold_is_boundary(sentence: str, cue: Cue, item: Optional[dict]) -> int:
    """Derive a 0/1 label from gold propositions/formula when available."""
    if cue.force:
        return 1
    left = sentence[:cue.start]
    right = sentence[cue.end:]
    if looks_verbal(left) and looks_verbal(right):
        if not item:
            return 1
    elif is_np_coordination(left, right) and cue.kind in {"AND", "OR"}:
        if not item:
            return 1
    if not item:
        return 0
    left_l, right_l = left.lower(), right.lower()
    props = [str(p).lower() for p in item.get("propositions", {}).values()]
    only_left = [p for p in props if p and p in left_l and p not in right_l]
    only_right = [p for p in props if p and p not in left_l and p in right_l]
    if only_left and only_right:
        return 1
    for p in props:
        if p and p[: max(3, len(p) // 2)] in left_l and p[len(p) // 2 :] in right_l:
            return 1
    if looks_verbal(left) and looks_verbal(right):
        op = cue.op
        formula = item.get("formula") or ""
        if op in formula or (cue.kind == "AND" and "∧" in formula) or (cue.kind == "OR" and "∨" in formula):
            return 1
    if is_np_coordination(left, right) and cue.kind in {"AND", "OR"}:
        formula = item.get("formula") or ""
        if (cue.kind == "OR" and "∨" in formula) or (cue.kind == "AND" and "∧" in formula):
            return 1
    return 0


def _strip_span(s: str, i: int, j: int) -> Tuple[int, int]:
    while i < j and s[i].isspace():
        i += 1
    while j > i and s[j - 1] in " \t.!?,;:":
        j -= 1
    return i, j


def _cues_in(cues: Sequence[Cue], i: int, j: int) -> List[Cue]:
    return [c for c in cues if c.is_boundary and c.start >= i and c.start < j]


_COORD_NEXT = {"y", "e", "o", "u", "entonces", "si", "cuando", "pero", "ni", "sino", "por", "de"}


def _comma_and_cues(s: str, i: int, j: int, existing: Sequence[Cue]) -> List[Cue]:
    """Treat a comma between two verbal clauses as AND, unless a coordinator follows."""
    extra: List[Cue] = []
    k = s.find(",", i, j)
    while k != -1:
        if any(c.start <= k < c.end for c in existing):
            k = s.find(",", k + 1, j)
            continue
        rest = s[k + 1:j].lstrip()
        first = re.match(r"\w+", rest, flags=re.UNICODE)
        if first and first.group(0).lower() in _COORD_NEXT:
            k = s.find(",", k + 1, j)
            continue
        left, right = s[i:k], s[k + 1:j]
        if looks_verbal(left) and looks_verbal(right):
            extra.append(Cue(k, k + 1, "AND", "∧", ",", True, prob=0.7, is_boundary=True))
        k = s.find(",", k + 1, j)
    return extra


def _eff_prec(s: str, i: int, cue: Cue) -> int:
    p = PREC.get(cue.kind, 0)
    if _AND_WIDER.get() and cue.kind == "AND":
        p = max(p, PREC["OR"] + 1)
    if cue.kind == "AND" and s[i:cue.start].rstrip().endswith(","):
        return max(p, 45)
    return p


def parse_range(s: str, i: int, j: int, cues: Sequence[Cue], registry: PropositionRegistry) -> LogicNode:
    """Stage 3: precedence split on remaining boundaries, then recurse (stage 4 at leaves)."""
    i, j = _strip_span(s, i, j)
    if i >= j:
        return AtomicNode("∅")
    local = _cues_in(cues, i, j)
    local = sorted(local + _comma_and_cues(s, i, j, local), key=lambda c: (c.start, c.end))
    if not local:
        return _atomic(s[i:j], registry)

    # Leading "o bien" / "o" when another OR remains (XOR / "o A o B").
    lead = [c for c in local if c.start == i or s[i:c.start].strip() == ""]
    if lead and lead[0].kind == "OR":
        rest = [c for c in local if c is not lead[0]]
        if any(c.kind == "OR" for c in rest):
            return parse_range(s, lead[0].end, j, cues, registry)

    if lead and lead[0].kind == "AND":
        rest = [c for c in local if c is not lead[0]]
        head = re.sub(r"[^\wñ]+", "", (lead[0].text or "").lower())
        if any(c.kind == "AND" for c in rest) and head in {"tanto", "asi"}:
            return parse_range(s, lead[0].end, j, cues, registry)

    # Leading "entonces" after a comma split: drop it and parse the consequent.
    if lead and lead[0].kind == "THEN":
        return parse_range(s, lead[0].end, j, cues, registry)

    # Block negation wrapping the rest.
    for c in local:
        if c.kind == "NEG_BLOCK" and s[i:c.start].strip() == "":
            child = parse_range(s, c.end, j, cues, registry)
            return UnaryOpNode("¬", child)

    # XOR tail: (P ∨ Q) ∧ ¬(P ∧ Q)
    xor = [c for c in local if c.kind == "XOR_TAIL"]
    if xor:
        inner = parse_range(s, i, xor[0].start, cues, registry)
        if isinstance(inner, BinaryOpNode) and inner.op == "∨":
            return BinaryOpNode(
                "∧",
                inner,
                UnaryOpNode("¬", BinaryOpNode("∧", inner.left, inner.right)),
            )
        return inner

    # no solo A sino B
    no_solo = [c for c in local if c.kind == "NO_SOLO" and s[i:c.start].strip() == ""]
    sino = [c for c in local if c.kind == "SINO"]
    if no_solo and sino:
        left = parse_range(s, no_solo[0].end, sino[0].start, cues, registry)
        right = parse_range(s, sino[0].end, j, cues, registry)
        return BinaryOpNode("∧", left, right)

    # ni A ni B
    nors = [c for c in local if c.kind == "NOR"]
    if len(nors) >= 2 and s[i:nors[0].start].strip() in {"", "no"}:
        left = parse_range(s, nors[0].end, nors[1].start, cues, registry)
        right = parse_range(s, nors[1].end, j, cues, registry)
        return BinaryOpNode("∧", UnaryOpNode("¬", left), UnaryOpNode("¬", right))
    if len(nors) == 1 and s[i:nors[0].start].lower().strip().startswith("no"):
        left_text = s[i:nors[0].start]
        m = re.match(r"^no\s+", left_text, re.I)
        left_from = i + (m.end() if m else 0)
        left = parse_range(s, left_from, nors[0].start, cues, registry)
        right = parse_range(s, nors[0].end, j, cues, registry)
        return BinaryOpNode("∧", UnaryOpNode("¬", left), UnaryOpNode("¬", right))

    # aunque A, B  →  A ∧ B (comma marks the main clause)
    aunque = [c for c in local if c.kind == "AUNQUE"]
    if aunque:
        c = aunque[0]
        body_i = c.end if s[i:c.start].strip() == "" else i
        comma = s.find(",", max(c.end, body_i), j)
        if s[i:c.start].strip() == "" and comma != -1:
            return BinaryOpNode(
                "∧",
                parse_range(s, c.end, comma, cues, registry),
                parse_range(s, comma + 1, j, cues, registry),
            )
        if s[i:c.start].strip() == "":
            return parse_range(s, c.end, j, cues, registry)
        return BinaryOpNode(
            "∧",
            parse_range(s, i, c.start, cues, registry),
            parse_range(s, c.end, j, cues, registry),
        )

    # Outer si … entonces / si …, …  before any inner IFF.
    ifs = [c for c in local if c.kind == "IF"]
    thens = [c for c in local if c.kind == "THEN"]
    if ifs and s[i:ifs[0].start].strip() == "":
        head = ifs[0]
        inner_iff = [c for c in local if c.kind == "IFF" and c.start > head.end]
        wrapping = bool(thens and thens[0].start > head.end) or (s.find(",", head.end, j) != -1)
        if wrapping:
            if thens and thens[0].start > head.end:
                t = thens[0]
                return BinaryOpNode(
                    "→",
                    parse_range(s, head.end, t.start, cues, registry),
                    parse_range(s, t.end, j, cues, registry),
                )
            comma = s.find(",", head.end, j)
            if comma != -1:
                return BinaryOpNode(
                    "→",
                    parse_range(s, head.end, comma, cues, registry),
                    parse_range(s, comma + 1, j, cues, registry),
                )
        elif not inner_iff:
            para = re.search(r"\bpara que\b", s[head.end:j], flags=re.IGNORECASE)
            if para:
                mid = head.end + para.start()
                return BinaryOpNode(
                    "→",
                    parse_range(s, head.end, mid, cues, registry),
                    parse_range(s, head.end + para.end(), j, cues, registry),
                )
            rest_cues = [
                replace(c, is_boundary=False) if (c.start == head.start and c.end == head.end and c.kind == "IF") else c
                for c in cues
            ]
            return parse_range(s, head.end, j, rest_cues, registry)

    # si y solo si / equivalente
    iffs = [c for c in local if c.kind == "IFF"]
    if iffs:
        c = iffs[0]
        return BinaryOpNode(
            "↔",
            parse_range(s, i, c.start, cues, registry),
            parse_range(s, c.end, j, cues, registry),
        )

    # solo si ANT, CONS  →  CONS → ANT
    only = [c for c in local if c.kind == "ONLY_IF" and s[i:c.start].strip() == ""]
    if only:
        c = only[0]
        rest_i, rest_j = _strip_span(s, c.end, j)
        rest = s[rest_i:rest_j]
        comma = rest.find(",")
        if comma != -1:
            mid = rest_i + comma
            ant = parse_range(s, rest_i, mid, cues, registry)
            cons = parse_range(s, mid + 1, rest_j, cues, registry)
            return BinaryOpNode("→", cons, ant)
        return BinaryOpNode("→", parse_range(s, rest_i, rest_j, cues, registry), _atomic(rest, registry))

    only_mid = [c for c in local if c.kind == "ONLY_IF" and s[i:c.start].strip() != ""]
    if only_mid:
        c = only_mid[0]
        return BinaryOpNode(
            "→",
            parse_range(s, i, c.start, cues, registry),
            parse_range(s, c.end, j, cues, registry),
        )

    # P a menos que Q  →  ¬Q → P
    unless = [c for c in local if c.kind == "UNLESS"]
    if unless:
        c = unless[0]
        if s[i:c.start].strip() == "":
            comma = s.find(",", c.end, j)
            if comma != -1:
                return BinaryOpNode(
                    "→",
                    UnaryOpNode("¬", parse_range(s, c.end, comma, cues, registry)),
                    parse_range(s, comma + 1, j, cues, registry),
                )
            return UnaryOpNode("¬", parse_range(s, c.end, j, cues, registry))
        return BinaryOpNode(
            "→",
            UnaryOpNode("¬", parse_range(s, c.end, j, cues, registry)),
            parse_range(s, i, c.start, cues, registry),
        )

    # P, por lo tanto Q
    mid_then = [c for c in local if c.kind == "THEN" and s[i:c.start].strip()]
    if mid_then:
        c = mid_then[0]
        return BinaryOpNode(
            "→",
            parse_range(s, i, c.start, cues, registry),
            parse_range(s, c.end, j, cues, registry),
        )

    # si ... entonces ...   or   si ..., ...  (non-wrapping leftover)
    ifs = [c for c in local if c.kind == "IF"]
    thens = [c for c in local if c.kind == "THEN"]
    if ifs and s[i:ifs[0].start].strip() == "":
        head = ifs[0]
        if thens and thens[0].start > head.end:
            t = thens[0]
            return BinaryOpNode(
                "→",
                parse_range(s, head.end, t.start, cues, registry),
                parse_range(s, t.end, j, cues, registry),
            )
        comma = s.find(",", head.end, j)
        if comma != -1:
            return BinaryOpNode(
                "→",
                parse_range(s, head.end, comma, cues, registry),
                parse_range(s, comma + 1, j, cues, registry),
            )
        rest_cues = [
            replace(c, is_boundary=False) if (c.start == head.start and c.end == head.end and c.kind == "IF") else c
            for c in cues
        ]
        return parse_range(s, head.end, j, rest_cues, registry)

    # P, de lo contrario Q  →  ¬P → Q
    otherwise = [
        c for c in local
        if c.kind == "IF" and "contrario" in (c.text or "").lower()
    ]
    if otherwise and s[i:otherwise[0].start].strip():
        c = otherwise[0]
        return BinaryOpNode(
            "→",
            UnaryOpNode("¬", parse_range(s, i, c.start, cues, registry)),
            parse_range(s, c.end, j, cues, registry),
        )

    # "X si Y" inverted, except dummy "o si" / "y si" after a coordinator.
    if ifs and s[i:ifs[0].start].strip() != "":
        c = ifs[0]
        glued = [
            x for x in local
            if x.kind in {"OR", "AND"} and x.end <= c.start
            and s[x.end:c.start].strip() == ""
        ]
        if glued:
            rest_cues = [
                replace(x, is_boundary=False)
                if (x.start == c.start and x.end == c.end and x.kind == "IF")
                else x
                for x in cues
            ]
            return parse_range(s, i, j, rest_cues, registry)
        return BinaryOpNode(
            "→",
            parse_range(s, c.end, j, cues, registry),
            parse_range(s, i, c.start, cues, registry),
        )

    # Clause-level "no" (A y no B) wraps the next atom, not the whole rest.
    lead_neg = [
        c for c in local
        if c.kind == "NEG" and (c.start == i or s[i:c.start].strip() == "")
    ]
    if lead_neg:
        c = lead_neg[0]
        nxt = [
            x for x in local
            if x.start >= c.end and x.kind in {"AND", "OR", "IF", "THEN", "IFF", "SINO", "AUNQUE"}
        ]
        if nxt:
            return BinaryOpNode(
                nxt[0].op,
                UnaryOpNode("¬", parse_range(s, c.end, nxt[0].start, cues, registry)),
                parse_range(s, nxt[0].end, j, cues, registry),
            )
        return UnaryOpNode("¬", parse_range(s, c.end, j, cues, registry))

    # Generic binary: highest precedence, but clause-level AND/OR beats NP coordination.
    # Leftover NEG_BLOCK / XOR_TAIL / THEN in the middle must not freeze the span.
    ranked = sorted(
        [c for c in local if c.kind not in {"NEG_BLOCK", "XOR_TAIL", "THEN", "NO_SOLO", "UNLESS"}],
        key=lambda c: (-_eff_prec(s, i, c), c.start),
    )
    for cand in ranked:
        left_txt, right_txt = s[i:cand.start], s[cand.end:j]
        if not left_txt.strip() or not right_txt.strip():
            continue
        if cand.kind in {"AND", "OR"} and is_np_coordination(left_txt, right_txt):
            clause_ops = [
                x for x in local
                if x is not cand
                and x.kind in {"AND", "OR", "IF", "THEN", "IFF"}
                and looks_verbal(s[i:x.start])
                and looks_verbal(s[x.end:j])
            ]
            if clause_ops:
                continue
        if cand.kind in {"AND", "OR", "SINO", "AUNQUE"}:
            return BinaryOpNode(
                cand.op,
                parse_range(s, i, cand.start, cues, registry),
                parse_range(s, cand.end, j, cues, registry),
            )

    return _atomic(s[i:j], registry)


def _atomic(text: str, registry: PropositionRegistry) -> LogicNode:
    text = text.strip(" \t.!?,;:")
    if not text:
        return AtomicNode("∅")
    pid = registry.register(text)
    if pid.startswith("¬"):
        return UnaryOpNode("¬", AtomicNode(pid[1:]))
    return AtomicNode(pid)


def apply_threshold(cues: Sequence[Cue], threshold: float) -> List[Cue]:
    out = []
    for c in cues:
        cc = replace(c)
        cc.is_boundary = bool(c.force or c.prob >= threshold)
        out.append(cc)
    return out


def flip_uncertain(cues: Sequence[Cue], lo: float = 0.35, hi: float = 0.65) -> Optional[List[Cue]]:
    ambiguous = [c for c in cues if (not c.force) and lo <= c.prob <= hi]
    if not ambiguous:
        return None
    target = min(ambiguous, key=lambda c: abs(c.prob - 0.5))
    out = []
    for c in cues:
        cc = replace(c)
        if c.key() == target.key():
            cc.is_boundary = not cc.is_boundary
        out.append(cc)
    return out


def score_tree(cues: Sequence[Cue]) -> float:
    used = [c for c in cues if c.is_boundary]
    if not used:
        return 0.0
    return sum(c.prob if c.is_boundary else (1.0 - c.prob) for c in cues) / len(cues)


def node_to_result(ast: LogicNode, registry: PropositionRegistry) -> dict:
    formula = ast.to_formula()
    props = {k: v for k, v in registry.get_dictionary().items() if k in re.findall(r"[A-Z]", formula) or True}
    # drop unused ∅
    if formula == "∅":
        props = {}
    return {
        "formula": formula,
        "propositions": registry.get_dictionary(),
        "ops": [{"cue": c.text, "op": c.op, "conf": round(c.prob, 3), "kind": c.kind} for c in []],
    }


class PipelineParser:
    """Four-stage parser with a 2–3 candidate beam."""

    def __init__(self, boundary_clf=None, fallback: bool = True):
        self.boundary_clf = boundary_clf
        self.fallback = fallback

    def annotate_cues(self, sentence: str) -> List[Cue]:
        clf = self.boundary_clf
        if clf is not None and hasattr(clf, "predict_cues"):
            try:
                cues = list(clf.predict_cues(sentence) or [])
            except Exception:
                cues = []
        else:
            cues = []
        for cue in cues:
            if clf is not None and hasattr(clf, "predict_boundary_proba"):
                cue.prob = float(clf.predict_boundary_proba(sentence, cue))
                cue.is_boundary = bool(cue.force or cue.prob >= 0.5)
            else:
                cue.prob = 1.0 if cue.force or gold_is_boundary(sentence, cue, None) else 0.2
                cue.is_boundary = cue.prob >= 0.5 or cue.force
        return cues

    def parse_with_cues(self, sentence: str, cues: Sequence[Cue]) -> dict:
        registry = PropositionRegistry()
        s = sentence.strip()
        try:
            ast = parse_range(s, 0, len(s), cues, registry)
            result = node_to_result(ast, registry)
            result["cues"] = cues
            result["clauses"] = list(registry.get_dictionary().values())
            result["ops"] = [
                {"cue": c.text, "op": c.op, "conf": round(c.prob, 3), "kind": c.kind}
                for c in cues if c.is_boundary
            ]
            if result["formula"] in {"∅", "", "¬∅"} and self.fallback:
                return FullRecursiveParser().parse(sentence)
            return result
        except Exception as exc:
            logger.info(f"Pipeline parse fallback: {exc}")
            if self.fallback:
                return FullRecursiveParser().parse(sentence)
            registry = PropositionRegistry()
            return node_to_result(_atomic(sentence, registry), registry)

    def parse(self, sentence: str) -> dict:
        candidates = self.parse_candidates(sentence)
        if candidates:
            return candidates[0]
        registry = PropositionRegistry()
        return node_to_result(_atomic(sentence, registry), registry)

    def parse_candidates(self, sentence: str, max_k: int = 3) -> List[dict]:
        """Stage-2 beam: default / aggressive / conservative / flipped uncertain cue."""
        base = self.annotate_cues(sentence)
        variants: List[List[Cue]] = [
            apply_threshold(base, 0.5),
            apply_threshold(base, 0.35),
            apply_threshold(base, 0.7),
        ]
        flipped = flip_uncertain(apply_threshold(base, 0.5))
        if flipped:
            variants.append(flipped)

        scored: List[dict] = []
        seen = []
        for var in variants:
            result = self.parse_with_cues(sentence, var)
            result = dict(result)
            result["cues"] = var
            result["score"] = score_tree(var)
            formula = result.get("formula", "")
            dup = False
            for prev in seen:
                try:
                    if son_formulas_equivalentes(prev, formula):
                        dup = True
                        break
                except Exception:
                    if prev == formula:
                        dup = True
                        break
            if dup:
                continue
            seen.append(formula)
            scored.append(result)
            if len(scored) >= max_k:
                break
        self._append_and_wider(sentence, apply_threshold(base, 0.5), scored, seen)
        scored.sort(key=lambda r: r.get("score", 0.0), reverse=True)
        if scored:
            return scored
        registry = PropositionRegistry()
        return [node_to_result(_atomic(sentence, registry), registry)]

    def _append_and_wider(self, sentence: str, cues: Sequence[Cue], scored: List[dict], seen: List[str]) -> None:
        """Keep the written precedence as the first tree, and add the other «y»/«o» grouping."""
        bounds = [c for c in cues if c.is_boundary]
        if not any(c.kind == "AND" for c in bounds) or not any(c.kind == "OR" for c in bounds):
            return
        token = _AND_WIDER.set(True)
        try:
            result = self.parse_with_cues(sentence, cues)
        finally:
            _AND_WIDER.reset(token)
        result = dict(result)
        result["cues"] = cues
        result["score"] = score_tree(cues)
        result["scope"] = "y_sobre_o"
        formula = result.get("formula", "")
        for prev in seen:
            try:
                if son_formulas_equivalentes(prev, formula):
                    return
            except Exception:
                if prev == formula:
                    return
        seen.append(formula)
        scored.append(result)


def credit_labels(sentence: str, winner: dict, losers: Iterable[dict]) -> List[dict]:
    """Map a chosen tree back to per-cue 0/1 boundary labels."""
    win_cues: Sequence[Cue] = winner.get("cues") or []
    win_keys = {c.key() for c in win_cues if c.is_boundary}
    loser_used = set()
    for cand in losers:
        for c in cand.get("cues") or []:
            if c.is_boundary:
                loser_used.add(c.key())
    labels = []
    all_cues = win_cues or detect_cues(sentence)
    for c in all_cues:
        if c.key() in win_keys:
            labels.append({"start": c.start, "end": c.end, "kind": c.kind, "text": c.text, "boundary": 1})
        elif c.key() in loser_used:
            labels.append({"start": c.start, "end": c.end, "kind": c.kind, "text": c.text, "boundary": 0})
    return labels
