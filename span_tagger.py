#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Joint token tagger: proposition spans vs connective glue. No regex at inference.

Gold labels come from aligning dataset propositions in the sentence. Leftover
tokens inherit the operator that joins the neighboring variables in the formula.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from config import logger
from pipeline_parser import Cue, KIND_TO_OP, ML_FORCE_KINDS, nms_cues


_STOP = {
    "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del", "al",
    "a", "en", "con", "por", "para", "su", "sus",
}
_GLUE_START = {"si", "y", "e", "o", "u", "entonces", "solo", "ni", "cuando", "siempre", "jamas", "nunca", "tanto"}
_FUNC = {"y", "e", "o", "u", "de", "a", "que", "el", "la"}
_CONTINUE = {
    "IFF", "ONLY_IF", "NEG_BLOCK", "XOR_TAIL", "NO_SOLO", "SINO", "PROP", "AUNQUE",
    "UNLESS",
}
_AND_CONTINUE = {"pesar", "embargo", "obstante", "ademas"}
_IFF_ANCHORS = {"si", "cuando"}
_IFF_FILL = {
    "y", "e", "solo", "solamente", "unicamente", "cuando", "si",
    "necesario", "suficiente", "para", "que", "es",
}
_IF_MARKERS = {
    "si", "cuando", "siempre", "tal", "caso", "condicion", "mientras",
    "cuanto", "vez", "basta", "puesto", "dado", "contrario", "ya",
    "como", "supuesto", "poco", "fin", "debido", "pronto",
}
_IF_FILL = {
    "que", "tal", "con", "siempre", "de", "caso", "una", "en", "lo", "ya",
    "condicion", "supuesto", "poco", "fin", "debido", "pronto", "tan",
    "ser", "el",
}
_UNLESS_FILL = {"a", "menos", "que", "no", "ser", "salvo", "excepto", "si", "sea"}
_THEN_FILL = {
    "por", "lo", "tanto", "consiguiente", "luego", "asi", "que", "modo",
    "manera", "forma", "ende", "consecuencia", "eso", "ello", "pues", "fin",
    "de",
}
_XOR_FILL = {"no", "ambos", "ambas", "las", "dos", "cosas", "vez", "a"}
_NEG_BLOCK_FILL = {
    "es", "posible", "cierto", "verdad", "que", "caso", "ocurre", "falso",
    "mentira", "se", "cumple", "jamas", "nunca",
}
_CLAUSE_PREV = {
    "y", "e", "o", "u", "si", "cuando", "entonces", "que", "sino", "ni",
    "pero", "aunque",
}
# Adjective-internal *no* (no nulo, no inicializada): keep inside the atom.
_LEXICAL_NO_NEXT = {
    "nulo", "nula", "nulos", "nulas",
    "inicializado", "inicializada", "inicializados", "inicializadas",
    "forzado", "forzada", "forzados", "forzadas",
    "resuelto", "resuelta", "resueltos", "resueltas",
}
_GLUE_WORDS = {
    "y", "e", "o", "u", "si", "ni", "aunque", "pero", "sino", "entonces",
    "solo", "cuando", "tampoco", "no",
}
_HARD_GLUE = {
    "y": "B-AND", "e": "B-AND",
    "o": "B-OR", "u": "B-OR",
    "aunque": "B-AUNQUE", "pero": "B-AND",
    "ni": "B-NOR",
    "sino": "B-SINO",
    "entonces": "B-THEN",
    "no": "B-NEG",
}
_PUNCT = re.compile(r"^\W+$")
_FORMULA_TOKS = re.compile(r"↔|→|∧|∨|¬|[A-Z]+|\(|\)")
_WORD = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def _fold(text: str) -> str:
    folded = unicodedata.normalize("NFD", (text or "").lower())
    folded = "".join(c for c in folded if unicodedata.category(c) != "Mn")
    return folded


def _norm_token(text: str) -> str:
    return re.sub(r"[^\wñ]+", "", _fold(text))


def _stem(text: str) -> str:
    w = _norm_token(text)
    for end in ("aron", "eron", "aban", "ían", "ando", "iendo", "ado", "ido",
                "ará", "erá", "irá", "aba", "ía", "ó", "ió", "é", "á"):
        if len(w) > len(end) + 2 and w.endswith(end):
            return w[: -len(end)]
    if len(w) > 4 and w.endswith("e"):
        return w[:-1]
    return w


def _same_word(a: str, b: str) -> bool:
    a, b = _norm_token(a), _norm_token(b)
    if not a or not b:
        return False
    if a == b:
        return True
    sa, sb = _stem(a), _stem(b)
    if sa == sb and len(sa) >= 3:
        return True
    if len(a) >= 4 and len(b) >= 4 and (a.startswith(b[:4]) or b.startswith(a[:4])):
        return True
    return False


def tokenize(sentence: str):
    return list(_WORD.finditer(sentence or ""))


def content_words(text: str) -> List[str]:
    words = re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+", text or "")
    kept = [w for w in words if _norm_token(w) not in _STOP]
    return kept or words


def align_proposition(tokens, prop: str) -> Optional[Tuple[int, int]]:
    """Shortest token window covering most content words of `prop`, in order."""
    needles = content_words(prop)
    if not tokens or not needles:
        return None
    n = len(tokens)
    need = max(1, int(np.ceil(0.6 * len(needles))))
    best = None
    for i in range(n):
        k = 0
        for j in range(i, n):
            if k < len(needles) and _same_word(tokens[j].group(), needles[k]):
                k += 1
            if k >= need and k >= len(needles) * 0.6:
                length = j + 1 - i
                missed = len(needles) - k
                score = (missed, length)
                if best is None or score < best[0]:
                    best = (score, i, j + 1)
                if k == len(needles):
                    break
    if best is None:
        return None
    i, j = best[1], best[2]
    while i > 0 and _norm_token(tokens[i - 1].group()) in _STOP:
        i -= 1
    return i, j


def align_all(sentence: str, props: dict) -> Dict[str, Tuple[int, int]]:
    tokens = tokenize(sentence)
    found = {}
    for var, phrase in (props or {}).items():
        span = align_proposition(tokens, str(phrase))
        if span:
            found[str(var)] = span
    return found


@dataclass
class FNode:
    op: Optional[str] = None
    var: Optional[str] = None
    children: Optional[List["FNode"]] = None

    def vars(self) -> List[str]:
        if self.var:
            return [self.var]
        out = []
        for ch in self.children or []:
            out.extend(ch.vars())
        return out


def parse_formula(formula: str) -> Optional[FNode]:
    toks = _FORMULA_TOKS.findall((formula or "").replace(" ", ""))
    if not toks:
        return None
    pos = [0]

    def peek():
        return toks[pos[0]] if pos[0] < len(toks) else None

    def eat(expected=None):
        cur = peek()
        if expected is not None and cur != expected:
            return None
        if cur is None:
            return None
        pos[0] += 1
        return cur

    def parse_atom():
        t = peek()
        if t == "¬":
            eat()
            return FNode(op="¬", children=[parse_atom()])
        if t == "(":
            eat()
            node = parse_iff()
            eat(")")
            return node
        if t and re.fullmatch(r"[A-Z]+", t):
            eat()
            return FNode(var=t)
        return FNode(var="?")

    def bin_loop(parse_down, ops):
        node = parse_down()
        while peek() in ops:
            op = eat()
            rhs = parse_down()
            node = FNode(op=op, children=[node, rhs])
        return node

    def parse_and():
        return bin_loop(parse_atom, {"∧"})

    def parse_or():
        return bin_loop(parse_and, {"∨"})

    def parse_impl():
        node = parse_or()
        if peek() == "→":
            eat()
            node = FNode(op="→", children=[node, parse_impl()])
        return node

    def parse_iff():
        return bin_loop(parse_impl, {"↔"})

    tree = parse_iff()
    return tree


def _path(node: FNode, var: str, acc=None):
    acc = acc or []
    if node.var == var:
        return acc + [node]
    for ch in node.children or []:
        found = _path(ch, var, acc + [node])
        if found is not None:
            return found
    return None


def lca_op(tree: Optional[FNode], v1: str, v2: str) -> Optional[str]:
    if tree is None:
        return None
    p1, p2 = _path(tree, v1), _path(tree, v2)
    if not p1 or not p2:
        return None
    last = None
    for a, b in zip(p1, p2):
        if a is not b:
            break
        last = a
    if last is None:
        return None
    return last.op


def _under_neg(tree: Optional[FNode], var: str) -> bool:
    path = _path(tree, var) if tree else None
    if not path or len(path) < 2:
        return False
    return any(n.op == "¬" for n in path[:-1])


def _gap_kind(op: Optional[str], tokens, a: int, b: int, tree: Optional[FNode],
              v1: str, v2: str) -> Optional[str]:
    blob = _fold(" ".join(t.group() for t in tokens[a:b]))
    words = [_norm_token(t.group()) for t in tokens[a:b]]
    if op == "↔":
        return "IFF"
    if op == "∨":
        return "OR"
    if op == "→":
        if "entonces" in words or "consiguiente" in words or (
            "tanto" in words and "por" in words
        ) or "luego" in words or "asi" in words or "ende" in words or "consecuencia" in words or (
            "eso" in words and "por" in words
        ) or "modo" in words or "manera" in words or "forma" in words or "fin" in words or "pues" in words:
            return "THEN"
        if "solo" in words and "si" in words:
            return "ONLY_IF"
        if "menos" in words or ("ser" in words and "no" in words) or (
            "sea" in words and "no" in words
        ) or "salvo" in words or "excepto" in words:
            return "UNLESS"
        return "IF"
    if op == "∧":
        if "ni" in words and _under_neg(tree, v1) and _under_neg(tree, v2):
            return "NOR"
        if "sino" in words:
            return "SINO"
        if "aunque" in words or "pesar" in words or "pese" in words or "incluso" in words or (
            "aun" in words and "si" in words
        ) or ("si" in words and "bien" in words) or "embargo" in words or "obstante" in words or (
            "mas" in words and "por" in words
        ):
            return "AUNQUE"
        if "solo" in words and any(w == "no" for w in words):
            return "NO_SOLO"
        return "AND"
    if op == "¬":
        return "NEG_BLOCK"
    return None


def _prefix_kind(tokens, end: int, tree: Optional[FNode], first_var: str) -> Optional[str]:
    words = [_norm_token(t.group()) for t in tokens[:end]]
    blob = " ".join(words)
    if not any(words):
        return None
    if tree and tree.op == "¬":
        return "NEG_BLOCK"
    if "solo" in words and "si" in words:
        return "ONLY_IF"
    if "menos" in words or ("ser" in words and "no" in words) or (
        "sea" in words and "no" in words
    ) or "salvo" in words or "excepto" in words:
        return "UNLESS"
    if "aunque" in words or "pesar" in words or "pese" in words or (
        "aun" in words and "si" in words
    ) or "incluso" in words or ("si" in words and "bien" in words):
        return "AUNQUE"
    if "como" in words and "si" not in words:
        return "IF"
    if "ni" in words:
        return "NOR"
    if blob.startswith("no solo") or " ".join(words[:2]) == "no solo":
        return "NO_SOLO"
    if tree and (tree.op == "→" or any(n.op == "→" for n in (tree.children or []) if n.op)):
        return "IF"
    if "si" in words:
        return "IF"
    return None


def _suffix_kind(tokens, start: int, formula: str) -> Optional[str]:
    if start >= len(tokens):
        return None
    words = [_norm_token(t.group()) for t in tokens[start:]]
    if not words:
        return None
    f = formula or ""
    if "∨" in f and "∧" in f and "¬" in f:
        return "XOR_TAIL"
    if "entonces" in words:
        return "THEN"
    if "sino" in words:
        return "SINO"
    return None


def _write_kind(labs: List[str], i: int, j: int, kind: str) -> None:
    first = True
    for k in range(i, j):
        if labs[k] != "O":
            first = False
            continue
        labs[k] = f"B-{kind}" if first else f"I-{kind}"
        first = False


def _peel_neg(tokens, labs: List[str]) -> None:
    """Mark clause-level 'no' as NEG so it is not swallowed by AND/PROP."""
    for i, tok in enumerate(tokens):
        if _norm_token(tok.group()) != "no":
            continue
        kind = labs[i].split("-", 1)[-1] if "-" in labs[i] else labs[i]
        if kind in {"NEG_BLOCK", "NO_SOLO", "NOR", "IFF", "ONLY_IF", "NEG", "XOR_TAIL"}:
            continue
        nxt = [_norm_token(t.group()) for t in tokens[i:min(i + 3, len(tokens))]]
        if "solo" in nxt[:2]:
            continue
        if i + 1 < len(tokens) and _norm_token(tokens[i + 1].group()) in {"ambos", "ambas", "las", "es", "ocurre", "se", "ser"}:
            continue
        if i + 1 < len(tokens) and _norm_token(tokens[i + 1].group()) in _LEXICAL_NO_NEXT:
            continue
        labs[i] = "B-NEG"
        if i + 1 < len(labs) and labs[i + 1].startswith("I-PROP"):
            labs[i + 1] = "B-PROP"


def gold_labels(sentence: str, item: dict) -> Tuple[list, List[str]]:
    tokens = tokenize(sentence)
    labs = ["O"] * len(tokens)
    if not tokens:
        return tokens, labs
    aligned = align_all(sentence, item.get("propositions") or {})
    for _var, (i, j) in aligned.items():
        labs[i] = "B-PROP"
        for k in range(i + 1, j):
            labs[k] = "I-PROP"
    tree = parse_formula(item.get("formula") or "")
    ordered = sorted(
        ((i, j, var) for var, (i, j) in aligned.items()),
        key=lambda x: x[0],
    )
    for (i1, j1, v1), (i2, j2, v2) in zip(ordered, ordered[1:]):
        if j1 >= i2:
            continue
        op = lca_op(tree, v1, v2)
        kind = _gap_kind(op, tokens, j1, i2, tree, v1, v2)
        if kind:
            _write_kind(labs, j1, i2, kind)
    if ordered:
        first_i, _, first_var = ordered[0]
        kind = _prefix_kind(tokens, first_i, tree, first_var)
        if kind == "NEG_BLOCK":
            cut = 0
            while cut < first_i and _norm_token(tokens[cut].group()) not in _GLUE_START:
                cut += 1
            _write_kind(labs, 0, cut, "NEG_BLOCK")
            rest_words = [_norm_token(t.group()) for t in tokens[cut:first_i]]
            rest_kind = None
            if "solo" in rest_words and "si" in rest_words:
                rest_kind = "ONLY_IF"
            elif "si" in rest_words or "cuando" in rest_words:
                rest_kind = "IF"
            elif "aunque" in rest_words:
                rest_kind = "AUNQUE"
            elif "ni" in rest_words:
                rest_kind = "NOR"
            if rest_kind:
                _write_kind(labs, cut, first_i, rest_kind)
        elif kind:
            _write_kind(labs, 0, first_i, kind)
        last_j = ordered[-1][1]
        kind = _suffix_kind(tokens, last_j, item.get("formula") or "")
        if kind:
            _write_kind(labs, last_j, len(tokens), kind)
    _peel_neg(tokens, labs)
    for i, tok in enumerate(tokens):
        w = _norm_token(tok.group())
        if w in {"jamas", "nunca"} and not labs[i].endswith("NEG_BLOCK"):
            labs[i] = "B-NEG_BLOCK"
        if w == "tanto" and labs[i] in {"O", "I-PROP"}:
            labs[i] = "B-AND"
    for i, tok in enumerate(tokens):
        if _PUNCT.match(tok.group() or "") and labs[i] != "O" and not labs[i].endswith("IFF"):
            if labs[i].startswith("B-") and i + 1 < len(labs) and labs[i + 1].startswith("I-"):
                labs[i + 1] = "B-" + labs[i].split("-", 1)[1]
            labs[i] = "O"
    return tokens, labs


def gold_cues(sentence: str, item: dict) -> List[Cue]:
    tokens, labs = gold_labels(sentence, item)
    return _bio_to_cues(sentence, tokens, labs, [1.0] * len(labs))


def _token_features(tokens, i: int) -> dict:
    n = len(tokens)
    w = _norm_token(tokens[i].group())
    prev = _norm_token(tokens[i - 1].group()) if i else "<s>"
    nxt = _norm_token(tokens[i + 1].group()) if i + 1 < n else "</s>"
    prev2 = _norm_token(tokens[i - 2].group()) if i >= 2 else "<s>"
    nxt2 = _norm_token(tokens[i + 2].group()) if i + 2 < n else "</s>"
    feat = {
        "w": w,
        "p": prev,
        "n": nxt,
        "p2": prev2,
        "n2": nxt2,
        "wp": f"{prev}|{w}",
        "wn": f"{w}|{nxt}",
        "tri": f"{prev}|{w}|{nxt}",
        "suf2": w[-2:],
        "suf3": w[-3:],
        "pre3": w[:3],
        "len": str(min(len(w), 15)),
        "prev_si": str(prev == "si"),
        "next_si": str(nxt == "si"),
        "flank_si": str(prev == "si" and nxt == "si"),
        "si_win": str((prev == "si" or prev2 == "si") and (nxt == "si" or nxt2 == "si")),
        "is_y": str(w in {"y", "e"}),
        "is_o": str(w in {"o", "u"}),
        "is_si": str(w == "si"),
        "is_no": str(w == "no"),
    }
    padded = f" {w} "
    for k in range(2, 5):
        for j in range(len(padded) - k + 1):
            feat[f"c{k}_{padded[j:j + k]}"] = True
    blob = f"{prev} {w} {nxt}"
    for k in range(3, 6):
        for j in range(0, max(len(blob) - k + 1, 0)):
            feat[f"t{k}_{blob[j:j + k]}"] = True
    return feat


def _dropout_sequences(tokens, labs):
    droppable = []
    for i, (tok, lab) in enumerate(zip(tokens, labs)):
        if lab == "O" or not lab.endswith("-IFF"):
            continue
        if _norm_token(tok.group()) in _FUNC:
            droppable.append(i)
    if not droppable:
        return
    keep = [i for i in range(len(tokens)) if i not in set(droppable)]
    if len(keep) >= 2:
        yield [tokens[i] for i in keep], [labs[i] for i in keep]
    for d in droppable:
        keep_one = [i for i in range(len(tokens)) if i != d]
        yield [tokens[i] for i in keep_one], [labs[i] for i in keep_one]


def _mask_interior(tokens, labs):
    idxs = [
        i for i, lab in enumerate(labs)
        if lab.endswith("-IFF") and _norm_token(tokens[i].group()) not in {"si", "y", "e"}
    ]
    if not idxs:
        return

    class _T:
        def __init__(self, match, text):
            self._m = match
            self._t = text

        def group(self, *args):
            return self._t

        def start(self):
            return self._m.start()

        def end(self):
            return self._m.end()

    out = []
    drop = set(idxs)
    for i, tok in enumerate(tokens):
        out.append(_T(tok, "xxxx") if i in drop else tok)
    yield out, list(labs)


def _nec_suf_context(tokens, i: int) -> bool:
    lo, hi = max(0, i - 4), min(len(tokens), i + 5)
    window = [_norm_token(t.group()) for t in tokens[lo:hi]]
    return "necesario" in window and "suficiente" in window


def _tanto_context(tokens, i: int) -> bool:
    for j in range(max(0, i - 14), i):
        if _norm_token(tokens[j].group()) == "tanto":
            return True
    return False


def _iff_context(tokens, i: int) -> bool:
    lo, hi = max(0, i - 3), min(len(tokens), i + 4)
    window = [_norm_token(t.group()) for t in tokens[lo:hi]]
    anchors = sum(1 for w in window if w in _IFF_ANCHORS)
    if "siempre" in window and "cuando" in window:
        return True
    if anchors >= 2:
        return True
    return any(w in _IFF_ANCHORS for w in window) and "solo" in window


def _neg_block_context(tokens, i: int) -> bool:
    nxt = [_norm_token(t.group()) for t in tokens[i:min(len(tokens), i + 5)]]
    if not nxt or nxt[0] != "no":
        return False
    if len(nxt) >= 2 and nxt[1] in {"es", "ocurre"}:
        return True
    if len(nxt) >= 3 and nxt[1] == "se" and nxt[2] in {"cumple", "da", "verifica"}:
        return True
    return False


def _bio_to_cues(sentence: str, tokens, labels: List[str], confs: List[float]) -> List[Cue]:
    cues = []
    i = 0
    n = len(tokens)
    while i < n:
        lab = labels[i]
        if lab in {"O", "B-PROP", "I-PROP"} or "-" not in lab:
            i += 1
            continue
        prefix, kind = lab.split("-", 1)
        w0 = _norm_token(tokens[i].group())
        if prefix == "I" and kind == "IF" and w0 not in _IF_MARKERS and w0 not in _IF_FILL:
            i += 1
            continue
        if prefix == "I" and kind == "THEN" and w0 != "entonces":
            i += 1
            continue
        start = tokens[i].start()
        end = tokens[i].end()
        ps = [confs[i]]
        if prefix == "I" and kind == "XOR_TAIL" and i > 0 and _norm_token(tokens[i - 1].group()) == "no":
            start = tokens[i - 1].start()
        i += 1
        if kind == "AND":
            while i < n:
                raw = tokens[i].group() or ""
                w2 = _norm_token(raw)
                lab2 = labels[i]
                if _PUNCT.match(raw):
                    break
                if w2 == "no":
                    break
                if lab2 == "I-AND" or w2 in _AND_CONTINUE:
                    end = tokens[i].end()
                    ps.append(confs[i])
                    i += 1
                    continue
                break
        elif kind == "OR":
            while i < n:
                raw = tokens[i].group() or ""
                w2 = _norm_token(raw)
                lab2 = labels[i]
                if lab2 == "I-OR" or w2 in {"bien", "sea"}:
                    end = tokens[i].end()
                    ps.append(confs[i])
                    i += 1
                    continue
                break
        elif kind == "IF":
            while i < n:
                raw = tokens[i].group() or ""
                w2 = _norm_token(raw)
                lab2 = labels[i]
                if lab2 in {"I-IF", "B-IF"} or w2 in _IF_FILL:
                    if lab2.startswith("B-") and w2 in _IFF_ANCHORS and tokens[i].start() > end + 2:
                        break
                    end = tokens[i].end()
                    ps.append(confs[i])
                    i += 1
                    continue
                break
        elif kind == "THEN":
            while i < n:
                raw = tokens[i].group() or ""
                w2 = _norm_token(raw)
                lab2 = labels[i]
                if lab2 in {"I-THEN", "B-THEN"} or w2 in _THEN_FILL:
                    end = tokens[i].end()
                    ps.append(confs[i])
                    i += 1
                    continue
                break
        elif kind in _CONTINUE:
            while i < n:
                raw = tokens[i].group() or ""
                w2 = _norm_token(raw)
                lab2 = labels[i]
                if _PUNCT.match(raw):
                    if kind in {"IFF", "ONLY_IF"}:
                        end = tokens[i].end()
                        i += 1
                        continue
                    break
                if kind == "NO_SOLO" and w2 in {"solo", "solamente", "unicamente"}:
                    end = tokens[i].end()
                    ps.append(confs[i])
                    i += 1
                    continue
                if kind == "AUNQUE" and w2 in {"si", "de", "que", "pesar", "pese", "bien"}:
                    end = tokens[i].end()
                    ps.append(confs[i])
                    i += 1
                    continue
                if kind == "UNLESS" and w2 in _UNLESS_FILL:
                    end = tokens[i].end()
                    ps.append(confs[i])
                    i += 1
                    continue
                if kind == "IFF" and w2 in {
                    "necesario", "suficiente", "para", "equivalente", "equivale",
                    "a", "cuando", "y", "si", "siempre",
                }:
                    end = tokens[i].end()
                    ps.append(confs[i])
                    i += 1
                    continue
                if kind == "IFF" and w2 == "no":
                    break
                if kind == "NEG_BLOCK" and w2 in _NEG_BLOCK_FILL:
                    end = tokens[i].end()
                    ps.append(confs[i])
                    i += 1
                    continue
                if lab2 in {f"I-{kind}", f"B-{kind}"}:
                    if lab2.startswith("B-") and tokens[i].start() > end + 2 and kind != "IFF":
                        break
                    end = tokens[i].end()
                    ps.append(confs[i])
                    i += 1
                    continue
                break
        if kind == "NEG" and _norm_token(sentence[start:end]) == "no" and i < n:
            if _norm_token(tokens[i].group()) in {"ambos", "ambas", "las"}:
                kind = "XOR_TAIL"
                while i < n:
                    raw = tokens[i].group() or ""
                    w2 = _norm_token(raw)
                    if _PUNCT.match(raw) or w2 in _XOR_FILL:
                        end = tokens[i].end()
                        ps.append(confs[i])
                        i += 1
                        continue
                    break
        if kind in {"IFF", "IF"}:
            anchor0 = _norm_token(sentence[start:end].split()[0]) if sentence[start:end].split() else ""
            if anchor0 in _IFF_ANCHORS:
                limit = min(n, i + 5)
                for k in range(i, limit):
                    wk = _norm_token(tokens[k].group())
                    if wk not in _IFF_ANCHORS:
                        continue
                    mid_ok = True
                    saw_solo = False
                    for j in range(i, k):
                        wj = _norm_token(tokens[j].group())
                        rawj = tokens[j].group() or ""
                        if wj in {"solo", "solamente", "unicamente"}:
                            saw_solo = True
                        if _PUNCT.match(rawj) or wj in _IFF_FILL:
                            continue
                        if labels[j].endswith("IFF"):
                            continue
                        if labels[j].endswith("PROP") and confs[j] >= 0.7:
                            mid_ok = False
                            break
                    if mid_ok and (kind == "IFF" or saw_solo):
                        end = tokens[k].end()
                        ps.extend(confs[j] for j in range(i, k + 1))
                        i = k + 1
                        kind = "IFF"
                    break
        p = float(np.mean(ps))
        n_tok = max(1, len(re.findall(r"\w+", sentence[start:end])))
        min_p = 0.38 if n_tok >= 2 else 0.48
        if kind in {"AND", "OR", "NEG", "THEN"} and n_tok == 1:
            min_p = 0.35
        if p < min_p:
            continue
        if kind == "AND":
            words = {_norm_token(w) for w in re.findall(r"\w+", sentence[start:end], flags=re.UNICODE)}
            if not words & {"y", "e", "pero", "ademas", "tanto", "como", "cambio"}:
                continue
        if kind == "XOR_TAIL":
            words = {_norm_token(w) for w in re.findall(r"\w+", sentence[start:end], flags=re.UNICODE)}
            if not (words & {"ambos", "ambas"} or ({"las", "dos"} <= words)):
                continue
        if kind == "THEN":
            words = {_norm_token(w) for w in re.findall(r"\w+", sentence[start:end], flags=re.UNICODE)}
            if not words & {
                "entonces", "tanto", "consiguiente", "luego", "asi", "modo",
                "manera", "forma", "ende", "consecuencia", "eso", "ello", "pues", "fin",
            }:
                continue
            if "como" in words:
                continue
        if kind == "NEG_BLOCK":
            words = {_norm_token(w) for w in re.findall(r"\w+", sentence[start:end], flags=re.UNICODE)}
            if not words & {"no", "falso", "mentira", "nunca", "jamas"}:
                continue
        if kind == "UNLESS":
            words = {_norm_token(w) for w in re.findall(r"\w+", sentence[start:end], flags=re.UNICODE)}
            if not (
                {"menos", "que"} <= words
                or {"no", "ser"} <= words
                or {"no", "sea"} <= words
                or "salvo" in words
                or "excepto" in words
            ):
                continue
        if kind == "NO_SOLO":
            words = {_norm_token(w) for w in re.findall(r"\w+", sentence[start:end], flags=re.UNICODE)}
            if "no" not in words or "solo" not in words:
                continue
        if kind == "AUNQUE":
            words = {_norm_token(w) for w in re.findall(r"\w+", sentence[start:end], flags=re.UNICODE)}
            if not (
                words & {"aunque", "aun", "incluso", "pesar", "pese", "embargo", "obstante", "bien"}
                or {"por", "mas"} <= words
                or {"por", "mucho"} <= words
            ):
                continue
        if kind == "IFF":
            words = {_norm_token(w) for w in re.findall(r"\w+", sentence[start:end], flags=re.UNICODE)}
            iff_ok = (
                (("si" in words or "cuando" in words) and words & {"solo", "solamente", "unicamente", "siempre"})
                or ("siempre" in words and "cuando" in words)
                or ({"necesario", "suficiente"} <= words)
                or bool(words & {"equivale", "equivalente"})
            )
            if not iff_ok:
                continue
        if kind == "IF":
            words = {_norm_token(w) for w in re.findall(r"\w+", sentence[start:end], flags=re.UNICODE)}
            if not (
                words & _IF_MARKERS
                or {"con", "que"} <= words
                or {"de", "ser"} <= words
                or {"de", "haberlo"} <= words
            ):
                continue
            if words <= {"caso"}:
                continue
        op = KIND_TO_OP.get(kind, "∧")
        text = sentence[start:end]
        force = kind in ML_FORCE_KINDS
        cues.append(Cue(start, end, kind, op, text, force, prob=p, is_boundary=True))
    return nms_cues(cues)


class JointSpanTagger:
    """One label per token: PROP vs connective kind. Inference is ML only."""

    def __init__(self):
        self.vec: Optional[DictVectorizer] = None
        self.clf: Optional[LogisticRegression] = None
        self._ready = False

    def train_from_dataset(self, items: List[dict]) -> None:
        feats: List[dict] = []
        labels: List[str] = []
        aligned_items = 0
        for item in items:
            sentence = item.get("sentence") or ""
            tokens, labs = gold_labels(sentence, item)
            if not tokens:
                continue
            props = item.get("propositions") or {}
            if len(align_all(sentence, props)) == len(props):
                aligned_items += 1
            seqs = [(tokens, labs)]
            for extra in _dropout_sequences(tokens, labs):
                seqs.append(extra)
            for extra in _mask_interior(tokens, labs):
                seqs.append(extra)
            for toks, lbs in seqs:
                for i, lab in enumerate(lbs):
                    feats.append(_token_features(toks, i))
                    labels.append(lab)
        if len(set(labels)) < 2:
            logger.info("Joint-tagger: clases insuficientes.")
            return
        self.vec = DictVectorizer(sparse=True)
        X = self.vec.fit_transform(feats)
        self.clf = LogisticRegression(
            max_iter=400, C=0.4, class_weight="balanced", solver="saga", random_state=42,
        )
        self.clf.fit(X, labels)
        self._ready = True
        pred = self.clf.predict(X)
        logger.info(
            f"Joint-tagger: n={len(labels)} acc={accuracy_score(labels, pred):.3f} "
            f"items_alineados={aligned_items}/{len(items)} clases={sorted(set(labels))}"
        )

    def predict_labels(self, sentence: str) -> Tuple[list, List[str], List[float]]:
        tokens = tokenize(sentence)
        if not tokens or not self._ready:
            return tokens, [], []
        X = self.vec.transform([_token_features(tokens, i) for i in range(len(tokens))])
        proba = self.clf.predict_proba(X)
        classes = list(self.clf.classes_)
        labels, confs = [], []
        for i, dist in enumerate(proba):
            top = int(np.argmax(dist))
            lab = classes[top]
            p = float(dist[top])
            w = _norm_token(tokens[i].group())
            nxt = _norm_token(tokens[i + 1].group()) if i + 1 < len(tokens) else ""
            prev = _norm_token(tokens[i - 1].group()) if i else ""
            prev2 = _norm_token(tokens[i - 2].group()) if i >= 2 else ""
            prev_raw = (tokens[i - 1].group() if i else "") or ""
            if w == "no" and nxt in {"ambos", "ambas", "las"}:
                lab = "B-XOR_TAIL"
                p = max(p, 0.75)
            elif w == "no" and nxt == "solo":
                lab = "B-NO_SOLO"
                p = max(p, 0.8)
            elif w == "no" and nxt == "sea":
                lab = "B-UNLESS"
                p = max(p, 0.8)
            elif w == "no" and (_neg_block_context(tokens, i) or nxt in {"ser", "obstante"}):
                pass
            elif w == "no":
                if nxt in _LEXICAL_NO_NEXT:
                    lab = "I-PROP" if labels and labels[-1].endswith("PROP") else "B-PROP"
                else:
                    clause = i == 0 or prev in _CLAUSE_PREV or bool(_PUNCT.match(prev_raw))
                    verbish = nxt in {"se", "hay", "esta", "estan", "fue", "era", "son"}
                    if clause or verbish:
                        lab = "B-NEG"
                        p = max(p, 0.6)
                    elif lab.endswith("NEG"):
                        lab = "I-PROP" if labels and labels[-1].endswith("PROP") else "B-PROP"
                    elif lab.endswith(("AND", "OR", "AUNQUE", "THEN", "IF", "IFF", "SINO", "NOR")):
                        lab = "I-PROP" if labels and labels[-1].endswith("PROP") else "B-PROP"
            elif w in {"jamas", "nunca"}:
                lab = "B-NEG_BLOCK"
                p = max(p, 0.75)
            elif w == "como" and _tanto_context(tokens, i):
                if nxt == "si" or prev == "si":
                    lab = "B-OR"
                    p = max(p, 0.75)
                else:
                    lab = "B-AND"
                    p = max(p, 0.7)
            elif w == "como" and nxt != "si" and (i == 0 or bool(_PUNCT.match(prev_raw))):
                lab = "B-IF"
                p = max(p, 0.75)
            elif w == "si" and nxt == "bien":
                lab = "B-AUNQUE"
                p = max(p, 0.8)
            elif w == "por" and nxt == "si":
                lab = "B-IF"
                p = max(p, 0.8)
            elif w == "siempre" and nxt == "que":
                lab = "B-IF"
                p = max(p, 0.75)
            elif w == "siempre" and nxt in {"y", "cuando"}:
                lab = "B-IFF"
                p = max(p, 0.8)
            elif w == "cuando" and (
                (prev == "y" and prev2 == "siempre") or prev == "siempre"
            ):
                lab = "I-IFF" if labels and labels[-1].endswith("IFF") else "B-IFF"
                p = max(p, 0.8)
            elif w == "a" and nxt == "poco":
                lab = "B-IF"
                p = max(p, 0.8)
            elif w in {"salvo", "excepto"}:
                lab = "B-UNLESS"
                p = max(p, 0.8)
            elif w == "caso" and prev == "en":
                lab = "B-IF"
                p = max(p, 0.75)
            elif w in {"condicion", "mientras", "basta", "puesto"}:
                lab = "B-IF"
                p = max(p, 0.75)
            elif w == "cuanto" and prev == "en":
                lab = "B-IF"
                p = max(p, 0.75)
            elif w == "vez" and prev == "una":
                lab = "B-IF"
                p = max(p, 0.75)
            elif w == "dado" and nxt == "que":
                lab = "B-IF"
                p = max(p, 0.75)
            elif w == "ya" and nxt == "que":
                lab = "B-IF"
                p = max(p, 0.75)
            elif w == "ya" and nxt == "sea":
                lab = "B-IF"
                p = max(p, 0.75)
            elif w == "contrario" and prev == "lo" and prev2 == "de":
                lab = "B-IF"
                p = max(p, 0.8)
            elif w == "contrario":
                lab = "I-PROP" if labels and labels[-1].endswith("PROP") else "B-PROP"
            elif w in {"aun", "incluso"} and nxt in {"si", "asi"}:
                lab = "B-AUNQUE"
                p = max(p, 0.75)
            elif w == "pese":
                lab = "B-AUNQUE"
                p = max(p, 0.8)
            elif w == "mas" and prev == "por" and nxt == "que":
                lab = "B-AUNQUE"
                p = max(p, 0.75)
            elif w == "mucho" and prev == "por" and nxt == "que":
                lab = "B-AUNQUE"
                p = max(p, 0.75)
            elif w == "ahora" and nxt == "bien":
                lab = "B-AUNQUE"
                p = max(p, 0.7)
            elif w == "por" and nxt == "lo" and i + 2 < len(tokens) and _norm_token(tokens[i + 2].group()) == "tanto":
                lab = "B-THEN"
                p = max(p, 0.8)
            elif w == "consiguiente":
                lab = "B-THEN"
                p = max(p, 0.75)
            elif w == "asi" and nxt in {"que", "pues"}:
                lab = "B-THEN"
                p = max(p, 0.8)
            elif w == "ende":
                lab = "B-THEN"
                p = max(p, 0.8)
            elif w == "consecuencia" and prev == "en":
                lab = "B-THEN"
                p = max(p, 0.8)
            elif w == "eso" and prev == "por":
                lab = "B-THEN"
                p = max(p, 0.75)
            elif w == "de" and nxt in {"modo", "manera", "forma"}:
                lab = "B-THEN"
                p = max(p, 0.8)
            elif w == "fin" and prev == "a":
                lab = "B-THEN"
                p = max(p, 0.8)
            elif w == "conque":
                lab = "B-THEN"
                p = max(p, 0.8)
            elif w in {"equivale", "equivalente"}:
                lab = "B-IFF"
                p = max(p, 0.8)
            elif w == "menos" and prev == "a" and nxt == "que":
                lab = "B-UNLESS"
                p = max(p, 0.8)
            elif w == "a" and nxt == "menos":
                lab = "B-UNLESS"
                p = max(p, 0.8)
            elif w == "a" and nxt == "no" and (i + 2 < len(tokens) and _norm_token(tokens[i + 2].group()) == "ser"):
                lab = "B-UNLESS"
                p = max(p, 0.8)
            elif w == "supuesto":
                lab = "B-IF"
                p = max(p, 0.8)
            elif w == "en" and nxt == "el" and i + 2 < len(tokens) and _norm_token(tokens[i + 2].group()) == "supuesto":
                lab = "B-IF"
                p = max(p, 0.8)
            elif w == "poco" and prev == "a" and nxt == "que":
                lab = "B-IF"
                p = max(p, 0.8)
            elif w == "debido":
                lab = "B-IF"
                p = max(p, 0.75)
            elif w == "pronto" and prev == "tan":
                lab = "B-IF"
                p = max(p, 0.75)
            elif w == "de" and nxt == "ser" and (i == 0 or bool(_PUNCT.match(prev_raw))):
                lab = "B-IF"
                p = max(p, 0.8)
            elif w == "de" and nxt == "haberlo" and (i == 0 or bool(_PUNCT.match(prev_raw))):
                lab = "B-IF"
                p = max(p, 0.8)
            elif w == "cambio" and prev == "en":
                lab = "B-AND"
                p = max(p, 0.7)
            elif w in {"con", "tal"} and (nxt == "tal" or (w == "tal" and nxt == "que") or (w == "con" and nxt == "que")):
                lab = "B-IF"
                p = max(p, 0.7)
            elif w == "si" and not _iff_context(tokens, i) and nxt not in {"solo", "solamente", "unicamente", "bien"}:
                if not lab.endswith("IFF") and not lab.endswith("ONLY_IF"):
                    lab = "B-IF"
                    p = max(p, 0.7)
            elif w in _HARD_GLUE and not (
                w in {"y", "e"} and (_iff_context(tokens, i) or _nec_suf_context(tokens, i))
            ) and not (w == "no" and nxt in _LEXICAL_NO_NEXT):
                forced = _HARD_GLUE[w]
                if not lab.endswith(forced.split("-", 1)[-1]):
                    lab = forced
                idx = classes.index(lab) if lab in classes else -1
                p = max(float(dist[idx]) if idx >= 0 else p, 0.62)
            labels.append(lab)
            confs.append(p)
        return tokens, labels, confs

    def predict_cues(self, sentence: str) -> List[Cue]:
        tokens, labels, confs = self.predict_labels(sentence)
        if not labels:
            return []
        return _bio_to_cues(sentence, tokens, labels, confs)

    def to_dict(self) -> dict:
        return {"vec": self.vec, "clf": self.clf, "ready": self._ready}

    @classmethod
    def from_dict(cls, data: dict) -> "JointSpanTagger":
        obj = cls()
        obj.vec = data.get("vec")
        obj.clf = data.get("clf")
        obj._ready = bool(data.get("ready") and obj.clf is not None and obj.vec is not None)
        return obj
