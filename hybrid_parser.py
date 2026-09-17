#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hybrid logic parsers for Spanish-to-propositional-logic translation.

Provides:
  - FullRecursiveParser: rule-based recursive descent (full pattern coverage)
  - GuidedParser: ML-guided binary split using the predicted main connective
  - HybridLogicParser: orchestrator that dispatches based on ML confidence
  - AST node classes (LogicNode, AtomicNode, UnaryOpNode, BinaryOpNode)
  - PropositionRegistry: maps cleaned text to propositional variables
"""

import re
import string
from typing import Dict, Optional, Tuple
from config import logger

# Adjective-internal *no* stays inside the atom (no nulo, no inicializada, …).
_LEXICAL_NO_NEXT = re.compile(
    r"^(nulo|nula|nulos|nulas|inicializad[oa]s?|forzad[oa]s?|resuelt[oa]s?)\b",
    re.IGNORECASE,
)

try:
    from config import NLP_SPANISH
except Exception:
    NLP_SPANISH = None

# ── Spanish Verb Suffixes & Subject Pronouns (clause-splitting helpers) ──
VERB_ENDINGS = [
    "ar", "er", "ir", "ando", "iendo", "ado", "ido",
    "aba", "abas", "ábamos", "abais", "aban",
    "ía", "ías", "íamos", "íais", "ían",
    "é", "aste", "ó", "amos", "asteis", "aron",
    "í", "iste", "ió", "imos", "isteis", "ieron",
    "aré", "eré", "iré", "arás", "erás", "irás",
    "ará", "erá", "irá", "aremos", "eremos", "iremos",
    "aréis", "eréis", "iréis", "arán", "erán", "irán"
]
SUBJECT_PRONOUNS = {"yo", "tú", "él", "ella", "usted", "nosotros", "vosotros", "ellos", "ellas", "ustedes"}

# ── Clause-Splitting Utilities ──

def split_clauses_regex(text: str) -> Optional[Tuple[str, str]]:
    """Split a sentence into two clauses at the first verb-subject/verb-verb boundary (regex fallback)."""
    words = text.split()
    if len(words) < 4:
        return None

    verb1_idx = None
    for i in range(1, len(words)):
        w = words[i].lower().rstrip(".,;")
        if len(w) > 3 and any(w.endswith(e) for e in VERB_ENDINGS):
            verb1_idx = i
            break
    if verb1_idx is None:
        return None

    for j in range(verb1_idx + 1, len(words)):
        w = words[j].lower().rstrip(".,;")
        is_pronoun = w in SUBJECT_PRONOUNS
        is_verb = len(w) > 3 and any(w.endswith(e) for e in VERB_ENDINGS)
        if is_pronoun or is_verb:
            antecedent = " ".join(words[1:j])
            consequent = " ".join(words[j:])
            if antecedent and consequent:
                return antecedent, consequent
            break
    return None

class SpacyClauseSplitter:
    """Split conditional clauses using spaCy dependency parsing, with regex fallback."""
    def __init__(self):
        self.nlp = NLP_SPANISH

    def split(self, text: str) -> Optional[Tuple[str, str]]:
        t = text.strip()
        if not t.lower().startswith("si "):
            return None
        if "," in t or " entonces " in t or "si y solo si" in t:
            return None
        if self.nlp is not None:
            result = self._split_spacy(t)
            if result:
                return result
        return self._split_regex(t)

    def _split_spacy(self, text: str) -> Optional[Tuple[str, str]]:
        doc = self.nlp(text)
        si_tok = None
        for tok in doc:
            if tok.text.lower() == "si" and tok.i == 0:
                si_tok = tok
                break
        if si_tok is None: return None
        sub_verb = si_tok.head
        if sub_verb == si_tok: return None
        sub_indices = {t.i for t in sub_verb.subtree}
        sub_end = max(sub_indices)
        if sub_end + 1 >= len(doc): return None
        antecedent = doc[1:sub_end + 1].text.strip()
        consequent = doc[sub_end + 1:].text.strip()
        if antecedent and consequent:
            return antecedent, consequent
        return None

    def _split_regex(self, text: str) -> Optional[Tuple[str, str]]:
        words = text.split()
        if len(words) < 4: return None
        verb1_idx = None
        for i in range(1, len(words)):
            w = words[i].lower().rstrip(".,;")
            if len(w) > 3 and any(w.endswith(e) for e in VERB_ENDINGS):
                verb1_idx = i
                break
        if verb1_idx is None: return None
        for j in range(verb1_idx + 1, len(words)):
            w = words[j].lower().rstrip(".,;")
            is_pronoun = w in SUBJECT_PRONOUNS
            is_verb = len(w) > 3 and any(w.endswith(e) for e in VERB_ENDINGS)
            if is_pronoun or is_verb:
                antecedent = " ".join(words[1:j])
                consequent = " ".join(words[j:])
                if antecedent and consequent:
                    return antecedent, consequent
                break
        return None

_CLAUSE_SPLITTER = SpacyClauseSplitter()

# ── AST Node Classes ──

class LogicNode:
    def to_formula(self) -> str:
        raise NotImplementedError

class AtomicNode(LogicNode):
    """Leaf node representing a single propositional variable."""
    def __init__(self, p_id: str):
        self.p_id = p_id
    def to_formula(self) -> str:
        return self.p_id

class UnaryOpNode(LogicNode):
    """Unary operator node (e.g., negation ¬)."""
    def __init__(self, op: str, child: LogicNode):
        self.op = op
        self.child = child
    def to_formula(self) -> str:
        child_formula = self.child.to_formula()
        if isinstance(self.child, AtomicNode):
            return f"{self.op}{child_formula}"
        return f"{self.op}({child_formula})"

class BinaryOpNode(LogicNode):
    """Binary operator node (∧, ∨, →, ↔)."""
    def __init__(self, op: str, left: LogicNode, right: LogicNode):
        self.op = op
        self.left = left
        self.right = right
    def to_formula(self) -> str:
        return f"({self.left.to_formula()} {self.op} {self.right.to_formula()})"

class PropositionRegistry:
    """Maps cleaned proposition text to propositional variables (A–Z, then Pn).
    
    Handles negation extraction during registration: if the text starts with
    a negation phrase (e.g., "no es cierto que"), the negated prefix is stripped
    and a ¬ prefix is returned with the variable.
    """
    def __init__(self):
        self._propositions: Dict[str, str] = {}
        self._text_to_id: Dict[str, str] = {}
        self._counter = 0

    def _normalize(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r'[.!?,\;]+$', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def register(self, raw_text: str) -> str:
        clean_text = self._normalize(raw_text)
        is_negated = False
        neg_pattern = r'^(no es cierto que|no es verdad que|es falso que|es mentira que|no se cumple que|no)\s+(.+)'
        match = re.match(neg_pattern, clean_text, re.IGNORECASE)
        if match:
            prefix, rest = match.group(1), match.group(2).strip()
            if not (prefix.lower() == "no" and _LEXICAL_NO_NEXT.match(rest)):
                clean_text = rest
                is_negated = True
        if not is_negated:
            mid = re.search(r'\bno\s+', clean_text, re.IGNORECASE)
            if mid and not _LEXICAL_NO_NEXT.match(clean_text[mid.end():].lstrip()):
                clean_text = (clean_text[:mid.start()] + clean_text[mid.end():]).strip()
                clean_text = re.sub(r'\s+', ' ', clean_text)
                is_negated = True
        if not clean_text:
            return "∅"
        if clean_text in self._text_to_id:
            p_id = self._text_to_id[clean_text]
        else:
            if self._counter < 26:
                p_id = string.ascii_uppercase[self._counter]
            else:
                p_id = f"P{self._counter - 25}"
            self._propositions[p_id] = clean_text
            self._text_to_id[clean_text] = p_id
            self._counter += 1
        return f"¬{p_id}" if is_negated else p_id

    def get_dictionary(self) -> Dict[str, str]:
        return self._propositions

class FullRecursiveParser:
    """Recursive-descent parser covering all connective patterns.
    
    Precedence (highest first):
      block_negation > biconditional > conditional > nor > disjunction > conjunction > negation > atomic
    """
    def __init__(self):
        self.registry = PropositionRegistry()

    @staticmethod
    def _normalize(text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r'[.!?,\;:]+$', '', text)
        text = re.sub(r'^(y\s+además|y\s+solamente|y|o|u|e|entonces|luego|además|pero|o\s+bien)\b\s+', '', text)
        text = re.sub(r'\s+\b(y|o|u|e|entonces|luego|además|pero)$', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _split_si_comma(self, text: str) -> Optional[Tuple[str, str]]:
        match = re.search(r'^si\s+(.+?),\s+([^,]+)$', text, re.IGNORECASE)
        if match: return match.group(1), match.group(2)
        match = re.search(r'^si\s+(.+?),\s*(.+)', text, re.IGNORECASE | re.DOTALL)
        if match: return match.group(1), match.group(2)
        return None

    def _smart_si_split(self, text: str) -> Optional[Tuple[str, str]]:
        text_lower = text.lower().strip()
        if not text_lower.startswith("si "):
            return None
        
        content = text[3:].strip()
        words = content.split()
        
        indicators = {'el', 'la', 'los', 'las', 'un', 'una', 'este', 'esta', 'estos', 'estas', 'su', 'sus'}
        
        def has_verb(s: str) -> bool:
            s_words = s.lower().split()
            common_verbs = {'es', 'son', 'fue', 'fuesen', 'sea', 'sean', 'apaga', 'detiene', 'falla', 'termina', 'continúe', 'naveguen', 'compilas', 'consuma'}
            for w in s_words:
                w_clean = re.sub(r'[^\w]', '', w)
                if w_clean in common_verbs:
                    return True
                if len(w_clean) > 2 and any(w_clean.endswith(end) for end in ('ar', 'er', 'ir', 'amos', 'áis', 'emos', 'éis', 'imos', 'ís', 'aba', 'abas', 'an', 'ía', 'ías', 'ían', 'as', 'es', 'a', 'e', 'ó', 'ió', 'ará', 'erá', 'irá', 'arán', 'erán', 'irán', 'ando', 'iendo', 'ado', 'ido')):
                    if w_clean not in {'para', 'que', 'en', 'un', 'una', 'el', 'la', 'los', 'las', 'y', 'o', 'si', 'no', 'modo', 'debug', 'router', 'señal', 'memoria', 'binario', 'falso', 'cierto', 'verdad', 'caso'}:
                        return True
            return False

        for i in range(1, len(words)):
            if words[i].lower() in indicators:
                left = " ".join(words[:i])
                right = " ".join(words[i:])
                if has_verb(left) and has_verb(right):
                    return left, right

        for i in range(1, len(words)):
            left = " ".join(words[:i])
            right = " ".join(words[i:])
            if has_verb(left) and has_verb(right):
                return left, right
                
        return None

    def _build_binary(self, op: str, left_text: str, right_text: str, reverse_ast: bool = False) -> LogicNode:
        node_first = self._parse_recursive(left_text)
        node_second = self._parse_recursive(right_text)
        return BinaryOpNode(op, node_second, node_first) if reverse_ast else BinaryOpNode(op, node_first, node_second)

    # ── Block negation: "no es cierto que...", "es falso que..." ──
    def _parse_block_negation(self, text: str) -> Optional[LogicNode]:
        pattern = r"^(no es cierto que|no es verdad que|no ocurre que|no es el caso que|no se cumple que|es falso que|es mentira que)\s+(.+)"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return UnaryOpNode('¬', self._parse_recursive(match.group(2)))
        m2 = re.match(r'^(es falso|es mentira|es incorrecto|es inexacto)\s+(?!que\b)(.+)', text, re.IGNORECASE)
        if m2:
            return UnaryOpNode('¬', self._parse_recursive(m2.group(2)))
        return None

    # ── Biconditional: "si y solo si", "es equivalente a", "equivale a" ──
    def _parse_biconditional(self, text: str) -> Optional[LogicNode]:
        m = re.search(r'(.+?)\s+si,?\s+y\s+solo\s+si,?\s+(.+)', text, re.IGNORECASE)
        if m: return self._build_binary('↔', m.group(1), m.group(2))
        m = re.search(r'^si\s+(.+?)\s+entonces\s+(.+?)\s+y\s+viceversa$', text, re.IGNORECASE)
        if m: return self._build_binary('↔', m.group(1), m.group(2))
        m = re.search(r'(.+?)\s+es\s+equivalente\s+a\s+(.+)', text, re.IGNORECASE)
        if m: return self._build_binary('↔', m.group(1), m.group(2))
        m = re.search(r'(.+?)\s+equivale\s+a\s+(.+)', text, re.IGNORECASE)
        if m: return self._build_binary('↔', m.group(1), m.group(2))
        m = re.search(r'(.+?)\s+es\s+necesari[oa]s?\s+y\s+suficiente\s+para\s+(.+)', text, re.IGNORECASE)
        if m: return self._build_binary('↔', m.group(1), m.group(2))
        return None

    # ── Conditional/implication: si...entonces, solo si, cuando, necesario/suficiente ──
    def _parse_conditional(self, text: str) -> Optional[LogicNode]:
        m = re.search(r'(.+?)\s+o\s+si\s+(.+)', text, re.IGNORECASE)
        if m:
            return self._build_binary('∨', m.group(1), f"si {m.group(2)}")

        m = re.match(r'^si\s+(.+?)\s+entonces\s+(.+)', text, re.IGNORECASE | re.DOTALL)
        if m: return self._build_binary('→', m.group(1), m.group(2))

        m = re.match(r'^solo\s+si\s+(.+?)(?:,\s*|\s+)(.+)', text, re.IGNORECASE | re.DOTALL)
        if m:
            antecedente_complejo = m.group(1).strip()
            consecuente = m.group(2).strip()
            return self._build_binary('→', consecuente, antecedente_complejo)

        if text.lower().startswith('si ') and ',' in text:
            partes = self._split_si_comma(text)
            if partes: return self._build_binary('→', partes[0], partes[1])

        if text.lower().startswith('si '):
            partes = self._smart_si_split(text)
            if partes: return self._build_binary('→', partes[0], partes[1])

        m = re.match(r'^siempre\s+que\s+(.+?),\s*(.+)', text, re.IGNORECASE | re.DOTALL)
        if m: return self._build_binary('→', m.group(1), m.group(2))
        m = re.match(r'^siempre\s+que\s+(.+?)\s+entonces\s+(.+)', text, re.IGNORECASE | re.DOTALL)
        if m: return self._build_binary('→', m.group(1), m.group(2))
        m = re.match(r'^siempre\s+que\s+(.+)', text, re.IGNORECASE | re.DOTALL)
        if m:
            rest = m.group(1).strip()
            partes = split_clauses_regex(rest)
            if partes:
                return self._build_binary('→', partes[0], partes[1])

        m = re.match(r'^cuando\s+(.+?),\s*(.+)', text, re.IGNORECASE | re.DOTALL)
        if m: return self._build_binary('→', m.group(1), m.group(2))
        m = re.match(r'^cuando\s+(.+)', text, re.IGNORECASE | re.DOTALL)
        if m:
            rest = m.group(1).strip()
            partes = split_clauses_regex(rest)
            if partes:
                return self._build_binary('→', partes[0], partes[1])

        m = re.search(r'(.+?)\s+\bcuando\b\s+(.+)', text, re.IGNORECASE)
        if m and not (', ' in text and text.find(',') < m.start(2)):
            return self._build_binary('→', m.group(1), m.group(2), reverse_ast=True)

        m = re.match(r'^es\s+necesario\s+(.+?)\s+para\s+(.+)$', text, re.IGNORECASE)
        if m: return self._build_binary('→', m.group(2), m.group(1))

        m = re.match(r'^una\s+condici[oó]n\s+necesaria\s+para\s+(.+?)\s+es\s+(.+)$', text, re.IGNORECASE)
        if m: return self._build_binary('→', m.group(1), m.group(2))

        m = re.match(r'^una\s+condici[oó]n\s+suficiente\s+para\s+(.+?)\s+es\s+(.+)$', text, re.IGNORECASE)
        if m: return self._build_binary('→', m.group(2), m.group(1))

        m = re.match(r'^(.+?)\s+es\s+suficiente\s+para\s+(.+)$', text, re.IGNORECASE)
        if m: return self._build_binary('→', m.group(1), m.group(2))

        m = re.match(r'^(.+?)\s+es\s+necesari[oa]\s+para\s+(.+)$', text, re.IGNORECASE)
        if m: 
            return self._build_binary('→', m.group(2), m.group(1))

        m = re.match(r'^basta\s+(.+?)\s+para\s+(.+)$', text, re.IGNORECASE)
        if m: return self._build_binary('→', m.group(1), m.group(2))

        m = re.search(r'(.+?)\s+\bsolo\s+si\b\s+(.+)', text, re.IGNORECASE)
        if m and not re.search(r'si,?\s+y\s+solo\s+si', text, re.IGNORECASE):
            return self._build_binary('→', m.group(1), m.group(2))

        m = re.search(r'(.+?)\s+\bsi\b\s+(.+)', text, re.IGNORECASE)
        if m and not text.lower().startswith('si ') and not re.search(r'si,?\s+y\s+solo\s+si', text, re.IGNORECASE):
            return self._build_binary('→', m.group(1), m.group(2), reverse_ast=True)

        m = re.search(r'(.+?)\s+\ba\s+menos\s+que\b\s+(.+)', text, re.IGNORECASE)
        if m:
            n1 = self._parse_recursive(m.group(1))
            n2 = self._parse_recursive(m.group(2))
            return BinaryOpNode('→', UnaryOpNode('¬', n2), n1)

        return None

    # ── Disjunction: "o", "o bien", "u" ──
    def _parse_disjunction(self, text: str) -> Optional[LogicNode]:
        m = re.match(r'^\s*(?:o\s+bien|o|u)\s+(.+?)\s+(?:o\s+bien|o|u)\s+(.+?)(?:\s+(?:o|u)\s+ambas?\s*(?:cosas?)?)?\s*$', text, re.IGNORECASE)
        if m:
            left = self._parse_recursive(m.group(1))
            right = self._parse_recursive(m.group(2))
            return BinaryOpNode('∨', left, right)

        _t = re.sub(r'\s+o\s+ambos$', '', text, flags=re.IGNORECASE)
        _t = re.sub(r'^(al\s+menos|como\s+m[ií]nimo)\s+', '', _t, flags=re.IGNORECASE)
        _t = re.sub(r'\bo\s+bien\b', ' o ', _t, flags=re.IGNORECASE)
        m = re.search(r'(.+?)\s+\bo\b\s+(.+)', _t, re.IGNORECASE)
        if m: return self._build_binary('∨', m.group(1), m.group(2))
        return None

    # ── Conjunction: "y", "e", "pero", "sin embargo", commas ──
    def _parse_conjunction(self, text: str) -> Optional[LogicNode]:
        pattern = r'(.+?)\s+\b(sin\s+embargo|no\s+obstante|a\s+pesar\s+de(?:\s+que)?|adem[aá]s\s+de|en\s+adici[oó]n\s+a|aunque)\b\s+(.+)'
        m = re.match(r'^(si\s+.+?),\s*(y|e|pero)\s+(si\s+.+)$', text, re.IGNORECASE)
        if m:
            return self._build_binary('∧', m.group(1), m.group(3))
        m = re.search(pattern, text, re.IGNORECASE)
        if m: return self._build_binary('∧', m.group(1), m.group(3))
        if ',' in text:
            parts = text.split(',', 1)
            left, right = parts[0].strip(), parts[1].strip()
            if right: return self._build_binary('∧', left, right)
        m = re.search(r'(.+?)\s+\b(y|e|pero)\b\s+(.+)', text, re.IGNORECASE)
        if m: return self._build_binary('∧', m.group(1), m.group(3))
        
        return None

    # ── Neither-nor: "ni...ni", "no...ni" ──
    def _parse_nor(self, text: str) -> Optional[LogicNode]:
        _m = re.match(r'^ni\s+(.+?)\s+ni\s+(.+)$', text, re.IGNORECASE)
        if _m:
            n1 = self._parse_recursive(_m.group(1))
            n2 = self._parse_recursive(_m.group(2))
            return BinaryOpNode('∧', UnaryOpNode('¬', n1), UnaryOpNode('¬', n2))
        _m = re.match(r'^no\s+(.+?)\s+ni\s+(.+)$', text, re.IGNORECASE)
        if _m:
            n1 = self._parse_recursive(_m.group(1))
            n2 = self._parse_recursive(_m.group(2))
            return BinaryOpNode('∧', UnaryOpNode('¬', n1), UnaryOpNode('¬', n2))
        return None

    # ── Simple negation: "no ..." ──
    def _parse_negation(self, text: str) -> Optional[LogicNode]:
        m = re.match(r'^no\s+es\s+(?:falso|mentira|incorrecto|inexacto)\s+que\s+(.+)', text, re.IGNORECASE)
        if m: return self._parse_recursive(m.group(1).strip())
        m = re.match(r'^no\s+(.+)', text, re.IGNORECASE)
        if m: return UnaryOpNode('¬', self._parse_recursive(m.group(1).strip()))
        return None

    # ── Atomic proposition fallback ──
    def _parse_atomic(self, text: str) -> LogicNode:
        p_id_or_negated = self.registry.register(text)
        if p_id_or_negated.startswith('¬'):
            return UnaryOpNode('¬', AtomicNode(p_id_or_negated[1:]))
        return AtomicNode(p_id_or_negated)

    def _parse_recursive(self, text: str) -> LogicNode:
        text = text.strip()
        if not text: return AtomicNode("∅")
        
        if text.lower().startswith('si ') and ',' in text and not text.lower().startswith('si y solo si'):
            partes = self._split_si_comma(text)
            if partes: 
                return self._build_binary('→', partes[0], partes[1])

        if re.search(r'\bno\s*$', text):
            clean_text = re.sub(r'\bno\s*$', '', text).strip()
            return UnaryOpNode('¬', self._parse_recursive(clean_text))

        res = self._parse_block_negation(text)
        if res: return res
        res = self._parse_biconditional(text)
        if res: return res
        res = self._parse_conditional(text)
        if res: return res
        res = self._parse_nor(text)
        if res: return res
        res = self._parse_disjunction(text)
        if res: return res
        res = self._parse_conjunction(text)
        if res: return res
        res = self._parse_negation(text)
        if res: return res
        return self._parse_atomic(text)

    def parse(self, sentence: str) -> dict:
        """Parse a sentence from scratch using the full recursive descent parser."""
        self.registry = PropositionRegistry()
        ast = self._parse_recursive(self._normalize(sentence))
        return {"formula": ast.to_formula(), "propositions": self.registry.get_dictionary()}

class GuidedParser:
    """Simpler parser guided by the ML-predicted main connective.
    
    Splits the sentence at the appropriate linguistic marker for the given
    connective and registers both halves as atomic propositions.
    """
    def __init__(self):
        self.registry = PropositionRegistry()

    @staticmethod
    def _normalize(text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[.!?]+$', '', text)
        return text.strip()

    def _split_si_comma(self, text: str) -> Optional[Tuple[str, str]]:
        match = re.search(r'^si\s+(.+?),\s+([^,]+)$', text, re.IGNORECASE)
        if match: return match.group(1), match.group(2)
        match = re.search(r'^si\s+(.+?),\s*(.+)', text, re.IGNORECASE | re.DOTALL)
        if match: return match.group(1), match.group(2)
        return None

    def _build_binary(self, op: str, left_text: str, right_text: str) -> LogicNode:
        node_left = AtomicNode(self.registry.register(left_text))
        node_right = AtomicNode(self.registry.register(right_text))
        return BinaryOpNode(op, node_left, node_right)

    def parse(self, sentence: str, main_conn: str) -> dict:
        text = self._normalize(sentence)

        if re.search(r'\bno\s*$', text):
            text = re.sub(r'\bno\s*$', '', text).strip()
            text = "no " + text

        left = right = None

        if main_conn == '→':
            if re.match(r'^solo\s+si\s+(.+?)(?:,\s*|\s+)(.+)', text, re.IGNORECASE | re.DOTALL):
                m = re.match(r'^solo\s+si\s+(.+?)(?:,\s*|\s+)(.+)', text, re.IGNORECASE | re.DOTALL)
                if m: 
                    left, right = m.group(2), m.group(1)
            if re.search(r'\bentonces\b', text, re.IGNORECASE):
                m = re.search(r'si\s+(.+?)\s+entonces\s+(.+)', text, re.IGNORECASE | re.DOTALL)
                if m: left, right = m.group(1), m.group(2)
            elif text.lower().startswith('si ') and ',' in text:
                partes = self._split_si_comma(text)
                if partes: left, right = partes
            elif re.match(r'^siempre\s+que\s+(.+?)\s+entonces\s+(.+)', text, re.IGNORECASE):
                m = re.match(r'^siempre\s+que\s+(.+?)\s+entonces\s+(.+)', text, re.IGNORECASE | re.DOTALL)
                if m: left, right = m.group(1), m.group(2)
            elif re.match(r'^siempre\s+que\s+(.+?),\s*(.+)', text, re.IGNORECASE):
                m = re.match(r'^siempre\s+que\s+(.+?),\s*(.+)', text, re.IGNORECASE | re.DOTALL)
                if m: left, right = m.group(1), m.group(2)
            elif re.match(r'^cuando\s+(.+?)\s+entonces\s+(.+)', text, re.IGNORECASE):
                m = re.match(r'^cuando\s+(.+?)\s+entonces\s+(.+)', text, re.IGNORECASE | re.DOTALL)
                if m: left, right = m.group(1), m.group(2)
            elif re.match(r'^cuando\s+(.+?),\s*(.+)', text, re.IGNORECASE):
                m = re.match(r'^cuando\s+(.+?),\s*(.+)', text, re.IGNORECASE | re.DOTALL)
                if m: left, right = m.group(1), m.group(2)
            elif re.search(r'\bsolo\s+si\b', text, re.IGNORECASE):
                m = re.search(r'(.+?)\s+\bsolo\s+si\b\s+(.+)', text, re.IGNORECASE)
                if m: left, right = m.group(1), m.group(2)
            elif re.search(r'\ba\s+menos\s+que\b', text, re.IGNORECASE):
                m = re.search(r'(.+?)\s+a\s+menos\s+que\s+(.+)', text, re.IGNORECASE)
                if m:
                    left_text = m.group(1).strip()
                    right_text = m.group(2).strip()
                    left_node = AtomicNode(self.registry.register(left_text))
                    right_node = AtomicNode(self.registry.register(right_text))
                    ast = BinaryOpNode('→', UnaryOpNode('¬', right_node), left_node)
                    return {"formula": ast.to_formula(), "propositions": self.registry.get_dictionary()}
            elif re.match(r'^es\s+necesario\s+(.+?)\s+para\s+(.+)$', text, re.IGNORECASE):
                m = re.match(r'^es\s+necesario\s+(.+?)\s+para\s+(.+)$', text, re.IGNORECASE)
                if m: left, right = m.group(2), m.group(1)
            elif re.match(r'^una\s+condici[oó]n\s+necesaria\s+para\s+(.+?)\s+es\s+(.+)$', text, re.IGNORECASE):
                m = re.match(r'^una\s+condici[oó]n\s+necesaria\s+para\s+(.+?)\s+es\s+(.+)$', text, re.IGNORECASE)
                if m: left, right = m.group(1), m.group(2)
            elif re.match(r'^una\s+condici[oó]n\s+suficiente\s+para\s+(.+?)\s+es\s+(.+)$', text, re.IGNORECASE):
                m = re.match(r'^una\s+condici[oó]n\s+suficiente\s+para\s+(.+?)\s+es\s+(.+)$', text, re.IGNORECASE)
                if m: left, right = m.group(2), m.group(1)
            elif re.match(r'^(.+?)\s+es\s+necesari[oa]\s+para\s+(.+)$', text, re.IGNORECASE):
                m = re.match(r'^(.+?)\s+es\s+necesari[oa]\s+para\s+(.+)$', text, re.IGNORECASE)
                if m: left, right = m.group(2), m.group(1)
            elif re.match(r'^(.+?)\s+es\s+suficiente\s+para\s+(.+)$', text, re.IGNORECASE):
                m = re.match(r'^(.+?)\s+es\s+suficiente\s+para\s+(.+)$', text, re.IGNORECASE)
                if m: left, right = m.group(1), m.group(2)
            elif re.match(r'^basta\s+(.+?)\s+para\s+(.+)$', text, re.IGNORECASE):
                m = re.match(r'^basta\s+(.+?)\s+para\s+(.+)$', text, re.IGNORECASE)
                if m: left, right = m.group(1), m.group(2)

        elif main_conn == '↔':
            if re.search(r'\b(?:si\s+y\s+solo\s+si|cuando\s+y\s+solamente\s+cuando|cuando\s+y\s+solo\s+cuando)\b', text, re.IGNORECASE):
                m = re.search(r'(.+?)\s+(?:si,?\s+y\s+solo\s+si,?|cuando\s+y\s+solamente\s+cuando|cuando\s+y\s+solo\s+cuando)\s+(.+)', text, re.IGNORECASE)
                if m: left, right = m.group(1), m.group(2)
            elif re.search(r'\bes\s+equivalente\s+a\b', text, re.IGNORECASE):
                m = re.search(r'(.+?)\s+es\s+equivalente\s+a\s+(.+)', text, re.IGNORECASE)
                if m: left, right = m.group(1), m.group(2)
            elif re.search(r'\bequivale\s+a\b', text, re.IGNORECASE):
                m = re.search(r'(.+?)\s+equivale\s+a\s+(.+)', text, re.IGNORECASE)
                if m: left, right = m.group(1), m.group(2)
            elif re.search(r'\bnecesari[oa]s?\s+y\s+suficiente\b', text, re.IGNORECASE):
                m = re.search(r'(.+?)\s+es\s+necesari[oa]s?\s+y\s+suficiente\s+para\s+(.+)', text, re.IGNORECASE)
                if m: left, right = m.group(1), m.group(2)

        elif main_conn == '∧':
            m = re.search(r'(.+?)\s+\b(y|e|pero)\b\s+(.+)', text, re.IGNORECASE)
            if m: left, right = m.group(1), m.group(3)
            elif ',' in text:
                parts = text.split(',', 1)
                left, right = parts[0], parts[1]

        elif main_conn == '∨':
            m = re.match(r'^\s*(?:o\s+bien|o|u)\s+(.+?)\s+(?:o\s+bien|o|u)\s+(.+?)(?:\s+(?:o|u)\s+ambas?\s*(?:cosas?)?)?\s*$', text, re.IGNORECASE)
            if m:
                left = self._parse_recursive(m.group(1))
                right = self._parse_recursive(m.group(2))
                return BinaryOpNode('∨', left, right)
            else:
                m = re.search(r'(.+?)\s+\bo\b\s+(.+)', text, re.IGNORECASE)
                if m: left, right = m.group(1), m.group(2)

        if left and right:
            if main_conn == '→' and re.search(r'\bsolo\s+si\b', text, re.IGNORECASE):
                ast = self._build_binary('→', right.strip(), left.strip())
            else:
                ast = self._build_binary(main_conn, left.strip(), right.strip())
        elif main_conn == '¬' and re.match(r'^no\s+(.+)', text, re.IGNORECASE):
            m = re.match(r'^no\s+(.+)', text, re.IGNORECASE)
            child_text = m.group(1).strip()
            child_id = self.registry.register(child_text)
            if child_id.startswith('¬'):
                ast = UnaryOpNode('¬', AtomicNode(child_id[1:]))
            else:
                ast = UnaryOpNode('¬', AtomicNode(child_id))
        else:
            p_id = self.registry.register(text)
            if p_id.startswith('¬'):
                ast = UnaryOpNode('¬', AtomicNode(p_id[1:]))
            else:
                ast = AtomicNode(p_id)

        return {"formula": ast.to_formula(), "propositions": self.registry.get_dictionary()}

def get_main_connective_from_template(template: str) -> str:
    depth = 0
    for ch in template:
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        elif depth == 0 and ch in ('→', '↔', '∧', '∨'):
            return ch
            
    if template.strip().startswith('¬'):
        return '¬'
        
    return 'ATOMIC'

class HybridLogicParser:
    """Public parser API: staged pipeline with beam, FullRecursive as fallback."""

    def __init__(self, ml_pipeline=None, confidence_threshold=0.6):
        self.ml_pipeline = ml_pipeline
        self.threshold = confidence_threshold
        self.guided_parser = GuidedParser()
        self.full_parser = FullRecursiveParser()
        self._pipeline = None

    def _get_pipeline(self):
        if self._pipeline is None:
            from pipeline_parser import PipelineParser
            self._pipeline = PipelineParser(boundary_clf=self.ml_pipeline, fallback=False)
        return self._pipeline

    def parse(self, sentence: str) -> dict:
        return self._get_pipeline().parse(sentence)

    def parse_candidates(self, sentence: str, max_k: int = 3) -> list:
        return self._get_pipeline().parse_candidates(sentence, max_k=max_k)

    def parse_to_semantics(self, sentence: str, main_conn: Optional[str] = None) -> dict:
        """Parse with an explicit main connector override (bypasses ML prediction)."""
        if main_conn is not None:
            logger.info(f"DEBUG: Intentando usar parser guiado con: {main_conn}")
            result = self.guided_parser.parse(sentence, main_conn)
            if not self._is_structure_compatible(result["formula"], main_conn):
                logger.info("DEBUG: La estructura no es compatible, cayendo a parser recursivo.")
                return self.full_parser.parse(sentence)
            return result
        return self.parse(sentence)

    @staticmethod
    def _is_structure_compatible(formula: str, expected_conn: str) -> bool:
        if expected_conn == 'ATOMIC':
            return re.match(r'^[¬]?[A-Z]$', formula.replace('(', '').replace(')', '')) is not None
        if expected_conn == '¬':
            return formula.startswith('¬')
        depth = 0
        for ch in formula:
            if ch == '(': depth += 1
            elif ch == ')': depth -= 1
            elif depth == 0 and ch == expected_conn:
                return True
        return False

if __name__ == "__main__":
    import sys
    from ml_pipeline import HybridPipelineTrainer
    confidence : float = 0.7
    trainer = HybridPipelineTrainer(data_path='examples.json')
    parser = HybridLogicParser(trainer, confidence_threshold=confidence)
    if len(sys.argv) < 2:
        print("Uso: python hybrid_parser.py \"oración\" [conector]")
        sys.exit(1)
    sentence = sys.argv[1]
    main_conn = sys.argv[2] if len(sys.argv) >= 3 else None
    result = parser.parse_to_semantics(sentence, main_conn)
    print("Fórmula:", result["formula"])
    print("Proposiciones:", result["propositions"])