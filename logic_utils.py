#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Formula equivalence checking via SymPy (logical equivalence) and structural
fallback. Includes AST-based formula parsing and variable normalization.
"""

import re
import string
import logging
from collections import OrderedDict
from typing import Dict, Tuple
import sympy
from sympy.logic.boolalg import And, Or, Not, Implies, Equivalent, simplify_logic

try:
    import sympy
    from sympy.logic.boolalg import simplify_logic, Equivalent
    from sympy.abc import _clash
    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False

from config import logger


# ── SymPy AST Conversion ──

def ast_node_to_sympy(node: tuple, symbols_cache: dict) -> sympy.Basic:
    """Convert a parsed AST node tuple into a SymPy boolean expression."""
    node_type = node[0]

    if node_type == 'var':
        var_name = node[1]
        if var_name not in symbols_cache:
            symbols_cache[var_name] = sympy.Symbol(var_name)
        return symbols_cache[var_name]

    elif node_type == 'not':
        return Not(ast_node_to_sympy(node[1], symbols_cache))

    elif node_type == '∧':
        return And(ast_node_to_sympy(node[1], symbols_cache), 
                   ast_node_to_sympy(node[2], symbols_cache))

    elif node_type == '∨':
        return Or(ast_node_to_sympy(node[1], symbols_cache), 
                  ast_node_to_sympy(node[2], symbols_cache))

    elif node_type == '→':
        return Implies(ast_node_to_sympy(node[1], symbols_cache), 
                       ast_node_to_sympy(node[2], symbols_cache))

    elif node_type == '↔':
        return Equivalent(ast_node_to_sympy(node[1], symbols_cache), 
                          ast_node_to_sympy(node[2], symbols_cache))

    else:
        raise ValueError(f"Unsupported AST node operator encountered: '{node_type}'")


def formula_to_sympy_expr(formula_str: str) -> sympy.Basic:
    """Parse a formula string and convert it to a SymPy expression."""
    custom_ast = _parse_formula(formula_str)
    
    symbols_cache = {}
    return ast_node_to_sympy(custom_ast, symbols_cache)

# ── Formula String Parser (Custom AST) ──

def _parse_formula(s: str):
    s = s.strip()
    while s.startswith('(') and s.endswith(')'):
        depth = 0
        covers_whole = True
        for i, ch in enumerate(s):
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
            if depth == 0 and i != len(s) - 1:
                covers_whole = False
                break
        if covers_whole:
            s = s[1:-1].strip()
        else:
            break

    if s.startswith('¬'):
        rest = s[1:].strip()
        return ('not', _parse_formula(rest))

    depth = 0
    for i, ch in enumerate(s):
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        elif depth == 0 and ch in {'∧', '∨', '→', '↔'}:
            left = _parse_formula(s[:i].strip())
            right = _parse_formula(s[i + 1:].strip())
            return (ch, left, right)

    if s.isalpha() and len(s) == 1:
        return ('var', s)
    raise ValueError(f"Fórmula inválida o no soportada: '{s}'")


def _orden_variables(arbol):
    orden = []
    def dfs(t):
        if t[0] == 'var':
            if t[1] not in orden:
                orden.append(t[1])
        elif t[0] == 'not':
            dfs(t[1])
        else:
            dfs(t[1])
            dfs(t[2])
    dfs(arbol)
    return orden


def _renombrar_variables(arbol, mapeo):
    if arbol[0] == 'var':
        return ('var', mapeo[arbol[1]])
    elif arbol[0] == 'not':
        return ('not', _renombrar_variables(arbol[1], mapeo))
    else:
        return (arbol[0],
                _renombrar_variables(arbol[1], mapeo),
                _renombrar_variables(arbol[2], mapeo))


def _arbol_a_cadena(arbol):
    if arbol[0] == 'var':
        return arbol[1]
    elif arbol[0] == 'not':
        hijo = _arbol_a_cadena(arbol[1])
        if arbol[1][0] not in ('var', 'not'):
            return f"¬({hijo})"
        else:
            return f"¬{hijo}"
    else:
        izquierda = _arbol_a_cadena(arbol[1])
        derecha = _arbol_a_cadena(arbol[2])
        return f"({izquierda} {arbol[0]} {derecha})"


# ── Structural Equivalence Fallback ──

def _fallback_equivalencia_estructural(f1: str, f2: str) -> bool:
    try:
        arbol1 = _parse_formula(f1)
        arbol2 = _parse_formula(f2)
    except Exception:
        return f1.strip() == f2.strip()

    vars1 = _orden_variables(arbol1)
    vars2 = _orden_variables(arbol2)

    mapeo1 = {var: f"v{i}" for i, var in enumerate(vars1)}
    mapeo2 = {var: f"v{i}" for i, var in enumerate(vars2)}

    arbol1_norm = _renombrar_variables(arbol1, mapeo1)
    arbol2_norm = _renombrar_variables(arbol2, mapeo2)

    return _arbol_a_cadena(arbol1_norm) == _arbol_a_cadena(arbol2_norm)


# ── Variable Normalization ──

def _normalize_formula_variables(formula_str: str) -> str:
    try:
        arbol = _parse_formula(formula_str)
        vars_todas = []
        def dfs(t):
            if t[0] == 'var':
                if t[1] not in vars_todas:
                    vars_todas.append(t[1])
            elif t[0] == 'not':
                dfs(t[1])
            else:
                dfs(t[1])
                dfs(t[2])
        dfs(arbol)
        
        vars_ordenadas = list(vars_todas)
        mapeo = {var: string.ascii_uppercase[i] for i, var in enumerate(vars_ordenadas)}
        
        arbol_norm = _renombrar_variables(arbol, mapeo)
        return _arbol_a_cadena(arbol_norm)
    except Exception:
        return formula_str


# ── Formula Equivalence Public API ──

def son_formulas_equivalentes(f1: str, f2: str) -> bool:
    """Check whether two formula strings are logically equivalent.
    
    Tries SymPy logical equivalence first; falls back to structural
    comparison (variable-normalized string equality) if SymPy is unavailable
    or raises an exception.
    """
    if f1.strip() == f2.strip():
        return True
        
    f1_norm = _normalize_formula_variables(f1)
    f2_norm = _normalize_formula_variables(f2)

    if f1_norm.strip() == f2_norm.strip():
        return True

    if not HAS_SYMPY:
        return _fallback_equivalencia_estructural(f1_norm, f2_norm)
    
    try:
        expr1 = formula_to_sympy_expr(f1_norm)
        expr2 = formula_to_sympy_expr(f2_norm)

        return bool(expr1.equals(expr2))

    except Exception as e:
        logger.warning(f"Error compiling AST to SymPy ({e}). Switching to structural fallback.")
        return _fallback_equivalencia_estructural(f1, f2)


def standardize_formula_variables(raw_formula: str) -> Tuple[str, Dict[str, str]]:
    """Rename propositional variables to A, B, C, ... in order of appearance.
    
    Returns (normalized_formula, inverse_mapping).
    """
    tokens = re.split(r'(\s*[∧∨→↔¬()]\s*)', raw_formula)
    atom_order = []
    for token in tokens:
        stripped = token.strip()
        if not stripped or stripped in {'∧', '∨', '→', '↔', '¬', '(', ')'}:
            continue
        if stripped not in atom_order:
            atom_order.append(stripped)

    var_map = OrderedDict()
    for atom in atom_order:
        if atom not in var_map:
            idx = len(var_map)
            if idx < 26:
                var_map[atom] = string.ascii_uppercase[idx]
            else:
                var_map[atom] = f"P{idx - 25}"

    result = raw_formula
    placeholders = {}
    for i, atom in enumerate(var_map.keys()):
        ph = f"__PH{i}__"
        placeholders[ph] = var_map[atom]
        result = result.replace(atom, ph)

    for ph, var in placeholders.items():
        result = result.replace(ph, var)

    inv_map = {v: k for k, v in var_map.items()}
    return result, inv_map