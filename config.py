#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Configuration, linguistic resources, stemmer, spaCy setup, stopwords,
negation markers, and logging for the N2LPI translator.
"""

import re

# ── NLTK Stemmer (Spanish) ──
try:
    from nltk.stem import SnowballStemmer
    SPANISH_STEMMER = SnowballStemmer('spanish')
except ImportError:
    SPANISH_STEMMER = None

# ── spaCy Spanish Model (clause splitting) ──
try:
    import spacy as _spacy
    NLP_SPANISH = _spacy.load('es_core_news_sm')
    print("[spaCy] Modelo 'es_core_news_sm' cargado — segmentación activa.")
except Exception:
    NLP_SPANISH = None
    print("[spaCy] No disponible — usando heurísticas regex de fallback.")

# ── Spanish Verb Suffixes (for heuristic verb detection) ──
VERB_ENDINGS = (
    "ar", "er", "ir", "amos", "áis", "emos", "éis", "imos", "ís",
    "aba", "abas", "ábamos", "abais", "aban", "ía", "ías", "íamos", 
    "íais", "ían", "ando", "iendo", "ado", "ido", "aré", "arás", 
    "ará", "aremos", "aréis", "arán", "eré", "erás", "erá", "eremos", 
    "eréis", "erán", "as", "es", "ás", "és",
)

# ── Spanish Subject Pronouns ──
SUBJECT_PRONOUNS = {
    "yo", "tú", "él", "ella", "usted", "nosotros", "nosotras",
    "vosotros", "vosotras", "ellos", "ellas", "ustedes",
}

# ── Stopwords Excluded from Proposition Extraction ──
CONNECTIVE_STOPWORDS = {
    'si', 'entonces', 'y', 'o', 'no', 'solo', 'sí',
    'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas',
    'de', 'del', 'al', 'a', 'ante', 'bajo', 'cabe', 'con', 'contra',
    'desde', 'durante', 'en', 'entre', 'hacia', 'hasta', 'mediante',
    'para', 'por', 'según', 'sin', 'so', 'sobre', 'tras', 'vs'
}

# ── Phrases That Trigger Block Negation (e.g., "no es cierto que...") ──
NEGATION_START_PHRASES = [
    'no es cierto que', 'es falso que', 'no ocurre que',
    'no es verdad que', 'en absoluto sucede que',
    'es mentira que', 'no es el caso que',
    'no se cumple que'
]

def simple_spanish_stem(word: str) -> str:
    """Stem a Spanish word using NLTK SnowballStemmer, falling back to suffix stripping."""
    if SPANISH_STEMMER:
        return SPANISH_STEMMER.stem(word.lower())
    
    word = word.lower()
    if len(word) < 4:
        return word
    for suffix in ['iendo', 'ando', 'ción', 'iones', 'mente', 'idades', 'idad', 'ado', 'ido']:
        if word.endswith(suffix) and len(word) - len(suffix) > 2:
            return word[:-len(suffix)]
    return word

# ── Logging Configuration ──
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(module)s] - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('N2LPI')