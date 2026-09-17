# N2LPI — Natural Language to Propositional Logic Interface

Translate Spanish sentences into classical propositional logic formulas.

> **Note:** The output is an **approximation**. Nested scope and AND/OR grouping without parentheses can still disagree with a gold tree. Sentential *no* is `¬` over a positive atom; adjective-internal *no* (*no nulo*, *no inicializada*) stays inside the atom.

## What changed vs the original N2LPI

The previous system guessed a **root connective** with two sklearn stages and then split the sentence with a regex-heavy recursive parser.

This version:

- Labels **each token** as a proposition piece or as glue (`si`, `y`, `a menos que`, `si bien`, …).
- Builds the formula with a **4-stage pipeline**: cues → clause boundary → operator precedence → AST.
- Keeps **train and test disjoint** (no leaked copies of the same sentence).
- Covers everyday Spanish nexos, not only `si y solo si` / `y` / `o`.

## Current results (CPU, `es_core_news_sm`)

| Bank | Score |
|---|---|
| Held-out JSON test (`data/test_dataset.json`) | **69/70** |
| Mixed 30-item stress set | **30/30** |
| 101 real-style sentences (`scripts/stress_real100.py`) | **101/101** |

Known JSON miss: item 460 (AND/OR grouping: gold `¬(P ∧ (Q ∨ R))` vs parser `¬((P ∧ Q) ∨ R)`).

## Features

- **Joint token tagger** (`span_tagger.py`): BIO labels for PROP vs IFF / IF / UNLESS / AND / OR / NEG / AUNQUE / THEN / …
- **Staged parser** (`pipeline_parser.py`): precedence for `↔`, `→` (including *a menos que*), `∨`, `∧`, block negation
- **Beam of 2–3 candidates** in the REPL, with optional gold/REPL feedback
- **Spanish nexos**: *si y solo si*, *cuando y solo cuando*, *siempre y cuando*, *a menos que*, *a no ser que*, *salvo que*, *excepto si*, *si bien*, *así que*, *por si*, *como* (threat), *pese a que*, *de lo contrario*, *tanto si … como si*, *no solo … sino*, XOR *o bien … no ambos*, …
- **Held-out datasets** and `scripts/validate_dataset.py` (schema, complexity, no train/test overlap)
- Optional **Ollama** comparison via `eval_models.py`

## Requirements

- Python 3.10+ recommended (3.8+ may work)
- `pandas`, `scikit-learn`, `joblib`, `sympy`, `nltk`, `spacy`, `requests`
- spaCy model `es_core_news_sm`
- Optional: [Ollama](https://ollama.ai) (e.g. `qwen2.5:3b`) for the LLM bake-off

## Installation

```bash
git clone https://github.com/3DKNHD/N2LPI-Natural-Language-to-Propositional-Logic-Interface
cd N2LPI-Natural-Language-to-Propositional-Logic-Interface/

python -m venv env
source env/bin/activate          # Windows: .\env\Scripts\activate

pip install pandas scikit-learn joblib sympy nltk spacy requests
python -m spacy download es_core_news_sm
```

Data already lives in `data/`:

- `data/training_dataset.json` — ~522 training sentences
- `data/test_dataset.json` — 70 held-out test sentences

Trained weights are written to `model/n2lpi_model_cache.joblib` (gitignored). First run trains if the cache is missing.

## Usage

```bash
python main.py
```

Interactive REPL: type a Spanish sentence, get formula + atoms. Optional 2–3 beam candidates and feedback.

```bash
python main.py data/test_dataset.json
```

Evaluate an external JSON file (same schema as train/test).

```bash
python scripts/validate_dataset.py
PYTHONPATH=. python scripts/stress_real100.py
python eval_models.py          # needs local Ollama
python hybrid_parser.py "Si llueve, me quedo en casa."
```

## Data format

```json
{
  "id": 1,
  "sentence": "El servidor responde y la aplicación funciona correctamente.",
  "propositions": { "P": "el servidor responde", "Q": "la aplicación funciona correctamente" },
  "formula": "P ∧ Q",
  "complexity": "simple",
  "connectives": ["∧"]
}
```

Proposition strings should match the sentence (same mood/lemmas). `complexity` follows the structural rubric in `scripts/validate_dataset.py`.

## Architecture

```
Spanish sentence
    ↓
JointSpanTagger (one BIO label per token: PROP vs connective kind)
    ↓
NMS cues + clause-boundary scores
    ↓
parse_range: precedence (NEG_BLOCK, IFF, UNLESS, IF/THEN, OR, AND, NEG)
    ↓
AST → formula + grounded atoms
```

1. **Tagging** — logistic regression on per-token dict features; hard glue for closed words (`y`, `o`, `si`, `no`, …) and frames (*siempre y cuando*, *de lo contrario*, *por si*).
2. **Boundaries** — histogram-gradient model over cue spans.
3. **Parse** — recursive split on the highest-precedence cue in range (wrapping `si … entonces` before inner IFF, inverted `X si Y`, *a menos que* as `¬Q → P`, etc.).
4. **Equivalence** — `logic_utils.son_formulas_equivalentes` (SymPy, variables in appearance order).

The old root-connective classifiers remain in `ml_pipeline.py` for cache compatibility; **inference of the formula uses the tagger + `parse_range`**.

## Project structure

```
├── main.py                    # Load/train cache, demo, REPL or eval JSON
├── cli.py                     # Demo, external eval, interactive loop
├── span_tagger.py             # Joint token tagger + gold BIO from aligned props
├── pipeline_parser.py         # Four-stage parser
├── ml_pipeline.py             # Features, trainers, model I/O
├── hybrid_parser.py           # HybridLogicParser → PipelineParser
├── logic_utils.py             # Formula equivalence
├── eval_models.py             # Optional vs Ollama
├── scripts/
│   ├── validate_dataset.py
│   └── stress_real100.py
├── data/
│   ├── training_dataset.json
│   └── test_dataset.json
└── model/                     # gitignored cache
```

## License

[MIT License]

---

# N2LPI — Lenguaje Natural a Lógica Proposicional

Traduce oraciones en español a fórmulas de lógica proposicional clásica.

> **Nota:** la traducción es una **aproximación**. El alcance anidado y el agrupado AND/OR sin paréntesis pueden no coincidir con el gold. El *no* sentencial es `¬` sobre un átomo positivo; el *no* interno de adjetivo (*no nulo*, *no inicializada*) se queda en el átomo.

## Qué cambió respecto al N2LPI original

El sistema anterior **adivinaba el conector raíz** con dos clasificadores y partía la oración con un parser recursivo lleno de regex.

Esta versión:

- Etiqueta **cada token** como pieza de proposición o como nexo.
- Arma la fórmula en **4 etapas**: cues → frontera de cláusula → precedencia → AST.
- Separa **train y test** (sin copias de la misma oración).
- Cubre nexos de español cotidiano, no solo `si y solo si` / `y` / `o`.

## Resultados actuales (CPU, `es_core_news_sm`)

| Banco | Resultado |
|---|---|
| JSON de prueba held-out | **69/70** |
| 30 de estrés mezclado | **30/30** |
| 101 oraciones de estilo real | **101/101** |

Fallo conocido del JSON: el 460 (agrupado AND/OR: gold `¬(P ∧ (Q ∨ R))` vs parser `¬((P ∧ Q) ∨ R)`).

## Características

- **Etiquetador conjunto de tokens** (`span_tagger.py`)
- **Parser por etapas** (`pipeline_parser.py`)
- **Beam de 2–3 candidatos** y feedback opcional en el REPL
- Nexos: *si y solo si*, *a menos que*, *si bien*, *así que*, *por si*, *como* de amenaza, *pese a que*, *de lo contrario*, *tanto si … como si*, *no solo … sino*, XOR, …
- Datasets held-out y validador `scripts/validate_dataset.py`

## Requisitos e instalación

Python 3.10+ recomendado. Igual que la sección en inglés: venv, `pip install pandas scikit-learn joblib sympy nltk spacy requests`, `python -m spacy download es_core_news_sm`.

Hay ~522 oraciones de train y 70 de test en `data/`. La caché del modelo se guarda en `model/` (no va a git); si no existe, `python main.py` reentrena.

## Uso

```bash
python main.py
python main.py data/test_dataset.json
python scripts/validate_dataset.py
PYTHONPATH=. python scripts/stress_real100.py
```

## Cómo funciona

```
Oración en español
    ↓
JointSpanTagger (BIO: PROP vs tipo de nexo)
    ↓
Cues + frontera de cláusula
    ↓
parse_range (precedencia: NEG_BLOCK, IFF, UNLESS, IF/THEN, OR, AND, NEG)
    ↓
AST → fórmula + átomos
```

Los clasificadores viejos de conector raíz siguen en `ml_pipeline.py` por la caché; **la fórmula se arma con el tagger y `parse_range`**.

## Licencia

[MIT License]
