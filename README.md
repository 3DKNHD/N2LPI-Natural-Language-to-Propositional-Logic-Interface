# N2LPI — Natural Language to Propositional Logic Interface

Translate Spanish sentences into classical propositional logic formulas using a hybrid ML + rule-based approach.

## Features

- **Two-stage ML pipeline**: Multi-label classifier predicts sub-connectors (¬, ∧, ∨), then a Random Forest predicts the main connective (→, ↔, ∧, ∨, ¬, ATOMIC)
- **Hybrid parser**: Uses ML-guided parsing when confident, full recursive-descent parser as fallback
- **Handcrafted features + TF-IDF**: 80+ linguistic features (conjunction/disjunction/conditional markers, negation patterns, structural hints) combined with character n-gram TF-IDF vectors
- **Spanish-optimized**: spaCy `es_core_news_sm` for clause splitting, NLTK SnowballStemmer, Spanish verb ending heuristics
- **Interactive REPL**: Type sentences and get instant logical formulas
- **Competitive evaluation**: Compare against Ollama LLMs (e.g., qwen2.5:3b) on accuracy and latency

## Requirements

- Python 3.8+
- Dependencies:
  ```
  pip install pandas scikit-learn joblib sympy nltk spacy ollama
  python -m spacy download es_core_news_sm
  ```
- Optional (for LLM comparison): [Ollama](https://ollama.ai) with a model like `qwen2.5:3b`

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/3DKNHD/N2LPI-Natural-Language-to-Propositional-Logic-Interface
   cd N2LPI-Natural-Language-to-Propositional-Logic-Interface/
   ```

2. (Recommended) Create and activate a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate   # Linux/Mac
   # or: .\env\Scripts\activate  # Windows
   ```

3. Install dependencies:
   ```bash
   pip install pandas scikit-learn joblib sympy nltk spacy requests ollama
   ```

4. Download the Spanish spaCy model:
   ```bash
   python -m spacy download es_core_news_sm
   ```

5. Place your data files in `data/`:
   - `data/training_dataset.json` — training set
   - `data/test_dataset.json` — test/evaluation set

## Usage

### Interactive Mode
```bash
python main.py
```
Type Spanish sentences at the prompt. The system will output the predicted logical formula and atomic propositions.

### Evaluate External Dataset
```bash
python main.py path/to/test_dataset.json
```

### Competitive Evaluation (vs Ollama LLM)
```bash
python eval_models.py
```
Requires Ollama running locally with the configured model.

### Command-Line Parsing
```bash
python hybrid_parser.py "Si llueve, me quedo en casa."
```

## Data Format

Training/test entries follow this JSON schema:
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

## Architecture

1. **Feature extraction** (`ml_pipeline.py:extract_advanced_features`) — 80+ linguistic features from the input sentence
2. **TF-IDF vectorization** — character n-grams (2–4) with sublinear TF scaling
3. **Stage 1 classifier** — `MultiOutputClassifier` with `HistGradientBoosting` predicts presence of ¬, ∧, ∨
4. **Feature stacking** — Stage 1 predictions become additional input features for Stage 2
5. **Stage 2 classifier** — `RandomForestClassifier` predicts the main connective (→, ↔, ∧, ∨, ¬, ATOMIC)
6. **Parser dispatch** — `HybridLogicParser` chooses between `GuidedParser` (ML-driven atomic split) and `FullRecursiveParser` (rule-based recursive descent with full pattern coverage) based on the confidence threshold

## Project Structure

```
├── main.py              # Entry point — train/load, demo, REPL
├── cli.py               # CLI: demonstration, external eval, interactive loop
├── config.py            # Linguistic resources, spaCy/NLTK setup, logging
├── ml_pipeline.py       # ML pipeline: features, training, prediction, model I/O
├── hybrid_parser.py     # Parsers: FullRecursive, Guided, HybridLogic, AST nodes
├── logic_utils.py       # Formula equivalence, SymPy bridge, variable normalization
├── eval_models.py       # Competitive evaluation vs Ollama LLMs
├── data/
│   ├── training_dataset.json   # 191 training examples
│   └── test_dataset.json       # ~60 test examples (simple → complex)
└── model/
    └── n2lpi_model_cache.joblib  # Cached trained models
```

## How It Works (High-Level)

```
Spanish sentence
    ↓
extract_advanced_features() + TF-IDF transform
    ↓
Stage 1: MultiOutputClassifier → predicts ¬/∧/∨ presence
    ↓  (predictions stacked as additional features)
Stage 2: RandomForest → predicts main connective (→ ↔ ∧ ∨ ¬ ATOMIC)
    ↓
confidence ≥ 0.7?  →  GuidedParser (splits by predicted connective)
confidence < 0.7?  →  FullRecursiveParser (recursive descent, handles all patterns)
    ↓
Propositional logic formula + grounded propositions
```

## License

[MIT License]

---

# N2LPI — Lenguaje Natural a Lógica Proposicional

Traduce oraciones en español a fórmulas de lógica proposicional clásica usando un enfoque híbrido de aprendizaje automático + reglas lingüísticas.

## Características

- **Pipeline ML de dos etapas**: Un clasificador multietiqueta predice los sub-conectores (¬, ∧, ∨), luego un Random Forest predice el conector principal (→, ↔, ∧, ∨, ¬, ATOMIC)
- **Parser híbrido**: Usa parsing guiado por ML cuando hay confianza suficiente, y un parser recursivo descendente completo como respaldo
- **Features artesanales + TF-IDF**: Más de 80 características lingüísticas (marcadores de conjunción/disyunción/condicional, patrones de negación, pistas estructurales) combinadas con vectores TF-IDF de n-gramas de caracteres
- **Optimizado para español**: spaCy `es_core_news_sm` para división de cláusulas, NLTK SnowballStemmer, heurísticas de terminaciones verbales en español
- **REPL interactivo**: Escribe oraciones y obtén fórmulas lógicas al instante
- **Evaluación competitiva**: Compara contra LLMs de Ollama (ej. qwen2.5:3b) en precisión y latencia

## Requisitos

- Python 3.8+
- Dependencias:
  ```
  pip install pandas scikit-learn joblib sympy nltk spacy ollama
  python -m spacy download es_core_news_sm
  ```
- Opcional (para comparación con LLM): [Ollama](https://ollama.ai) con un modelo como `qwen2.5:3b`

## Instalación

1. Clona el repositorio:
   ```bash
   git clone https://github.com/3DKNHD/N2LPI-Natural-Language-to-Propositional-Logic-Interface
   cd N2LPI-Natural-Language-to-Propositional-Logic-Interface/
   ```

2. (Recomendado) Crea y activa un entorno virtual:
   ```bash
   python -m venv env
   source env/bin/activate   # Linux/Mac
   # o: .\env\Scripts\activate  # Windows
   ```

3. Instala las dependencias:
   ```bash
   pip install pandas scikit-learn joblib sympy nltk spacy requests ollama
   ```

4. Descarga el modelo de spaCy en español:
   ```bash
   python -m spacy download es_core_news_sm
   ```

5. Coloca tus archivos de datos en `data/`:
   - `data/training_dataset.json` — conjunto de entrenamiento
   - `data/test_dataset.json` — conjunto de prueba/evaluación

## Uso

### Modo Interactivo
```bash
python main.py
```
Escribe oraciones en español en el prompt. El sistema mostrará la fórmula lógica predicha y las proposiciones atómicas.

### Evaluar Dataset Externo
```bash
python main.py ruta/al/test_dataset.json
```

### Evaluación Competitiva (vs LLM Ollama)
```bash
python eval_models.py
```
Requiere Ollama funcionando localmente con el modelo configurado.

### Parsing desde Línea de Comandos
```bash
python hybrid_parser.py "Si llueve, me quedo en casa."
```

## Formato de Datos

Las entradas de entrenamiento/prueba siguen este esquema JSON:
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

## Arquitectura

1. **Extracción de características** (`ml_pipeline.py:extract_advanced_features`) — más de 80 características lingüísticas extraídas de la oración
2. **Vectorización TF-IDF** — n-gramas de caracteres (2–4) con escalamiento TF sublineal
3. **Clasificador Etapa 1** — `MultiOutputClassifier` con `HistGradientBoosting` predice presencia de ¬, ∧, ∨
4. **Stacking de características** — las predicciones de la Etapa 1 se convierten en características adicionales para la Etapa 2
5. **Clasificador Etapa 2** — `RandomForestClassifier` predice el conector principal (→, ↔, ∧, ∨, ¬, ATOMIC)
6. **Despacho del parser** — `HybridLogicParser` elige entre `GuidedParser` (división atómica guiada por ML) y `FullRecursiveParser` (descenso recursivo basado en reglas con cobertura completa de patrones) según el umbral de confianza

## Estructura del Proyecto

```
├── main.py              # Punto de entrada — entrenar/cargar, demo, REPL
├── cli.py               # CLI: demostración, evaluación externa, bucle interactivo
├── config.py            # Recursos lingüísticos, configuración spaCy/NLTK, logging
├── ml_pipeline.py       # Pipeline ML: características, entrenamiento, predicción, E/S de modelos
├── hybrid_parser.py     # Parsers: FullRecursive, Guided, HybridLogic, nodos AST
├── logic_utils.py       # Equivalencia de fórmulas, puente SymPy, normalización de variables
├── eval_models.py       # Evaluación competitiva contra LLMs de Ollama
├── data/
│   ├── training_dataset.json   # 191 ejemplos de entrenamiento
│   └── test_dataset.json       # ~60 ejemplos de prueba (simple → complejo)
└── model/
    └── n2lpi_model_cache.joblib  # Modelos entrenados en caché
```

## Cómo Funciona (Alto Nivel)

```
Oración en español
    ↓
extract_advanced_features() + transformación TF-IDF
    ↓
Etapa 1: MultiOutputClassifier → predice presencia ¬/∧/∨
    ↓  (predicciones apiladas como características adicionales)
Etapa 2: RandomForest → predice conector principal (→ ↔ ∧ ∨ ¬ ATOMIC)
    ↓
confianza ≥ 0.7?  →  GuidedParser (divide por conector predicho)
confianza < 0.7?  →  FullRecursiveParser (descenso recursivo, cubre todos los patrones)
    ↓
Fórmula de lógica proposicional + proposiciones fundamentadas
```

## Licencia

[MIT License]
