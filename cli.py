#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CLI entry points for the N2LPI translator:
  - run_demonstration: shows a single example with full AST output
  - test_external_dataset: evaluates against a JSON test file (top-1 + beam)
  - interactive_loop: REPL with optional 2–3 candidates and feedback learning
"""

import json
from ml_pipeline import HybridPipelineTrainer
from logic_utils import son_formulas_equivalentes
from config import logger
from hybrid_parser import HybridLogicParser
from pipeline_parser import credit_labels

confidence: float = 0.7


def _parser(trainer: HybridPipelineTrainer) -> HybridLogicParser:
    return HybridLogicParser(trainer, confidence_threshold=confidence)


def run_demonstration(trainer: HybridPipelineTrainer):
    """Show a single demonstration: sentence → formula with proposition grounding."""
    test_example = trainer.df.iloc[-1]
    sentence = test_example['sentence']
    true_formula = test_example['formula']

    parser = _parser(trainer)
    candidates = parser.parse_candidates(sentence)
    result = candidates[0]
    generated_formula = result['formula']

    print("\n" + "=" * 60)
    print("DEMOSTRACIÓN DEL PIPELINE POR ETAPAS")
    print("=" * 60)
    print(f"Oración original: {sentence}")
    if result.get("ops"):
        print("Cues frontera:")
        for op in result["ops"]:
            print(f"  {op['cue']!r} → {op['op']} ({op['conf']})")

    print("\nProposiciones Extraídas (Grounding):")
    for var, phrase in result['propositions'].items():
        print(f"  {var} = \"{phrase}\"")

    print(f"\nFórmula lógica generada: {generated_formula}")
    print(f"Fórmula verdadera (GT): {true_formula}")
    print(f"Coincidencia estructural: {son_formulas_equivalentes(generated_formula, true_formula)}")
    if len(candidates) > 1:
        print("Candidatos del beam:")
        for i, cand in enumerate(candidates, 1):
            print(f"  {i}. {cand['formula']}")
    print("=" * 60)


def _gold_in_beam(candidates, true_formula):
    for i, cand in enumerate(candidates):
        try:
            if son_formulas_equivalentes(cand.get("formula", ""), true_formula):
                return i
        except Exception:
            if cand.get("formula") == true_formula:
                return i
    return None


def test_external_dataset(trainer: HybridPipelineTrainer, test_path: str) -> None:
    """Evaluate the pipeline against a JSON test file (top-1, beam hit, optional gold learn)."""
    try:
        with open(test_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Error cargando el dataset externo: {e}")
        return

    print(f"\n{'='*60}\nEVALUACIÓN SOBRE DATASET EXTERNO: {test_path}\n{'='*60}")
    total = 0
    correct_top1 = 0
    correct_beam = 0
    pending_learn = []
    parser = _parser(trainer)

    for item in test_data:
        sentence = item.get('sentence', '')
        if not sentence:
            continue
        true_formula = item.get('formula', None)
        candidates = parser.parse_candidates(sentence)
        generated_formula = candidates[0]['formula']

        total += 1
        eq = False
        beam_i = None
        if true_formula:
            eq = son_formulas_equivalentes(generated_formula, true_formula)
            if eq:
                correct_top1 += 1
            beam_i = _gold_in_beam(candidates, true_formula)
            if beam_i is not None:
                correct_beam += 1
                pending_learn.append((sentence, candidates, beam_i, true_formula))

        print(f"\nOración: {sentence}")
        print(f"  Fórmula generada: {generated_formula}")
        if true_formula:
            print(f"  Gold: {true_formula}")
            print(f"  Top-1: {'Sí' if eq else 'No'}  |  Beam: {'Sí' if beam_i is not None else 'No'}")

    if total > 0 and test_data[0].get('formula'):
        print(f"\n{'='*60}")
        print(f"Precisión top-1: {correct_top1 / total:.3f} ({correct_top1}/{total})")
        print(f"Precisión beam (gold en 2–3): {correct_beam / total:.3f} ({correct_beam}/{total})")
        print(f"{'='*60}")

    learned = 0
    if trainer.boundary is not None:
        for sentence, candidates, winner_i, gold in pending_learn:
            winner = candidates[winner_i]
            losers = [c for i, c in enumerate(candidates) if i != winner_i]
            labels = credit_labels(sentence, winner, losers)
            if labels:
                trainer.boundary.learn_labels(
                    sentence, labels, meta={"source": "gold", "formula": gold}
                )
                learned += 1
        if learned:
            logger.info(f"Oracle gold: aprendidas {learned} oraciones de feedback.")
            trainer.save_models("model/n2lpi_model_cache.joblib")


def _print_candidate(idx: int, cand: dict) -> None:
    print(f"\n  {idx}. {cand.get('formula')}")
    props = cand.get("propositions") or {}
    for var, phrase in props.items():
        print(f"       {var} = \"{phrase}\"")


def interactive_loop(trainer: HybridPipelineTrainer):
    """REPL: 2–3 candidates, optional learning from the chosen formula."""
    print("\n" + "#" * 60)
    print("   PIPELINE NL2PROP  —  varias fórmulas + feedback")
    print("   Oración | aprender on/off | salir")
    print("#" * 60 + "\n")

    learn_on = True
    parser = _parser(trainer)

    while True:
        try:
            user_sentence = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSaliendo...")
            break

        if not user_sentence:
            continue
        low = user_sentence.lower()
        if low in ('salir', 'exit', 'q'):
            print("¡Hasta luego!")
            break
        if low in ('aprender on', 'aprender sí', 'aprender si'):
            learn_on = True
            print("Aprendizaje activado.")
            continue
        if low in ('aprender off', 'aprender no'):
            learn_on = False
            print("Aprendizaje desactivado.")
            continue

        try:
            candidates = parser.parse_candidates(user_sentence)
            print("\n" + "-" * 50)
            print(f"Oración: \"{user_sentence}\"")
            for i, cand in enumerate(candidates, 1):
                _print_candidate(i, cand)
            print("-" * 50)

            if not learn_on or trainer.boundary is None:
                continue

            prompt = f"Elige 1-{len(candidates)}, ninguna, o Enter para el 1: "
            try:
                choice = input(prompt).strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                continue

            if choice in ('ninguna', 'n', 'none', '0'):
                print("Sin ganador; no se actualiza el modelo.")
                continue
            if choice in ('', '1'):
                idx = 0
            elif choice.isdigit() and 1 <= int(choice) <= len(candidates):
                idx = int(choice) - 1
            else:
                print("Opción no válida.")
                continue

            winner = candidates[idx]
            losers = [c for i, c in enumerate(candidates) if i != idx]
            labels = credit_labels(user_sentence, winner, losers)
            if labels:
                trainer.boundary.learn_labels(
                    user_sentence, labels, meta={"source": "repl", "formula": winner.get("formula")}
                )
                print(f"Aprendido ({len(labels)} fronteras). Fórmula: {winner.get('formula')}")
            else:
                print("No hubo fronteras que etiquetar.")

        except Exception as e:
            logger.error(f"Excepción no controlada al procesar la oración: {e}")
