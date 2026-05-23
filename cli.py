#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CLI entry points for the N2LPI translator:
  - run_demonstration: shows a single example with full AST output
  - test_external_dataset: evaluates against a JSON test file
  - interactive_loop: REPL for real-time sentence-to-formula translation
"""

import json
from ml_pipeline import HybridPipelineTrainer
from logic_utils import son_formulas_equivalentes
from config import logger
from hybrid_parser import HybridLogicParser

confidence : float = 0.7

def run_demonstration(trainer: HybridPipelineTrainer):
    """Show a single demonstration: sentence → formula with proposition grounding."""
    test_example = trainer.df.iloc[-1]
    sentence = test_example['sentence']
    true_formula = test_example['formula']

    main_conn, sub_flags = trainer.predict(sentence)
    
    parser = HybridLogicParser(trainer, confidence_threshold=confidence)
    result = parser.parse(sentence)
    
    generated_formula = result['formula']

    print("\n" + "=" * 60)
    print("DEMOSTRACIÓN DE LA ARQUITECTURA HÍBRIDA (AST SEMÁNTICO)")
    print("=" * 60)
    print(f"Oración original: {sentence}")
    print(f"Conector principal predicho: {main_conn}")
    
    print("\nProposiciones Extraídas (Grounding):")
    for var, phrase in result['propositions'].items():
        print(f"  {var} = \"{phrase}\"")

    print(f"\nFórmula lógica generada: {generated_formula}")
    print(f"Fórmula verdadera (GT): {true_formula}")
    print(f"Coincidencia estructural: {son_formulas_equivalentes(generated_formula, true_formula)}")
    print("=" * 60)


def test_external_dataset(trainer: HybridPipelineTrainer, test_path: str) -> None:
    """Evaluate the hybrid pipeline against an external JSON dataset and report structural accuracy."""
    try:
        with open(test_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Error cargando el dataset externo: {e}")
        return

    print(f"\n{'='*60}\nEVALUACIÓN SOBRE DATASET EXTERNO: {test_path}\n{'='*60}")
    total = 0
    correct_structure = 0
    
    for item in test_data:
        sentence = item.get('sentence', '')
        if not sentence: continue
        true_formula = item.get('formula', None)
        
        main_conn, sub_flags = trainer.predict(sentence)
        
        parser = HybridLogicParser(trainer, confidence_threshold=confidence)
        result = parser.parse(sentence)
        generated_formula = result['formula']

        total += 1
        eq = False
        if true_formula:
            eq = son_formulas_equivalentes(generated_formula, true_formula)
            if eq:
                correct_structure += 1

        print(f"\nOración: {sentence}")
        print(f"  Fórmula generada: {generated_formula}")
        if true_formula:
            print(f"  Coincidencia estructural: {'Sí' if eq else 'No'}")

    if total > 0 and test_data[0].get('formula'):
        print(f"\n{'='*60}\nPrecisión estructural global: {correct_structure / total:.3f} ({correct_structure}/{total})\n{'='*60}")


def interactive_loop(trainer: HybridPipelineTrainer):
    """REPL: read Spanish sentences, print logical formulas and proposition grounding."""
    print("\n" + "#" * 60)
    print("   INTERFAZ SEMÁNTICA - TRADUCTOR NL2PROP")
    print("   Escribe tu oración. Para salir escribe: salir / exit / q")
    print("#" * 60 + "\n")

    while True:
        try:
            user_sentence = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSaliendo...")
            break

        if not user_sentence: continue
        if user_sentence.lower() in ('salir', 'exit', 'q'):
            print("¡Hasta luego!")
            break

        try:
            main_conn, sub_flags = trainer.predict(user_sentence)
            
            parser = HybridLogicParser(trainer, confidence_threshold=confidence)
            result = parser.parse(user_sentence)

            print("\n" + "-" * 50)
            print(f"Oración Ingresada: \"{user_sentence}\"")
            print(f"Conector ML: {main_conn}")
            
            print("\nProposiciones Atómicas (Grounding):")
            if not result['propositions']:
                print("  (No se detectaron proposiciones válidas)")
            else:
                for var, phrase in result['propositions'].items():
                    print(f"  {var} = \"{phrase}\"")

            print(f"\nFórmula Lógica Generada: {result['formula']}")
            print("-" * 50 + "\n")

        except Exception as e:
            logger.error(f"Excepción no controlada al procesar la oración: {e}")