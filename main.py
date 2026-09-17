#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
N2LPI entry point.

Workflow:
  1. Load cached models from disk (or train if no cache exists)
  2. If argv[1] is provided, evaluate the external dataset at that path
  3. Otherwise, run a demonstration followed by the interactive REPL
"""

import sys
from ml_pipeline import HybridPipelineTrainer
from cli import run_demonstration, test_external_dataset, interactive_loop
from config import logger

def main():
    """Load/train models, then run demo + REPL or external evaluation."""
    logger.info("Iniciando sistema traductor N2LPI...")
    
    trainer = HybridPipelineTrainer(data_path='data/training_dataset.json')
    model_path = 'model/n2lpi_model_cache.joblib'

    if trainer.load_models(model_path):
        logger.info("Modelos cacheados cargados.")
        tagger_ready = bool(getattr(trainer.tagger, "_ready", False))
        if trainer.boundary is None or not tagger_ready:
            logger.info("Falta etiquetador de piezas/conectores; entrenando...")
            trainer.train_boundary_classifier()
            trainer.save_models(model_path)
    else:
        logger.info("No se encontró caché válida. Iniciando entrenamiento...")
        trainer.train_sub_connectors()
        trainer.train_main_connector()
        trainer.cross_validate_main()
        trainer.train_boundary_classifier()
        trainer.save_models(model_path)

    if len(sys.argv) > 1:
        test_path = sys.argv[1]
        logger.info(f"Iniciando evaluación externa con: {test_path}")
        test_external_dataset(trainer, test_path)
        sys.exit(0)

    run_demonstration(trainer)

    interactive_loop(trainer)

if __name__ == '__main__':
    main()