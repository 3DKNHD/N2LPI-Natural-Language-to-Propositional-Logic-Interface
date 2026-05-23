#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import time
import requests
import os
import pandas as pd

from ml_pipeline import HybridPipelineTrainer
from hybrid_parser import HybridLogicParser
from logic_utils import son_formulas_equivalentes

# Ollama API endpoint for local LLM inference
OLLAMA_API_URL = "http://localhost:11434/api/generate"
# Change to "llama3.2:3b" or the 3B/1B model you have downloaded in Ollama
LLM_MODEL_NAME = "qwen2.5:3b"


def consultar_ollama(oracion: str) -> str:
    """
    Send a Spanish sentence to the local Ollama LLM and request a classic propositional logic formula.
    The prompt enforces strict syntax: uppercase variables (P, Q, R, S) and specific logical connectors.
    """
    prompt = f"""Traduce la siguiente oración en español a una fórmula de lógica proposicional clásica.
Usa estrictamente las variables proposicionales necesarias en mayúsculas (P, Q, R, S) y ÚNICAMENTE los siguientes conectores lógicos exactos:
- Conjunción: ∧
- Disyunción: ∨
- Implicación: →
- Equivalencia: ↔
- Negación: ¬

Oración: "{oracion}"

Respuesta (Escribe ÚNICAMENTE la fórmula resultante, sin explicaciones, sin texto adicional, sin bloques de código Markdown ni comillas):"""

    try:
        payload = {
            "model": LLM_MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "top_k": 1,
                "top_p": 0.1
            }
        }
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=15)
        if response.status_code == 200:
            return response.json().get("response", "").strip()
    except Exception as e:
        return f"ERROR_LLM: {str(e)}"
    return "ERROR_LLM"


def evaluar_sistema():
    """Main evaluation routine: compares hybrid pipeline vs LLM on NL2PROP task."""
    print("=" * 70)
    print("   INICIANDO EVALUACIÓN COMPETITIVA DE MÉTRICAS (NL2PROP)")
    print("=" * 70)

    dataset_path = 'data/test_dataset.json'
    if not os.path.exists(dataset_path):
        print(f"Error: No se encontró el archivo '{dataset_path}' en el directorio.")
        return

    with open(dataset_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    total_instancias = len(test_data)
    print(f" -> Dataset de prueba cargado correctamente. Total de instancias: {total_instancias}\n")

    print(" -> Cargando e Inicializando el Modelo Híbrido Local...")
    trainer = HybridPipelineTrainer(data_path='data/training_dataset.json')
    model_cache_path = 'model/n2lpi_model_cache.joblib'

    # Try to load pre-trained models from cache
    if trainer.load_models(model_cache_path):
        print("    [OK] Modelos cargados desde la caché serializada.")
    else:
        print("    [!] No se encontró caché. Entrenando pipeline local rápidamente...")
        trainer.train_sub_connectors()
        trainer.train_main_connector()
        trainer.save_models(model_cache_path)

    # Initialize hybrid parser with confidence threshold
    parser_hibrido = HybridLogicParser(trainer, confidence_threshold=0.7)

    aciertos_hibrido = 0  # hybrid correct predictions
    aciertos_llm = 0      # LLM correct predictions

    tiempo_total_hibrido = 0.0
    tiempo_total_llm = 0.0

    registros_log = []    # store detailed results for CSV export

    print(f"\n -> Evaluando contra el modelo de Ollama '{LLM_MODEL_NAME}' en tiempo real...")
    print("-" * 70)

    for idx, item in enumerate(test_data):
        oracion = item['sentence']
        formula_verdadera = item['formula']

        # --- Hybrid pipeline prediction ---
        t0_hib = time.time()
        try:
            resultado_hib = parser_hibrido.parse(oracion)
            formula_hib = resultado_hib['formula']
        except Exception:
            formula_hib = "ERROR_PARSING"
        t1_hib = time.time()
        duracion_hib = t1_hib - t0_hib
        tiempo_total_hibrido += duracion_hib

        es_correcto_hib = son_formulas_equivalentes(formula_hib, formula_verdadera)
        if es_correcto_hib:
            aciertos_hibrido += 1

        # --- LLM prediction ---
        t0_llm = time.time()
        formula_llm = consultar_ollama(oracion)
        t1_llm = time.time()
        duracion_llm = t1_llm - t0_llm
        tiempo_total_llm += duracion_llm

        es_correcto_llm = son_formulas_equivalentes(formula_llm, formula_verdadera)
        if es_correcto_llm:
            aciertos_llm += 1

        # Progress report every 5 instances (print in Spanish)
        if (idx + 1) % 5 == 0 or (idx + 1) == total_instancias:
            print(f" Procesados: {idx + 1}/{total_instancias} | "
                  f"Precisión Parcial Híbrido: {(aciertos_hibrido/(idx+1))*100:.1f}% | "
                  f"LLM: {(aciertos_llm/(idx+1))*100:.1f}%")

        # Log detailed record for this instance
        registros_log.append({
            "id": item.get("id", idx),
            "oracion": oracion,
            "gt_formula": formula_verdadera,
            "hib_formula": formula_hib,
            "hib_ok": es_correcto_hib,
            "hib_time_ms": duracion_hib * 1000,
            "llm_formula": formula_llm,
            "llm_ok": es_correcto_llm,
            "llm_time_ms": duracion_llm * 1000
        })

    # Calculate final accuracy and latency metrics
    accuracy_hibrido = (aciertos_hibrido / total_instancias) * 100
    accuracy_llm = (aciertos_llm / total_instancias) * 100

    latencia_promedio_hib_ms = (tiempo_total_hibrido / total_instancias) * 1000
    latencia_promedio_llm_ms = (tiempo_total_llm / total_instancias) * 1000

    print("\n" + "=" * 70)
    print("                     MÉTRICAS FINALIZADAS")
    print("=" * 70)
    print(f" Métrica Evaluada         | Enfoque Híbrido (Local) | LLM ({LLM_MODEL_NAME})")
    print("-" * 70)
    print(f" Exactitud Lógica (ESM)   | {accuracy_hibrido:.2f}%{' ' * (22-len(f'{accuracy_hibrido:.2f}%'))} | {accuracy_llm:.2f}%")
    print(f" Latencia Promedio/Muestra| {latencia_promedio_hib_ms:.1f} ms{' ' * (22-len(f'{latencia_promedio_hib_ms:.1f} ms'))} | {latencia_promedio_llm_ms:.1f} ms")
    print(f" Eficiencia Relativa      | El Híbrido es {latencia_promedio_llm_ms/max(0.001, latencia_promedio_hib_ms):.1f} veces más rápido en CPU.")
    print("=" * 70)

    # Save detailed results to CSV for further analysis
    df_reporte = pd.DataFrame(registros_log)
    df_reporte.to_csv("reporte_competicion_jce.csv", index=False, encoding='utf-8')
    print("\n[Éxito] Resumen detallado guardado en 'reporte_competicion_jce.csv' para análisis adicional.")


if __name__ == "__main__":
    evaluar_sistema()