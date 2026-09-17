#!/usr/bin/env python3
"""Stress test: ~100 real-style Spanish sentences, mixed registers."""
from __future__ import annotations

import re
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

warnings.filterwarnings("ignore")

from hybrid_parser import HybridLogicParser
from logic_utils import son_formulas_equivalentes
from ml_pipeline import HybridPipelineTrainer

# (family, tag, sentence, gold)
# Gold uses P,Q,R… in appearance order of atoms the *intended* parse would name.
CASES = [
    # —— IFF / math / definitions (Wikimedia, UCM, ITAM) ——
    ("iff", "wiki-matriz",
     "Una matriz con coeficientes reales es invertible si y solo si su determinante es distinto de cero.",
     "P ↔ Q"),
    ("iff", "wiki-par",
     "Un entero es par si y solo si es divisible por dos.",
     "P ↔ Q"),
    ("iff", "iff-solamente",
     "Un triángulo es equilátero si solamente si sus tres lados son iguales.",
     "P ↔ Q"),
    ("iff", "iff-cuando",
     "Una función es continua cuando y solo cuando el límite existe en cada punto.",
     "P ↔ Q"),
    ("iff", "iff-si-solo",
     "El grafo es conexo si solo si existe un camino entre cualquier par de vértices.",
     "P ↔ Q"),
    ("iff", "iff-list",
     "Se aprueba la tesis si y solo si el director firma, no hay plagio y el tribunal vota a favor.",
     "P ↔ (Q ∧ ¬R ∧ S)"),
    ("iff", "nec-suf-par",
     "Que un entero sea múltiplo de cuatro es necesario y suficiente para que sus dos últimos dígitos formen un múltiplo de cuatro.",
     "P ↔ Q"),
    ("iff", "equivale",
     "Aprobar el curso equivale a entregar el proyecto y no copiar.",
     "P ↔ (Q ∧ ¬R)"),
    ("iff", "siempre-y-cuando",
     "Lo haré yo siempre y cuando los demás estén de acuerdo.",
     "P ↔ Q"),
    ("iff", "syc-ticket",
     "Me iré el martes siempre y cuando encuentre billete en el avión.",
     "P ↔ Q"),
    # —— si / cuando / entonces cotidianos (Burbuja, Victoria Monera) ——
    ("si", "si-estudias",
     "Si estudias español, apruebas el examen.",
     "P → Q"),
    ("si", "si-novio",
     "Si mi novio me invita a cenar, iremos a un restaurante italiano.",
     "P → Q"),
    ("si", "si-rojo",
     "Si el semáforo está en rojo y no hay una ambulancia cruzando, entonces el conductor se detiene.",
     "(P ∧ ¬Q) → R"),
    ("si", "cuando-alarma",
     "La alarma suena cuando se abre la puerta o se rompe el cristal.",
     "(Q ∨ R) → P"),
    ("si", "cuando-rio",
     "Cuando el río suena, agua lleva.",
     "P → Q"),
    ("si", "si-subjuntivo",
     "Si estudiaras español, aprobarías el examen.",
     "P → Q"),
    ("si", "si-no-reserva",
     "Si no tienes reserva o el restaurante está lleno, no podrás cenar.",
     "(¬P ∨ Q) → ¬R"),
    ("si", "o-si",
     "Puedes entrar si tienes pase y tu foto coincide, o si eres personal de seguridad.",
     "((P ∧ Q) ∨ R) → S"),
    ("si", "si-entonces-si",
     "Si configuras Nginx y usas TLS, entonces la web carga rápido si no hay un ataque.",
     "(P ∧ Q) → (¬R → S)"),
    ("si", "comma-or",
     "Si el vuelo se retrasa o se cancela, y no contrataste el seguro, entonces pierdes el hotel.",
     "((P ∨ Q) ∧ ¬R) → S"),
    ("si", "wrap-iff",
     "Si el cluster responde y el disco no está lleno, entonces el servicio está sano si y solo si el balanceador enruta el tráfico.",
     "(P ∧ ¬Q) → (R ↔ S)"),
    # —— UNLESS familia (RAE, ZonaELE, Burbuja, tráfico) ——
    ("unless", "a-menos-trafico",
     "El vehículo se detiene en el rojo a menos que transporte un enfermo grave o lo persiga un vehículo policial.",
     "¬(Q ∨ R) → P"),
    ("unless", "a-menos-imprevisto",
     "No vendré a menos que pase un imprevisto.",
     "¬Q → ¬P"),
    ("unless", "a-no-ser-pida",
     "No iré a no ser que él me lo pida.",
     "¬Q → ¬P"),
    ("unless", "a-no-ser-huelga",
     "El tren sale a las seis a no ser que haya huelga o la vía esté cortada.",
     "¬(Q ∨ R) → P"),
    ("unless", "salvo-llueva",
     "Hoy vamos a ir al parque salvo que llueva.",
     "¬Q → P"),
    ("unless", "salvo-si",
     "Hoy vamos a ir al parque salvo si llueve.",
     "¬Q → P"),
    ("unless", "excepto-que",
     "Vamos a votar excepto que tú digas lo contrario.",
     "¬Q → P"),
    ("unless", "excepto-si",
     "Vamos a votar excepto si tú dices lo contrario.",
     "¬Q → P"),
    ("unless", "salvo-arbitro",
     "El partido continúa salvo que el árbitro suspenda el juego o se corte la luz.",
     "¬(Q ∨ R) → P"),
    ("unless", "excepto-revisor",
     "El periódico sale excepto si la rotativa se avería.",
     "¬Q → P"),
    ("unless", "gala-imprevisto",
     "A menos que haya algún imprevisto, volveré a casa después.",
     "¬P → Q"),
    ("unless", "medico",
     "El paciente recibe el alta a menos que la fiebre persista y no coma.",
     "¬(Q ∧ ¬R) → P"),
    # —— otros condicionales ELE ——
    ("nexo", "siempre-que-taza",
     "Te presto mi taza siempre que tengas cuidado con ella.",
     "Q → P"),
    ("nexo", "siempre-que-tarea",
     "Siempre que el estudiante entregue la tarea y no copie, recibirá la nota completa.",
     "(P ∧ ¬Q) → R"),
    ("nexo", "con-tal-libro",
     "Te presto mi libro de español con tal de que me lo devuelvas pronto.",
     "Q → P"),
    ("nexo", "con-tal-que",
     "Haré lo que tú quieras con tal que me dejes tranquila.",
     "Q → P"),
    ("nexo", "con-tal-reunion",
     "Iré a la reunión con tal que me avisen con tiempo y no coincida con el vuelo.",
     "(Q ∧ ¬R) → P"),
    ("nexo", "condicion-cafe",
     "Te acompaño hasta tu casa a condición de que me invites a un café.",
     "Q → P"),
    ("nexo", "condicion-secreto",
     "Te lo explicaré a condición de que no se lo cuentes a nadie.",
     "¬Q → P"),
    ("nexo", "en-caso-salga",
     "En caso de que salga antes de trabajar, iré a tu casa.",
     "P → Q"),
    ("nexo", "en-caso-servidor",
     "En caso de que se corte el gas, cierra la llave general.",
     "P → Q"),
    ("nexo", "en-caso-llueva",
     "En caso de que falle el freno y no tengas arcén, reduce la marcha.",
     "(P ∧ ¬Q) → R"),
    ("nexo", "mientras-metas",
     "Mientras no te metas con nadie, puedes hacer lo que quieras.",
     "¬P → Q"),
    ("nexo", "mientras-token",
     "La radio emite mientras la licencia no expire.",
     "¬Q → P"),
    ("nexo", "en-cuanto",
     "En cuanto aterrice el vuelo, recoge el equipaje.",
     "P → Q"),
    ("nexo", "una-vez",
     "Una vez que el horno alcanza ciento ochenta grados, introduce el bizcocho.",
     "P → Q"),
    ("nexo", "basta",
     "Basta con que el testigo declare para que se reabra el caso.",
     "P → Q"),
    # —— concesivas / adversativas ——
    ("conc", "aunque-rae",
     "Aunque el crédito me lo nieguen o me lo recorten, ampliaré la casa.",
     "(P ∨ Q) ∧ R"),
    ("conc", "aunque-llueva",
     "Aunque llueva, saldré a caminar.",
     "P ∧ Q"),
    ("conc", "aun-si",
     "Aun si protestan los vecinos, el concierto sigue.",
     "P ∧ Q"),
    ("conc", "incluso-si",
     "Incluso si falta harina, preparamos la masa.",
     "P ∧ Q"),
    ("conc", "a-pesar",
     "A pesar de que el banco niegue el crédito, compro el local.",
     "P ∧ Q"),
    ("conc", "sin-embargo",
     "El ensayo salió limpio, sin embargo el motor vibra.",
     "P ∧ Q"),
    ("conc", "no-obstante",
     "El puente es nuevo, no obstante el peso máximo es bajo.",
     "P ∧ Q"),
    ("conc", "pero-no",
     "El build compiló, pero las pruebas fallaron y el artefact no se publicó.",
     "P ∧ Q ∧ ¬R"),
    # —— XOR / ni / no solo ——
    ("coord", "xor-tren",
     "O bien tomas el tren o bien tomas el metro, no ambos.",
     "(P ∨ Q) ∧ ¬(P ∧ Q)"),
    ("coord", "ni-ni",
     "Ni el servidor responde ni la base de datos arranca y el técnico no llega.",
     "¬P ∧ ¬Q ∧ ¬R"),
    ("coord", "ni-fuma",
     "Ni fuma ni bebe.",
     "¬P ∧ ¬Q"),
    ("coord", "no-solo-sino",
     "No solo compiló el módulo sino superó las pruebas y documentó la API.",
     "P ∧ Q ∧ R"),
    ("coord", "tanto-como",
     "Tanto el firewall bloquea el puerto como el proxy descarta la petición.",
     "P ∧ Q"),
    ("coord", "ya-sea",
     "Ya sea que cenes en casa o cenes fuera, avisa a tu hermana.",
     "(P ∨ Q) → R"),
    ("coord", "tanto-si",
     "Tanto si hay huelga como si hay atasco, llego tarde.",
     "(P ∨ Q) → R"),
    # —— negación de bloque ——
    ("neg", "no-cierto-mariscos",
     "No es cierto que si ingieres mariscos crudos y no los refrigeras, enfermarás o irás a urgencias.",
     "¬((P ∧ ¬Q) → (R ∨ S))"),
    ("neg", "es-falso-juez",
     "Es falso que el juez reciba el soborno y archive la causa o libere al imputado.",
     "¬((P ∧ Q) ∨ R)"),
    ("neg", "no-verdad-clima",
     "No es verdad que llueva y truene, o que no haga frío y nieve.",
     "¬((P ∧ Q) ∨ (¬R ∧ S))"),
    ("neg", "no-ocurre-pago",
     "No ocurre que el pago se confirme si la pasarela está caída o si el banco rechaza el cargo y no hay saldo.",
     "¬((Q ∨ (R ∧ ¬S)) → P)"),
    ("neg", "jamas-archivo",
     "Jamás el archivo se publica si el revisor no aprueba el parche o la prueba falla.",
     "¬((¬Q ∨ R) → P)"),
    ("neg", "nunca-contenedor",
     "Nunca el barco zarpa si hay temporal o falla el motor.",
     "¬((Q ∨ R) → P)"),
    ("neg", "es-mentira",
     "Es mentira que el testigo vio el choque y calló.",
     "¬(P ∧ Q)"),
    ("neg", "no-posible",
     "No es posible que el avión despegue y la pista esté cerrada.",
     "¬(P ∧ Q)"),
    # —— causal / consecutivo ——
    ("caus", "puesto-que",
     "Puesto que el río creció, cierran la carretera.",
     "P → Q"),
    ("caus", "dado-que",
     "Dado que no hay testigos, el juez archiva la causa.",
     "¬P → Q"),
    ("caus", "ya-que",
     "Ya que el testigo no comparece, se aplaza la vista.",
     "¬P → Q"),
    ("caus", "por-lo-tanto",
     "El termostato marca ochenta, por lo tanto el ventilador arranca.",
     "P → Q"),
    ("caus", "por-consiguiente",
     "El pasaporte caducó, por consiguiente no embarcas.",
     "P → ¬Q"),
    ("caus", "de-lo-contrario",
     "Firma el acta, de lo contrario se anula el acuerdo.",
     "¬P → Q"),
    ("caus", "si-no",
     "La boda se aplaza si no escampa.",
     "¬Q → P"),
    ("caus", "solo-si-rae",
     "Solo si me invitan formalmente, asistiré a la ceremonia.",
     "Q → P"),
    # —— anidados / kitchen ——
    ("nest", "ana-larga",
     "Ana aprueba el curso si solo si asiste los lunes, entrega el proyecto y no copia, o si el tribunal le concede una revisión.",
     "P ↔ ((Q ∧ R ∧ ¬S) ∨ T)"),
    ("nest", "list-iff-neg",
     "El sistema entra en mantenimiento si y solo si no hay usuarios conectados, la cola está vacía y el operador confirma.",
     "P ↔ (¬Q ∧ R ∧ S)"),
    ("nest", "legal-si-no",
     "Si el paciente demuestra la necesidad y no existe un tratamiento alternativo, entonces el médico firma el informe.",
     "(P ∧ ¬Q) → R"),
    ("nest", "devops-and-or",
     "Si el pod está vivo o el nodo responde, y no hay una alerta crítica, entonces el despliegue continúa.",
     "((P ∨ Q) ∧ ¬R) → S"),
    ("nest", "tienda",
     "La tienda abre si es laborable y no es festivo, o si el dueño autoriza el turno extra.",
     "((P ∧ ¬Q) ∨ R) → S"),
    ("nest", "medico-and",
     "Si tienes fiebre y no has tomado el antibiótico, entonces llamas al médico o vas a urgencias.",
     "(P ∧ ¬Q) → (R ∨ S)"),
    ("nest", "banco",
     "La transferencia se ejecuta si el saldo alcanza y el destinatario no está bloqueado, o si un cajero aprueba la excepción.",
     "((P ∧ ¬Q) ∨ R) → S"),
    ("nest", "clima-pack",
     "Si nieva o hiela, y no llevas cadenas, entonces no subes al puerto.",
     "((P ∨ Q) ∧ ¬R) → ¬S"),
    # —— OOD duro: ELE avanzado / habla real poco vista ——
    ("ood", "por-si-paraguas",
     "Llévate el paraguas por si llueve.",
     "Q → P"),
    ("ood", "por-si-chaqueta",
     "Puse una chaqueta en la maleta por si hacía fresco.",
     "Q → P"),
    ("ood", "como-amenaza",
     "Como no termines los deberes, no vas a salir a la calle.",
     "¬P → ¬Q"),
    ("ood", "como-beber",
     "Como sigas bebiendo así, no te dejaré conducir.",
     "P → ¬Q"),
    ("ood", "de-ser",
     "De ser eso cierto, lo habría publicado la prensa.",
     "P → Q"),
    ("ood", "a-poco",
     "A poco que te esfuerces, aprobarás.",
     "P → Q"),
    ("ood", "no-sea-que",
     "Habla bajo no sea que te oigan los vecinos.",
     "¬Q → P"),
    ("ood", "si-bien",
     "Si bien el contrato está firmado, la obra no arranca.",
     "P ∧ ¬Q"),
    ("ood", "pese-a-que",
     "Pese a que el juez recusó al perito, el juicio sigue.",
     "P ∧ Q"),
    ("ood", "supuesto",
     "En el supuesto de que alguien llamara, di que me he mudado de piso.",
     "P → Q"),
    ("ood", "a-fin-de",
     "Apaga el móvil a fin de que la reunión no se interrumpa.",
     "P → ¬Q"),
    ("ood", "asi-que",
     "El tren se retrasó, así que perdí la conexión.",
     "P → Q"),
    ("ood", "cuanto-mas",
     "Cuanto más estudias, mejor escribes.",
     "P → Q"),
    ("ood", "con-que",
     "Con que haya medio kilo de harina, es suficiente para hacer este pastel.",
     "P → Q"),
]


def ops(formula: str):
    return re.findall(r"[∧∨→↔¬]", formula or "")


def main_op(s: str):
    s = (s or "").strip()
    while s.startswith("(") and s.endswith(")"):
        depth = 0
        covers = True
        for i, ch in enumerate(s):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0 and i != len(s) - 1:
                    covers = False
                    break
        if covers:
            s = s[1:-1].strip()
        else:
            break
    for c in "↔→∨∧¬":
        if c in s:
            return c
    return "?"


def grade(got: str, gold: str) -> str:
    if son_formulas_equivalentes(got, gold):
        return "OK"
    g, h = ops(gold), ops(got)
    if g == h:
        return "MISMO_ESQUELETO"
    if main_op(got) == main_op(gold) and abs(len(g) - len(h)) <= 1:
        return "CERCA"
    return "FAIL"


def main():
    trainer = HybridPipelineTrainer("data/training_dataset.json")
    trainer.load_models("model/n2lpi_model_cache.joblib")
    parser = HybridLogicParser(trainer)

    rows = []
    for fam, tag, sent, gold in CASES:
        result = parser.parse(sent)
        got = result["formula"]
        mark = grade(got, gold)
        cues = [(c.text, c.kind) for c in trainer.predict_cues(sent)]
        rows.append((mark, fam, tag, gold, got, cues, result.get("propositions"), sent))

    print(f"{len(CASES)} casos")
    print("RESUMEN", dict(Counter(r[0] for r in rows)))
    print("por familia:")
    by = defaultdict(Counter)
    for mark, fam, *_ in rows:
        by[fam][mark] += 1
    for fam in ("iff", "si", "unless", "nexo", "conc", "coord", "neg", "caus", "nest", "ood"):
        print(f"  {fam:7} {dict(by[fam])}  n={sum(by[fam].values())}")

    order = {"OK": 0, "MISMO_ESQUELETO": 1, "CERCA": 2, "FAIL": 3}
    print("\n=== no OK ===")
    for mark, fam, tag, gold, got, cues, props, sent in sorted(rows, key=lambda x: (order[x[0]], x[1], x[2])):
        if mark == "OK":
            continue
        print(f"[{mark}] {fam}/{tag}")
        print(f"  {sent}")
        print(f"  gold {gold}")
        print(f"  got  {got}")
        print(f"  cues {cues}")
        print(f"  props {props}")
        print()


if __name__ == "__main__":
    main()
