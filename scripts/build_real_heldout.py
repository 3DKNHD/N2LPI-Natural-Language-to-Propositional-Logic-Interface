#!/usr/bin/env python3
"""Arma data/real_heldout.json con oraciones publicadas y su formalización.

No inventa frases. Cada ítem copia una oración y la fórmula que ya trae la fuente.
Se dejan fuera las soluciones ilegibles o que contradicen la oración.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from validate_dataset import check_item, formula_symbols, load_items, structural_complexity  # noqa: E402

FY = (
    "Filosofía y Ciudadanía. Lógica proposicional. Ejercicios resueltos. "
    "https://raposa.es/wp-content/uploads/2024/04/ejer_logica_resueltos.pdf"
)
CON = "https://conceptualia.es/ejemplos/ejemplos-de-proposiciones-simples-y-compuestas/"
URJC = (
    "Lógica para el Grado en Ingeniería de la Ciberseguridad. Universidad Rey Juan Carlos. "
    "https://burjcdigital.urjc.es/server/api/core/bitstreams/5c6464c1-a35e-49b0-ad41-3b3596e36c2c/content"
)
PIM = "https://pimath.es/logica-proposicional-20-ejercicios-resueltos-paso-paso"
TUT = "https://tutoriales.com/filosofia/logica/logica-proposicional-construyendo-argumentos-irrefutables-con-conectores-y-tablas"
XUN = (
    "IES Sánchez Cantón. Ejercicios de formalización resueltos. "
    "https://centros.edu.xunta.gal/iessanchezcanton/aulavirtual/pluginfile.php/141227/mod_resource/content/2/Ejercicios%20de%20formalizaci%C3%B3n%20resueltos.pdf"
)

# (oración, fórmula, proposiciones, fuente)
RAW: list[tuple[str, str, dict, str]] = [
    # --- Filosofía y Ciudadanía, bloque 1 ---
    ("No vi la película, pero leí la novela.", "¬P ∧ Q",
     {"P": "vi la película", "Q": "leí la novela"}, FY),
    ("Ni vi la película ni leí la novela.", "¬P ∧ ¬Q",
     {"P": "vi la película", "Q": "leí la novela"}, FY),
    ("No es cierto que viese la película y leyese la novela.", "¬(P ∧ Q)",
     {"P": "vi la película", "Q": "leí la novela"}, FY),
    ("Vi la película aunque no leí la novela.", "P ∧ ¬Q",
     {"P": "vi la película", "Q": "leí la novela"}, FY),
    ("No me gusta trasnochar ni madrugar.", "¬P ∧ ¬Q",
     {"P": "me gusta trasnochar", "Q": "me gusta madrugar"}, FY),
    ("O tú estás equivocado o es falsa la noticia que has leído.", "P ∨ Q",
     {"P": "tú estás equivocado", "Q": "es falsa la noticia que has leído"}, FY),
    ("Si no estuvieras loca, no habrías venido aquí.", "¬P → ¬Q",
     {"P": "estuvieras loca", "Q": "habrías venido aquí"}, FY),
    ("Llueve y o bien nieva o sopla el viento.", "P ∧ (Q ∨ R)",
     {"P": "llueve", "Q": "nieva", "R": "sopla el viento"}, FY),
    ("O está lloviendo y nevando o está soplando el viento.", "(P ∧ Q) ∨ R",
     {"P": "está lloviendo", "Q": "está nevando", "R": "está soplando el viento"}, FY),
    ("Si hay verdadera democracia, entonces no hay detenciones arbitrarias ni otras violaciones de los derechos civiles.",
     "P → (¬Q ∧ ¬R)",
     {"P": "hay verdadera democracia", "Q": "hay detenciones arbitrarias", "R": "hay otras violaciones de los derechos civiles"}, FY),
    ("Roberto hará el doctorado cuando y solamente cuando obtenga la licenciatura.", "P ↔ Q",
     {"P": "roberto hará el doctorado", "Q": "roberto obtenga la licenciatura"}, FY),
    # --- bloque 3 ---
    ("No es cierto que no me guste bailar.", "¬(¬P)",
     {"P": "me gusta bailar"}, FY),
    ("Me gusta bailar y leer libros de ciencia ficción.", "P ∧ Q",
     {"P": "me gusta bailar", "Q": "me gusta leer libros de ciencia ficción"}, FY),
    ("Si los gatos de mi hermana no soltaran tanto pelo, me gustaría acariciarlos.", "¬P → Q",
     {"P": "los gatos de mi hermana sueltan pelo", "Q": "me gusta acariciar los gatos"}, FY),
    ("Si y sólo si viera un marciano con mis propios ojos, creería que hay vida extraterrestre.", "P ↔ Q",
     {"P": "viera un marciano con mis propios ojos", "Q": "creería que hay vida extraterrestre"}, FY),
    ("Una de dos: o salgo a dar un paseo, o me pongo a estudiar como un energúmeno.", "P ∨ Q",
     {"P": "salgo a dar un paseo", "Q": "me pongo a estudiar como un energúmeno"}, FY),
    ("Si los elefantes volaran o supieran tocar el acordeón, pensaría que estoy como una regadera y dejaría que me internaran en un psiquiátrico.",
     "(P ∨ Q) → (R ∧ S)",
     {"P": "los elefantes vuelan", "Q": "los elefantes tocan el acordeón", "R": "estoy como una regadera", "S": "dejo que me internen en un psiquiátrico"}, FY),
    ("Prefiero ir de vacaciones o estar sin hacer nada si tengo tiempo para ello y no tengo que ir a trabajar.",
     "(R ∧ ¬S) → (P ∨ Q)",
     {"P": "prefiero ir de vacaciones", "Q": "prefiero estar sin hacer nada", "R": "tengo tiempo para ello", "S": "tengo que ir a trabajar"}, FY),
    # --- bloque 4 ---
    ("Llueve y hace sol.", "P ∧ Q", {"P": "llueve", "Q": "hace sol"}, FY),
    ("Llueve y no hace sol.", "P ∧ ¬Q", {"P": "llueve", "Q": "hace sol"}, FY),
    ("Llueve o hace sol.", "P ∨ Q", {"P": "llueve", "Q": "hace sol"}, FY),
    ("Si no llueve, hace sol.", "¬P → Q", {"P": "llueve", "Q": "hace sol"}, FY),
    ("No es cierto que llueva.", "¬P", {"P": "llueve"}, FY),
    ("No es cierto que no llueva.", "¬(¬P)", {"P": "llueve"}, FY),
    ("Hará sol si y sólo si no llueve.", "Q ↔ ¬P", {"P": "llueve", "Q": "hará sol"}, FY),
    # --- bloque 5 ---
    ("No es cierto que si llueve y hace sol las brujas se peinan.", "¬((P ∧ Q) → R)",
     {"P": "llueve", "Q": "hace sol", "R": "las brujas se peinan"}, FY),
    ("Cuando las brujas no se peinan, no llueve o no hace sol.", "¬R → (¬P ∨ ¬Q)",
     {"P": "llueve", "Q": "hace sol", "R": "las brujas se peinan"}, FY),
    ("Llueve y las brujas no se peinan, o bien hace sol y las brujas no se peinan.", "(P ∧ ¬R) ∨ (Q ∧ ¬R)",
     {"P": "llueve", "Q": "hace sol", "R": "las brujas se peinan"}, FY),
    # --- bloque 6 ---
    ("Si las estrellas emiten luz, entonces los planetas la reflejan y giran alrededor de ellas.", "P → (Q ∧ R)",
     {"P": "las estrellas emiten luz", "Q": "los planetas reflejan la luz", "R": "los planetas giran alrededor de las estrellas"}, FY),
    ("Las estrellas emiten luz o los planetas la reflejan y, por otra parte, los planetas giran alrededor de ellas.", "(P ∨ Q) ∧ R",
     {"P": "las estrellas emiten luz", "Q": "los planetas reflejan la luz", "R": "los planetas giran alrededor de las estrellas"}, FY),
    ("Los planetas reflejan luz si y sólo si las estrellas la emiten y los planetas giran alrededor de ellas.", "Q ↔ (P ∧ R)",
     {"P": "las estrellas emiten luz", "Q": "los planetas reflejan la luz", "R": "los planetas giran alrededor de las estrellas"}, FY),
    ("Si no es cierto que las estrellas emiten luz y que los planetas la reflejan, entonces éstos no giran alrededor de ellas.",
     "¬(P ∧ Q) → ¬R",
     {"P": "las estrellas emiten luz", "Q": "los planetas reflejan la luz", "R": "los planetas giran alrededor de las estrellas"}, FY),
    # --- bloque 7 ---
    ("Si Pablo no atiende en clase o no estudia en casa, fracasará en los exámenes y no será aplaudido.",
     "(¬P ∨ ¬Q) → (R ∧ ¬S)",
     {"P": "pablo atiende en clase", "Q": "pablo estudia en casa", "R": "pablo fracasa en los exámenes", "S": "pablo es aplaudido"}, FY),
    ("Si no es el caso que Pablo atiende en clase y estudia en casa, entonces fracasará en los exámenes o no será aplaudido.",
     "¬(P ∧ Q) → (R ∨ ¬S)",
     {"P": "pablo atiende en clase", "Q": "pablo estudia en casa", "R": "pablo fracasa en los exámenes", "S": "pablo es aplaudido"}, FY),
    ("Pablo atiende en clase y estudia en casa o, por otra parte, fracasa en los exámenes y no es aplaudido.",
     "(P ∧ Q) ∨ (R ∧ ¬S)",
     {"P": "pablo atiende en clase", "Q": "pablo estudia en casa", "R": "pablo fracasa en los exámenes", "S": "pablo es aplaudido"}, FY),
    # --- bloque 8, citas con fórmula publicada ---
    ("Si escoges tus deseos y tus miedos, no existirá para ti ningún tirano.", "(P ∧ Q) → ¬R",
     {"P": "escoges tus deseos", "Q": "escoges tus miedos", "R": "existirá para ti un tirano"}, FY),
    ("Quien tiene un porqué para vivir puede soportar cualquier cómo.", "P → Q",
     {"P": "tiene un porqué para vivir", "Q": "puede soportar cualquier cómo"}, FY),
    ("El mundo entero es un escenario y todos los humanos somos unos actores.", "P ∧ Q",
     {"P": "el mundo entero es un escenario", "Q": "todos los humanos somos unos actores"}, FY),
    ("Ojos que no ven, corazón que no siente.", "¬P → ¬Q",
     {"P": "los ojos ven", "Q": "el corazón siente"}, FY),
    ("Pienso, luego soy.", "P → Q", {"P": "pienso", "Q": "soy"}, FY),
    ("O estás seguro y lo que dices es cierto, o mientes como un bellaco.", "(P ∧ Q) ∨ R",
     {"P": "estás seguro", "Q": "lo que dices es cierto", "R": "mientes como un bellaco"}, FY),
    # --- bloque 19 ---
    ("Si no hay ruidos y no estás sordo, entonces debes oírme.", "(¬P ∧ ¬Q) → R",
     {"P": "hay ruidos", "Q": "estás sordo", "R": "debes oírme"}, FY),
    ("Iré al cine o al teatro si me invitas.", "R → (P ∨ Q)",
     {"P": "iré al cine", "Q": "iré al teatro", "R": "me invitas"}, FY),
    ("En el caso de que venga Cipriano, vendrán Fulgencia y Eustaquia.", "P → (Q ∧ R)",
     {"P": "viene cipriano", "Q": "vendrá fulgencia", "R": "vendrá eustaquia"}, FY),
    ("Si hay guerra, no crecerá el paro ni la inflación.", "P → (¬Q ∧ ¬R)",
     {"P": "hay guerra", "Q": "crecerá el paro", "R": "crecerá la inflación"}, FY),
    ("Juan debe declarar y ser sincero, o no debe declarar.", "(P ∧ Q) ∨ ¬P",
     {"P": "juan debe declarar", "Q": "juan debe ser sincero"}, FY),
    ("Federico se irá a las Fiji o a las Seychelles si y sólo si le toca la lotería y no se arruina en la ruleta.",
     "(P ∨ Q) ↔ (R ∧ ¬S)",
     {"P": "federico se irá a las fiji", "Q": "federico se irá a las seychelles", "R": "le toca la lotería", "S": "se arruina en la ruleta"}, FY),
    ("El Hombre Lobo es un invento, y si lo mismo ocurre con Papá Noel, entonces los niños son engañados.",
     "P ∧ (Q → R)",
     {"P": "el hombre lobo es un invento", "Q": "papá noel es un invento", "R": "los niños son engañados"}, FY),
    ("Cuando la bolsa baja mucho, es conveniente comprar, y cuando la bolsa sube mucho, también es conveniente comprar.",
     "(P → Q) ∧ (R → Q)",
     {"P": "la bolsa baja mucho", "Q": "es conveniente comprar", "R": "la bolsa sube mucho"}, FY),
    ("Si el aumento de la inflación implica la disminución de la balanza de pagos, entonces, si no disminuye la balanza de pagos, no aumenta la inflación.",
     "(P → Q) → (¬Q → ¬P)",
     {"P": "aumenta la inflación", "Q": "disminuye la balanza de pagos"}, FY),
    ("Dejaré de beber cuando suba el alcohol, pero voy a dejar de fumar tanto si sube el tabaco como si no.",
     "(Q → P) ∧ ((S ∨ ¬S) → R)",
     {"P": "dejaré de beber", "Q": "suba el alcohol", "R": "voy a dejar de fumar", "S": "suba el tabaco"}, FY),
    ("Si no sabes inglés ni francés, te contrato en mi empresa si y sólo si sabes informática o contabilidad.",
     "(¬P ∧ ¬Q) → (R ↔ (S ∨ T))",
     {"P": "sabes inglés", "Q": "sabes francés", "R": "te contrato en mi empresa", "S": "sabes informática", "T": "sabes contabilidad"}, FY),
    # --- premisas con fórmula propia en los bloques 16 y 20 ---
    ("Si acepto este trabajo o dejo de pintar por falta de tiempo, entonces no realizaré mis sueños.",
     "(P ∨ Q) → ¬R",
     {"P": "acepto este trabajo", "Q": "dejo de pintar por falta de tiempo", "R": "realizaré mis sueños"}, FY),
    ("He aceptado el trabajo y he dejado de pintar.", "P ∧ Q",
     {"P": "he aceptado el trabajo", "Q": "he dejado de pintar"}, FY),
    ("Si la tormenta continúa o anochece, nos quedaremos a cenar o a dormir.", "(P ∨ Q) → (R ∨ S)",
     {"P": "la tormenta continúa", "Q": "anochece", "R": "nos quedaremos a cenar", "S": "nos quedaremos a dormir"}, FY),
    ("Si nos quedamos a cenar o a dormir, no iremos mañana al concierto.", "(P ∨ Q) → ¬R",
     {"P": "nos quedamos a cenar", "Q": "nos quedamos a dormir", "R": "iremos mañana al concierto"}, FY),
    ("Iremos mañana al concierto.", "P", {"P": "iremos mañana al concierto"}, FY),
    ("La tormenta no continúa.", "¬P", {"P": "la tormenta continúa"}, FY),
    ("Aprobaré lógica si Dios quiere.", "Q → P",
     {"P": "aprobaré lógica", "Q": "dios quiere que apruebe lógica"}, FY),
    ("Aprobaré lógica si y sólo si estudio y hago todos los ejercicios.", "P ↔ (Q ∧ R)",
     {"P": "aprobaré lógica", "Q": "estudio", "R": "hago todos los ejercicios"}, FY),
    ("No he hecho los ejercicios.", "¬P", {"P": "he hecho los ejercicios"}, FY),
    ("Si el euro está fuerte, el petróleo está barato pero las exportaciones resultan caras.", "P → (Q ∧ R)",
     {"P": "el euro está fuerte", "Q": "el petróleo está barato", "R": "las exportaciones resultan caras"}, FY),
    ("Si Europa se endeuda o la economía no crece, el petróleo no estará barato.", "(P ∨ ¬Q) → ¬R",
     {"P": "europa se endeuda", "Q": "la economía crece", "R": "el petróleo está barato"}, FY),
    ("Habrá inflación a menos que se moderen los precios y los salarios.", "¬(Q ∧ R) → P",
     {"P": "habrá inflación", "Q": "se moderan los precios", "R": "se moderan los salarios"}, FY),
    ("Siempre que se moderan los salarios pero no los precios, si el Gobierno no interviene, el consumo interno disminuye y la economía se ralentiza.",
     "(P ∧ ¬Q) → (¬R → (S ∧ T))",
     {"P": "se moderan los salarios", "Q": "se moderan los precios", "R": "el gobierno interviene", "S": "el consumo interno disminuye", "T": "la economía se ralentiza"}, FY),
    ("Cuando no se moderan los precios, es necesario que el Gobierno intervenga para que la economía no se ralentice.",
     "¬P → (¬R → Q)",
     {"P": "se moderan los precios", "Q": "el gobierno interviene", "R": "la economía se ralentiza"}, FY),
    # --- URJC, solo los ejercicios con solución impresa ---
    ("Sólo si un servidor web es seguro, las comunicaciones que se establecen desde los navegadores garantizan la confidencialidad y también permiten al usuario garantizar la identidad de ese servidor.",
     "(Q ∧ R) → P",
     {"P": "un servidor web es seguro", "Q": "las comunicaciones garantizan la confidencialidad", "R": "las comunicaciones permiten garantizar la identidad de ese servidor"}, URJC),
    ("A menos que sufra un ataque por denegación de servicio, el servidor de la URJC estará operativo el lunes por la mañana.",
     "¬P → Q",
     {"P": "el servidor sufre un ataque por denegación de servicio", "Q": "el servidor de la urjc estará operativo el lunes por la mañana"}, URJC),
    # --- Pimath, ejercicios 3 y 4 ---
    ("Marco estudia y Laura lee.", "P ∧ Q",
     {"P": "marco estudia", "Q": "laura lee"}, PIM),
    ("Si llueve, entonces me quedo en casa.", "P → Q",
     {"P": "llueve", "Q": "me quedo en casa"}, PIM),
    # --- tutoriales.com, ejercicio de simbolización ---
    ("El sol brilla y hace calor.", "P ∧ Q",
     {"P": "el sol brilla", "Q": "hace calor"}, TUT),
    ("Si el sol no brilla, entonces no vamos a la playa.", "¬P → ¬Q",
     {"P": "el sol brilla", "Q": "vamos a la playa"}, TUT),
    ("Hace calor o no vamos a la playa.", "P ∨ ¬Q",
     {"P": "hace calor", "Q": "vamos a la playa"}, TUT),
    ("Vamos a la playa si y solo si el sol brilla y hace calor.", "R ↔ (P ∧ Q)",
     {"P": "el sol brilla", "Q": "hace calor", "R": "vamos a la playa"}, TUT),
    # --- IES Sánchez Cantón: solo las fórmulas que el extracto no contradice ---
    ("Ana es alta.", "P", {"P": "ana es alta"}, XUN),
    ("La manzana es roja o verde.", "P ∨ Q",
     {"P": "la manzana es roja", "Q": "la manzana es verde"}, XUN),
    ("Si llueve, la calle estará mojada.", "P → Q",
     {"P": "llueve", "Q": "la calle estará mojada"}, XUN),
    ("Si hoy es lunes, entonces mañana será martes.", "P → Q",
     {"P": "hoy es lunes", "Q": "mañana será martes"}, XUN),
    ("Si Ana estudia, entonces aprueba.", "P → Q",
     {"P": "ana estudia", "Q": "ana aprueba"}, XUN),
    ("Si llueve, la calle estará mojada y el paraguas será necesario.", "P → (Q ∧ R)",
     {"P": "llueve", "Q": "la calle estará mojada", "R": "el paraguas será necesario"}, XUN),
    ("Si hoy es lunes, entonces mañana será martes o miércoles.", "P → (Q ∨ R)",
     {"P": "hoy es lunes", "Q": "mañana será martes", "R": "mañana será miércoles"}, XUN),
    ("Si Ana estudia, entonces aprueba, y si no estudia, no aprueba.", "(P → Q) ∧ (¬P → ¬Q)",
     {"P": "ana estudia", "Q": "ana aprueba"}, XUN),
    ("Si Ana estudia o hace ejercicio, entonces está en forma.", "(P ∨ Q) → R",
     {"P": "ana estudia", "Q": "ana hace ejercicio", "R": "ana está en forma"}, XUN),
    ("Ana estudia si y solo si María trabaja.", "P ↔ Q",
     {"P": "ana estudia", "Q": "maría trabaja"}, XUN),
    ("Si Ana estudia, entonces aprueba, y si aprueba, su madre está contenta.", "(P → Q) ∧ (Q → R)",
     {"P": "ana estudia", "Q": "ana aprueba", "R": "la madre de ana está contenta"}, XUN),
    ("Si Ana estudia, entonces aprueba, y si no aprueba, su padre está triste.", "(P → Q) ∧ (¬Q → R)",
     {"P": "ana estudia", "Q": "ana aprueba", "R": "el padre de ana está triste"}, XUN),
    ("Si Ana estudia o María trabaja, entonces el examen será fácil.", "(P ∨ Q) → R",
     {"P": "ana estudia", "Q": "maría trabaja", "R": "el examen será fácil"}, XUN),
]


def _xor(p_text: str, q_text: str) -> tuple[str, dict]:
    return "(P ∨ Q) ∧ ¬(P ∧ Q)", {"P": p_text, "Q": q_text}


def conceptualia() -> list[tuple[str, str, dict, str]]:
    rows: list[tuple[str, str, dict, str]] = []

    def add(sentence: str, formula: str, props: dict) -> None:
        rows.append((sentence, formula, props, CON))

    simples = [
        "Madrid es la capital de España.",
        "Lisboa es la capital de Portugal.",
        "París está en Italia.",
        "7 es un número primo.",
        "2 + 2 = 5.",
        "15 es divisible entre 4.",
        "Un triángulo tiene cuatro lados.",
        "El Sol es una estrella.",
        "La Luna es un satélite natural de la Tierra.",
        "Marte tiene dos satélites.",
        "El agua hierve a 100 °C a nivel del mar.",
        "El oxígeno es un gas a temperatura ambiente.",
        "El hierro es un metal.",
        "La ballena azul es un pez.",
        "El esqueleto humano adulto tiene 206 huesos.",
        "El Teide es la montaña más alta de España.",
        "El Ebro desemboca en el Mediterráneo.",
        "Cervantes escribió el Quijote.",
        "Velázquez pintó Las meninas.",
        "La Segunda Guerra Mundial terminó en 1945.",
        "Valencia es una ciudad española.",
    ]
    for sentence in simples:
        atom = sentence[:-1].lower()
        add(sentence, "P", {"P": atom})

    add("Madrid es la capital de España y París es la capital de Francia.", "P ∧ Q",
        {"P": "madrid es la capital de españa", "Q": "parís es la capital de francia"})
    add("4 es par y 9 es primo.", "P ∧ Q", {"P": "4 es par", "Q": "9 es primo"})
    add("El Sol es una estrella y la Luna es un planeta.", "P ∧ Q",
        {"P": "el sol es una estrella", "Q": "la luna es un planeta"})
    add("Mercurio es el planeta más cercano al Sol y Neptuno es el más lejano.", "P ∧ Q",
        {"P": "mercurio es el planeta más cercano al sol", "Q": "neptuno es el planeta más lejano"})
    add("El cobre conduce la electricidad, pero el plástico no.", "P ∧ ¬Q",
        {"P": "el cobre conduce la electricidad", "Q": "el plástico conduce la electricidad"})
    add("3 es menor que 5 y 5 es menor que 2.", "P ∧ Q",
        {"P": "3 es menor que 5", "Q": "5 es menor que 2"})
    add("La primera parte del Quijote se publicó en 1605 y su autor fue Lope de Vega.", "P ∧ Q",
        {"P": "la primera parte del quijote se publicó en 1605", "Q": "su autor fue lope de vega"})
    add("Los delfines son mamíferos, aunque viven en el mar.", "P ∧ Q",
        {"P": "los delfines son mamíferos", "Q": "los delfines viven en el mar"})
    add("7 es primo y 8 es par.", "P ∧ Q", {"P": "7 es primo", "Q": "8 es par"})
    add("Juan estudia y trabaja.", "P ∧ Q", {"P": "juan estudia", "Q": "juan trabaja"})
    add("Juan no estudia o no trabaja.", "¬P ∨ ¬Q", {"P": "juan estudia", "Q": "juan trabaja"})
    add("Ana y Luis son altos.", "P ∧ Q", {"P": "ana es alta", "Q": "luis es alto"})
    add("Ana y Luis son hermanos.", "P", {"P": "ana y luis son hermanos"})

    add("6 es par o 6 es múltiplo de 3.", "P ∨ Q",
        {"P": "6 es par", "Q": "6 es múltiplo de 3"})
    add("Roma está en Grecia o Atenas está en Grecia.", "P ∨ Q",
        {"P": "roma está en grecia", "Q": "atenas está en grecia"})
    add("10 es primo o 10 es impar.", "P ∨ Q", {"P": "10 es primo", "Q": "10 es impar"})
    add("El Amazonas está en África o el Nilo está en Europa.", "P ∨ Q",
        {"P": "el amazonas está en áfrica", "Q": "el nilo está en europa"})
    add("El agua es un elemento químico o el oro es un compuesto.", "P ∨ Q",
        {"P": "el agua es un elemento químico", "Q": "el oro es un compuesto"})
    add("Picasso nació en Málaga o nació en Barcelona.", "P ∨ Q",
        {"P": "picasso nació en málaga", "Q": "picasso nació en barcelona"})
    add("La Tierra gira alrededor del Sol o el Sol gira alrededor de la Tierra.", "P ∨ Q",
        {"P": "la tierra gira alrededor del sol", "Q": "el sol gira alrededor de la tierra"})

    exclusives = [
        ("O bien 13 es par, o bien 13 es impar.", "13 es par", "13 es impar"),
        ("O bien 2 es primo, o bien 2 es par.", "2 es primo", "2 es par"),
        ("O bien Lisboa es la capital de España, o bien lo es Madrid.", "lisboa es la capital de españa", "madrid es la capital de españa"),
        ("O bien España ganó el Mundial de 2010, o bien lo ganó Países Bajos.", "españa ganó el mundial de 2010", "países bajos ganó el mundial de 2010"),
        ("El menú incluye postre o café, no ambos.", "el menú incluye postre", "el menú incluye café"),
    ]
    for sentence, p_text, q_text in exclusives:
        formula, props = _xor(p_text, q_text)
        add(sentence, formula, props)

    add("Si 12 es divisible entre 4, entonces 12 es par.", "P → Q",
        {"P": "12 es divisible entre 4", "Q": "12 es par"})
    add("Si 7 es par, entonces 7 es divisible entre 2.", "P → Q",
        {"P": "7 es par", "Q": "7 es divisible entre 2"})
    add("Si París está en Francia, entonces París está en Europa.", "P → Q",
        {"P": "parís está en francia", "Q": "parís está en europa"})
    add("Si Madrid es la capital de España, entonces 2 + 2 = 5.", "P → Q",
        {"P": "madrid es la capital de españa", "Q": "2 + 2 = 5"})
    add("Si la Luna es de queso, entonces el Sol es una estrella.", "P → Q",
        {"P": "la luna es de queso", "Q": "el sol es una estrella"})
    add("Si llueve, el suelo se moja.", "P → Q", {"P": "llueve", "Q": "el suelo se moja"})
    add("Si el agua pura se enfría por debajo de 0 °C a presión normal, se congela.", "P → Q",
        {"P": "el agua pura se enfría por debajo de 0 °c a presión normal", "Q": "el agua pura se congela"})
    add("Cuando el semáforo está en rojo, los coches deben detenerse.", "P → Q",
        {"P": "el semáforo está en rojo", "Q": "los coches deben detenerse"})
    add("Aprobarás solo si estudias.", "P → Q", {"P": "aprobarás", "Q": "estudias"})
    add("Basta con pulsar el interruptor para que se encienda la luz.", "P → Q",
        {"P": "pulsas el interruptor", "Q": "se enciende la luz"})
    add("Si no estudio, no apruebo.", "¬P → ¬Q", {"P": "estudio", "Q": "apruebo"})
    add("Si llueve, me quedo en casa.", "P → Q", {"P": "llueve", "Q": "me quedo en casa"})

    add("8 es par si y solo si 8 es divisible entre 2.", "P ↔ Q",
        {"P": "8 es par", "Q": "8 es divisible entre 2"})
    add("5 es par si y solo si 5 es divisible entre 2.", "P ↔ Q",
        {"P": "5 es par", "Q": "5 es divisible entre 2"})
    add("Madrid es la capital de España si y solo si París es la capital de Alemania.", "P ↔ Q",
        {"P": "madrid es la capital de españa", "Q": "parís es la capital de alemania"})
    add("Un triángulo es equilátero si y solo si tiene sus tres lados iguales.", "P ↔ Q",
        {"P": "un triángulo es equilátero", "Q": "un triángulo tiene sus tres lados iguales"})
    add("Un ángulo es recto si y solo si mide 90°.", "P ↔ Q",
        {"P": "un ángulo es recto", "Q": "un ángulo mide 90°"})
    add("Un número natural es primo si y solo si tiene exactamente dos divisores.", "P ↔ Q",
        {"P": "un número natural es primo", "Q": "un número natural tiene exactamente dos divisores"})

    add("No es cierto que Madrid sea la capital de Portugal.", "¬P",
        {"P": "madrid es la capital de portugal"})
    add("9 no es un número primo.", "¬P", {"P": "9 es un número primo"})
    add("El Sol no es una estrella.", "¬P", {"P": "el sol es una estrella"})
    add("No es verdad que 3 + 3 = 6.", "¬P", {"P": "3 + 3 = 6"})
    add("Los murciélagos no son aves.", "¬P", {"P": "los murciélagos son aves"})

    add("Si llueve y no llevo paraguas, entonces me mojo.", "(P ∧ ¬Q) → R",
        {"P": "llueve", "Q": "llevo paraguas", "R": "me mojo"})
    add("4 es par, y 5 es primo o 6 es primo.", "P ∧ (Q ∨ R)",
        {"P": "4 es par", "Q": "5 es primo", "R": "6 es primo"})
    add("No es cierto que 3 sea par y 4 sea impar.", "¬(P ∧ Q)",
        {"P": "3 es par", "Q": "4 es impar"})
    add("Si 2 es par, entonces 3 es par o 4 es par.", "P → (Q ∨ R)",
        {"P": "2 es par", "Q": "3 es par", "R": "4 es par"})
    add("Madrid está en España o París está en Italia, y Roma está en Francia.", "(P ∨ Q) ∧ R",
        {"P": "madrid está en españa", "Q": "parís está en italia", "R": "roma está en francia"})
    add("Si estudias y practicas, aprobarás o, al menos, mejorarás.", "(P ∧ Q) → (R ∨ S)",
        {"P": "estudias", "Q": "practicas", "R": "aprobarás", "S": "mejorarás"})
    add("Llueve o no llueve.", "P ∨ ¬P", {"P": "llueve"})
    add("Llueve y no llueve.", "P ∧ ¬P", {"P": "llueve"})
    return rows


def main() -> None:
    train = load_items(ROOT / "data" / "training_dataset.json")
    test = load_items(ROOT / "data" / "test_dataset.json")
    taken = {x["sentence"].strip().lower() for x in train + test}

    items = []
    seen = set()
    dropped = []
    errors = []
    next_id = 8001
    for sentence, formula, props, source in RAW + conceptualia():
        key = sentence.strip().lower()
        if key in taken or key in seen:
            dropped.append(sentence)
            continue
        seen.add(key)
        item = {
            "id": next_id,
            "sentence": sentence if sentence.endswith(".") else sentence + ".",
            "propositions": props,
            "formula": formula,
            "complexity": structural_complexity(formula),
            "connectives": formula_symbols(formula),
            "source": source,
        }
        errors.extend(check_item(item, "real", len(items)))
        items.append(item)
        next_id += 1

    out = ROOT / "data" / "real_heldout.json"
    out.write_text(json.dumps(items, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    from collections import Counter
    print(f"wrote {out} n={len(items)} dropped_overlap={len(dropped)}")
    print("complexity", dict(Counter(x["complexity"] for x in items)))
    if errors:
        print(f"FAIL {len(errors)}")
        for err in errors[:30]:
            print(" -", err)
        raise SystemExit(1)
    print("OK")


if __name__ == "__main__":
    main()
