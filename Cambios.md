# Cambios.md — Reorientación del TFG

> Documento de especificación para Claude Code. Léelo entero antes de tocar
> nada. Está pensado para complementar a `CLAUDE.md`, no para sustituirlo:
> el flujo descrito allí (orquestador, doble inferencia, agentes mock,
> guardrails, scorer, BD) sigue siendo el corazón del sistema. Este
> documento dice **qué se añade encima**, **qué se modifica** y **qué se
> conserva tal cual**.

---

## 1. Cambio de panorama (contexto)

### 1.1. De qué iba el TFG hasta ahora

Reproducción local de los *Targeted Promptware Attacks* del paper de Nassi
et al. (2025) contra un sistema multi-agente tipo Gemini, con una
contribución original: un **bucle adaptativo** donde un LLM atacante
local (`qwen2.5:7b`) refina iterativamente el payload hasta vencer al LLM
víctima (`llama3.1:8b`).

### 1.2. De qué va ahora

La hipótesis central del TFG cambia. La pregunta de investigación pasa a
ser:

> **¿Aumenta la tasa de éxito de un prompt-injection cuando el payload
> viaja envuelto en información posterior al *knowledge cutoff* del
> modelo víctima?**

La intuición: un modelo "viejo" (Llama 2, cutoff sept 2022 según pretraining
y jul 2023 incluyendo tuning data, según la Model Card oficial de Meta)
no puede contrastar la información post-cutoff contra conocimiento previo.
Eso lo deja en modo "confiar en el contexto", lo que probablemente **baja
su resistencia a inyecciones** que vengan envueltas en ese contenido
desconocido.

### 1.3. Qué se mantiene del trabajo previo

- El orquestador con doble inferencia (`simulation/orchestrator.py`).
- Todos los agentes mock (`MockGoogleCalendarAgent`, `MockGmailAgent`,
  `MockGoogleHomeAgent`, `MockUtilitiesAgent`).
- Los guardrails (I/O Validation, CFI).
- Las memorias short/long-term.
- Los prompts iniciales del paper en `attacker/initial_prompts.py`.
- El `AttackScorer` por clase de amenaza.
- El `RequestCatcher` y la `ExperimentDB`.
- **El bucle adaptativo se mantiene como una condición experimental
  adicional**, no se descarta.

### 1.4. Qué cambia conceptualmente

- Se añade un nuevo **factor experimental**: la *estrategia de wrapping*
  con información post-cutoff (`none`, `prefix`, `interleaved`, `authority`).
- Se añade un nuevo **factor experimental**: el modelo víctima
  (`llama2:7b` viejo vs. `llama3.1:8b` moderno como control).
- El bucle adaptativo se reduce a un **factor más** (`mode = adaptive` vs.
  `mode = static`), no es ya el centro narrativo.
- Se incorpora una métrica secundaria nueva: detección de confabulación
  sobre el hecho post-cutoff embebido (proxy léxico).

### 1.5. Diseño experimental resultante

Diseño factorial de hasta cuatro factores:

| Factor          | Niveles                                      |
|-----------------|----------------------------------------------|
| `victim_model`  | `llama2:7b`, `llama3.1:8b`                   |
| `wrap_strategy` | `none`, `prefix`, `interleaved`, `authority` |
| `mode`          | `static`, `adaptive`                         |
| `threat_class`  | T2, T5, T6, T7, T10, T13 (los seis del TFG)  |

Repeticiones por celda: 5 (configurable). Total ~240 ejecuciones.

La **predicción interesante** es la interacción `victim_model × wrap_strategy`:
esperamos que el efecto del wrapping sea **mayor en `llama2:7b`** que en
`llama3.1:8b`. Si esa interacción aparece, hay hallazgo.

---

## 2. Riesgos metodológicos a vigilar (importante para Claude Code)

Antes de implementar nada, ten estos confounds en mente. La discusión del
TFG depende de ellos.

1. **Confound de longitud**. El wrapper alarga el payload. Necesitamos un
   control: hechos **pre-cutoff** envueltos con la misma estructura. Si el
   efecto solo aparece con post-cutoff, está aislado. Esto está marcado
   como **TODO opcional** más abajo y se puede añadir tras la primera
   ronda de resultados.

2. **Confound de "Llama 2 es simplemente peor"**. Si `llama2:7b` cae más
   en *todas* las condiciones, incluido `wrap_strategy=none`, no estamos
   midiendo el wrapper sino la calidad general del modelo. El análisis
   debe enfatizar la **interacción**, no el efecto principal del modelo.

3. **Validación previa del corpus**. Si el modelo víctima ya conoce los
   "hechos post-cutoff", la hipótesis no se está testeando. Por eso
   `experiments/validate_cutoff.py` es un paso **obligatorio** antes de
   correr el factorial.

---

## 3. Archivos nuevos a crear

Los archivos ya están redactados en `outputs/promptware_extension/` (ver
artefactos adjuntos a este documento). Cópialos al repo respetando rutas.
Resumen rápido:

| Ruta destino                                       | Propósito                                       |
|----------------------------------------------------|-------------------------------------------------|
| `attacker/post_cutoff_corpus.py`                   | 16 hechos curados, todos > 2023-07-20           |
| `attacker/hallucination_wrapper.py`                | 3 estrategias: prefix, interleaved, authority   |
| `attacker/hallucination_detector.py`               | Detector léxico de confabulación                |
| `attacker/static_runner.py`                        | Runner sin LLM atacante (modo `static`)         |
| `experiments/validate_cutoff.py`                   | Probe de desconocimiento del modelo             |
| `experiments/run_factorial.py`                     | Sweep completo de la matriz factorial           |
| `analysis/report.py`                               | Tablas y figuras para el TFG                    |
| `storage/migrations/001_add_wrapper_columns.sql`   | Columnas nuevas en SQLite                       |
| `tests/test_wrapper.py`                            | Tests de los componentes nuevos                 |

> **Nota**: los archivos provistos asumen los nombres de módulos y clases
> tal como están descritos en `CLAUDE.md`. Si en el repo real los imports
> difieren (por ejemplo si `AdaptiveAttackLoop` vive en otro path),
> ajústalos al integrar.

---

## 4. Archivos existentes a modificar

### 4.1. `attacker/adaptive_loop.py` — patch obligatorio

El bucle adaptativo necesita aceptar un wrapper opcional. Cambios:

#### 4.1.1. Imports

```python
from typing import Optional
from .hallucination_wrapper import HallucinationWrapper, WrapResult
```

#### 4.1.2. Constructor

Añadir parámetro `wrapper` y atributo de metadata:

```python
def __init__(
    self,
    threat_class: str,
    guardrails: bool,
    victim_model: Optional[str] = None,
    attacker_model: Optional[str] = None,
    max_iterations: int = 40,
    wrapper: Optional[HallucinationWrapper] = None,    # NEW
    db_path: str = "data/results.db",
):
    ...
    self.wrapper = wrapper                              # NEW
    self.last_wrap_meta: Optional[WrapResult] = None    # NEW
```

#### 4.1.3. `_inject_payload`

El wrapping ocurre **justo antes** de inyectar, y solo cuando hay wrapper:

```python
def _inject_payload(self, prompt: str) -> None:
    if self.wrapper is not None:
        wrap = self.wrapper.wrap(prompt)
        self.last_wrap_meta = wrap
        text_to_inject = wrap.wrapped_text
    else:
        self.last_wrap_meta = None
        text_to_inject = prompt

    if self.threat_class == "T14_worm_email":
        self.orchestrator.gmail.add_poisoned_email(text_to_inject)
    else:
        self.orchestrator.calendar.add_poisoned_event(text_to_inject)
```

#### 4.1.4. Logging por iteración

Cuando construyas `iteration_data`, añade:

```python
iteration_data["wrap_strategy"] = (
    self.last_wrap_meta.strategy if self.last_wrap_meta else None
)
iteration_data["wrap_fact_id"] = (
    self.last_wrap_meta.fact_id if self.last_wrap_meta else None
)
iteration_data["payload_injected"] = (
    self.last_wrap_meta.wrapped_text if self.last_wrap_meta
    else self.current_prompt
)
```

#### 4.1.5. ⚠️ Decisión clave sobre el `PromptImprover`

El `PromptImprover` (qwen2.5:7b) debe seguir recibiendo **el payload base**
(`self.current_prompt`), NO el texto envuelto. El wrapping se reaplica en
cada iteración como variable controlada — si el atacante mutara el
wrapping, perderíamos la pureza del factor.

En la práctica esto ya pasa si tu código actual hace lo siguiente (que es
lo que describe `Flujo_del_programa.md`):

```python
self.current_prompt = self.improver.improve(
    objective=self.threat_class,
    previous_prompt=self.current_prompt,   # ← payload base, no wrapped
    ...
)
```

**Importante**: NO pases `self.last_wrap_meta.wrapped_text` al improver.
Si el código actual ya pasa `self.current_prompt`, no hay nada que hacer
aquí — el cambio del punto 4.1.3 es suficiente.

### 4.2. `experiments/run_experiment.py` — añadir flags

Añadir estos argumentos al `argparse`:

```python
parser.add_argument(
    "--wrapper",
    choices=["none", "prefix", "interleaved", "authority"],
    default="none",
    help="Wrapping strategy for the injected payload",
)
parser.add_argument(
    "--victim-model",
    default=None,
    help="Override the victim Ollama model tag (e.g. llama2:7b)",
)
parser.add_argument(
    "--mode",
    choices=["static", "adaptive"],
    default="adaptive",
    help="Static = fixed payload, no PromptImprover. Adaptive = current loop.",
)
parser.add_argument(
    "--repetitions",
    type=int,
    default=5,
    help="Number of independent samples (used in static mode).",
)
```

Y en el dispatch:

```python
from attacker.hallucination_wrapper import build_wrapper

wrapper = build_wrapper(args.wrapper)

if args.mode == "static":
    from attacker.static_runner import StaticAttackRun
    runner = StaticAttackRun(
        threat_class=args.threat,
        guardrails=args.guardrails == "on",
        victim_model=args.victim_model,
        wrapper=wrapper,
        repetitions=args.repetitions,
    )
else:
    runner = AdaptiveAttackLoop(
        threat_class=args.threat,
        guardrails=args.guardrails == "on",
        victim_model=args.victim_model,
        wrapper=wrapper,
        max_iterations=args.iterations,
    )
result = runner.run()
```

### 4.3. `storage/database.py` — `ExperimentDB`

Tres trabajos:

1. **Aplicar la migración SQL** `001_add_wrapper_columns.sql` al
   `data/results.db` existente. Si prefieres no migrar, recrea la BD —
   está en `.gitignore` y no hay datos críticos.

2. **Aceptar los nuevos parámetros en `start_experiment`**:

   ```python
   def start_experiment(
       self,
       threat_class: str,
       guardrails: bool,
       mode: str = "adaptive",
       victim_model: str | None = None,
       wrap_strategy: str = "none",
   ) -> int:
       ...
   ```

3. **Persistir las nuevas columnas en `log_iteration`**: `wrap_strategy`,
   `wrap_fact_id`, `payload_injected`, `hallucination_observed`. Si el
   dict pasado no las trae, usar `None` / `0`.

### 4.4. `attacker/scorer.py` — opcional pero recomendado

Añadir el campo `hallucination_observed` al dict que devuelve, calculado
con `attacker.hallucination_detector.hallucination_observed(...)`.

Para hacerlo, el `score(...)` necesita acceso al texto del fact embebido y
a los mensajes de usuario. Hay dos formas de cablearlo:

- **Opción A (mínima)**: que el orquestador o el runner guarden
  `last_user_messages` y `last_wrap_fact_text` y los pasen al scorer.
- **Opción B (más limpia)**: calcular `hallucination_observed` en el
  runner (no en el scorer) y mergearlo al dict del log antes de escribir.
  Esto evita acoplar el scorer al wrapper.

Recomiendo **Opción B**.

---

## 5. Plan de implementación (orden estricto)

> Implementa estas fases en orden. Cada una termina con una verificación.

### Fase 1 — Andamiaje no destructivo
1. Crear los archivos nuevos del punto 3 (todos copiables tal cual).
2. Aplicar migración SQL a `data/results.db` (o borrar el .db, da igual).
3. `python -m pytest tests/test_wrapper.py -v` debe pasar al 100%
   sin Ollama.

### Fase 2 — Integración en el loop existente
4. Aplicar el patch del punto 4.1 a `attacker/adaptive_loop.py`.
5. Añadir flags del punto 4.2 a `experiments/run_experiment.py`.
6. Ampliar `ExperimentDB` (punto 4.3).
7. **Smoke test sin Ollama**: `python -c "from attacker.adaptive_loop import AdaptiveAttackLoop"`
   debe importar sin errores.

### Fase 3 — Validación contra modelos reales
8. `ollama pull llama2:7b`.
9. **Validación de cutoff** (paso obligatorio):
   ```bash
   python -m experiments.validate_cutoff --victim-model llama2:7b
   python -m experiments.validate_cutoff --victim-model llama3.1:8b
   ```
   Inspeccionar `data/cutoff_validation_*.json`. Si Llama 2 conoce >30%
   del corpus, hay que filtrar el corpus a solo los hechos
   `judged_ignorant=true`. Para esto: crear `attacker/post_cutoff_corpus.py`
   con dos listas (`POST_CUTOFF_FACTS_RAW` y `POST_CUTOFF_FACTS` filtrada
   en runtime leyendo el JSON de validación).

10. **Smoke test del experimento**:
    ```bash
    python -m experiments.run_experiment \
        --threat T7_activate_boiler \
        --guardrails off \
        --mode static \
        --wrapper interleaved \
        --victim-model llama2:7b \
        --repetitions 1
    ```
    Verificar que el row aparece en `data/results.db` con
    `wrap_strategy='interleaved'` y `payload_injected` lleno.

### Fase 4 — Run completo del factorial
11. Smoke pequeño:
    ```bash
    python -m experiments.run_factorial \
        --victims llama2:7b \
        --threats T7_activate_boiler \
        --modes static \
        --wrappers none interleaved \
        --repetitions 1
    ```
12. Run completo (~210–240 celdas, 6–10 horas):
    ```bash
    python -m capture.request_catcher &     # en otra terminal
    python -m experiments.run_factorial \
        --victims llama2:7b llama3.1:8b \
        --threats T2_spamming T5_delete_events T6_open_window \
                  T7_activate_boiler T10_geolocation T13_exfiltrate_calendar \
        --modes static adaptive \
        --wrappers none prefix interleaved authority \
        --repetitions 5
    ```

### Fase 5 — Análisis
13. `python -m analysis.report` produce tablas CSV y figuras PNG en
    `results/`. Estos son los que entran al TFG.

---

## 6. Checklist final para Claude Code

Antes de declarar la migración terminada, verificar:

- [ ] Los 10 archivos nuevos del punto 3 existen en sus rutas correctas.
- [ ] `attacker/adaptive_loop.py` acepta `wrapper=` en `__init__`.
- [ ] `_inject_payload` envuelve el prompt si hay wrapper.
- [ ] `iteration_data` incluye `wrap_strategy`, `wrap_fact_id`,
      `payload_injected`.
- [ ] El `PromptImprover` sigue recibiendo el payload base, NO el envuelto.
- [ ] `experiments/run_experiment.py` tiene los flags `--wrapper`,
      `--victim-model`, `--mode`, `--repetitions`.
- [ ] `ExperimentDB.start_experiment(...)` y `log_iteration(...)` aceptan
      las columnas nuevas.
- [ ] La migración SQL está aplicada (o el `.db` recreado).
- [ ] `pytest tests/test_wrapper.py` pasa al 100%.
- [ ] El smoke test de Fase 3 paso 10 escribe un row con `wrap_strategy`
      bien poblado.

---

## 7. Lo que se queda fuera de esta iteración (TODOs explícitos)

Para no inflar el alcance, estos puntos quedan diferidos:

- **TODO:** corpus de hechos pre-cutoff (control para confound de
  longitud). Útil después de la primera ronda de resultados, si el revisor
  presiona en ese flanco.
- **TODO:** detector de confabulación basado en LLM-judge (más fiable que
  el heurístico actual, pero requiere otra ronda de Ollama).
- **TODO:** análisis estadístico formal (test de interacción 2-way ANOVA
  o equivalente no paramétrico) — el `analysis/report.py` actual genera
  tablas y gráficas, pero no calcula significancia.

Estos no son bloqueantes para entregar el TFG. Si Claude Code tiene tiempo
sobrante tras la checklist del punto 6, puede atacarlos en este orden.

---

## 8. Resumen ejecutivo en una frase

Se añade un wrapper que envuelve los payloads de prompt-injection del
paper dentro de hechos post-cutoff verificables, se añade Llama 2 7B como
modelo víctima "viejo", se mantiene el bucle adaptativo como una condición
más, y se sweep-ea todo en una matriz factorial cuyo resultado clave es la
**interacción modelo × wrapper** — la firma de la hipótesis de
"alucinación inducida facilita la inyección".
