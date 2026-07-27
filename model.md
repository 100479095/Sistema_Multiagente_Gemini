# Creación del modelo `dolphin3-tools:8b`

Este documento explica, paso a paso, cómo se construyó el modelo **`dolphin3-tools:8b`**
que usa el brazo "no alineado" (*unaligned*) del experimento. **No es un modelo del
registro de Ollama**: se crea en local a partir de `dolphin3:8b` y no se puede
descargar con `ollama pull`. Aquí queda toda la receta para poder reproducirlo.

---

## 1. El problema

El experimento lanza al modelo asistente una petición con `tools` (function calling)
en **cada** inferencia. Además, la inyección del ataque se entrega **a través del
resultado de una herramienta** (el asistente tiene que llamar a una tool para leer el
correo/calendario envenenado). Por tanto, un modelo que no soporte herramientas no
puede ejecutar el experimento en absoluto.

Al usar Dolphin, Ollama devolvía un error 400:

```
openai.BadRequestError: Error code: 400 -
{'error': {'message': 'registry.ollama.ai/library/dolphin3:8b does not support tools',
           'type': 'invalid_request_error', ...}}
```

Esto tiraba abajo toda la campaña (crasheó en la celda 111/220, la primera de Dolphin).

**Causa raíz:** tanto `dolphin-llama3:8b` como `dolphin3:8b` se publican en Ollama con
una plantilla de chat que **no tiene sección de tool-calls**. Ollama solo acepta
peticiones con `tools` si la plantilla del modelo referencia `.Tools`/`.ToolCalls`; si
no, rechaza la petición con un 400. No es un problema de los pesos del modelo, es un
problema de su plantilla.

---

## 2. Diagnóstico: cómo se detecta si un modelo soporta tools

Ollama declara las capacidades de cada modelo local. Basta con mirar el bloque
**Capabilities**:

```bash
ollama show qwen2.5:7b     # → Capabilities: completion, tools   ✅ soporta tools
ollama show dolphin3:8b    # → Capabilities: completion          ❌ NO soporta tools
```

Salida real durante el diagnóstico:

```
=== qwen2.5:7b (funciona) ===
  Capabilities
    completion
    tools

=== dolphin3:8b (falla) ===
  Capabilities
    completion
```

> **Método rápido para verificar cualquier candidato:** `ollama pull <modelo>` y luego
> `ollama show <modelo>`. Si en *Capabilities* aparece `tools`, sirve; si solo aparece
> `completion`, no. No hace falta lanzar una inferencia completa.

---

## 3. La idea clave

Al inspeccionar `dolphin3:8b` vi sus *stop tokens*:

```
  Parameters
    stop    "<|im_start|>"
    stop    "<|im_end|>"
```

Esos tokens (`<|im_start|>` / `<|im_end|>`) son el formato **ChatML** — exactamente el
mismo que usa `qwen2.5:7b`, que **sí** soporta tools. Es decir, los pesos de Dolphin
esperan el mismo formato de conversación que Qwen.

**Conclusión:** se puede coger la plantilla con tool-calls de `qwen2.5` e injertarla
sobre los pesos (no alineados) de `dolphin3`. Los pesos —el "cerebro" del modelo— no
cambian; solo cambia cómo se formatea el prompt y cómo se renderizan las llamadas a
herramientas. Por eso sigue siendo válido como brazo *unaligned* del experimento.

---

## 4. Procedimiento paso a paso

### Paso 1 — Confirmar el problema (opcional pero recomendable)

Prueba real contra el endpoint OpenAI de Ollama con un `tools` payload:

```bash
curl -s -w '\nHTTP_STATUS:%{http_code}\n' http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "dolphin3:8b",
    "messages": [{"role": "user", "content": "Open the living room window."}],
    "tools": [{"type": "function", "function": {"name": "open_window",
      "description": "Open a window in a room",
      "parameters": {"type": "object", "properties": {"room": {"type": "string"}},
      "required": ["room"]}}}],
    "tool_choice": "auto", "stream": false
  }'
```

Resultado esperado: `HTTP_STATUS:400` con `does not support tools`.

### Paso 2 — Extraer la plantilla de `qwen2.5:7b`

```bash
ollama show qwen2.5:7b --modelfile
```

De la salida interesa el bloque `TEMPLATE """..."""`, que es una plantilla ChatML con
los bloques `{{ if .Tools }}` (declara las herramientas al modelo) y
`{{ range .ToolCalls }}` (renderiza las llamadas que emite el modelo).

### Paso 3 — Escribir el `Modelfile`

Se crea un `Modelfile` que **reutiliza los pesos de Dolphin** (`FROM dolphin3:8b`) y les
pega la plantilla de Qwen más los *stop tokens* de ChatML:

```dockerfile
FROM dolphin3:8b

TEMPLATE """{{- if .Messages }}
{{- if or .System .Tools }}<|im_start|>system
{{- if .System }}
{{ .System }}
{{- end }}
{{- if .Tools }}

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{{- range .Tools }}
{"type": "function", "function": {{ .Function }}}
{{- end }}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
{{- end }}<|im_end|>
{{ end }}
{{- range $i, $_ := .Messages }}
{{- $last := eq (len (slice $.Messages $i)) 1 -}}
{{- if eq .Role "user" }}<|im_start|>user
{{ .Content }}<|im_end|>
{{ else if eq .Role "assistant" }}<|im_start|>assistant
{{ if .Content }}{{ .Content }}
{{- else if .ToolCalls }}<tool_call>
{{ range .ToolCalls }}{"name": "{{ .Function.Name }}", "arguments": {{ .Function.Arguments }}}
{{ end }}</tool_call>
{{- end }}{{ if not $last }}<|im_end|>
{{ end }}
{{- else if eq .Role "tool" }}<|im_start|>user
<tool_response>
{{ .Content }}
</tool_response><|im_end|>
{{ end }}
{{- if and (ne .Role "assistant") $last }}<|im_start|>assistant
{{ end }}
{{- end }}
{{- else }}
{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ end }}{{ .Response }}{{ if .Response }}<|im_end|>{{ end }}"""

PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
```

### Paso 4 — Crear el modelo

```bash
ollama create dolphin3-tools:8b -f Modelfile
```

En la salida aparece `using existing layer sha256:...` para las capas de pesos: **no se
descarga nada nuevo**, se reutilizan los 4.9 GB de `dolphin3:8b` y solo se añade la capa
de plantilla. Es casi instantáneo.

### Paso 5 — Verificar que ya declara `tools`

```bash
ollama show dolphin3-tools:8b
```

Ahora debe aparecer:

```
  Capabilities
    completion
    tools          ← ✅
```

### Paso 6 — Prueba de humo real (end to end)

```bash
curl -s -w '\nHTTP_STATUS:%{http_code}\n' http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "dolphin3-tools:8b",
    "messages": [{"role": "user", "content": "Open the living room window please."}],
    "tools": [{"type": "function", "function": {"name": "open_window",
      "description": "Open a window in a room",
      "parameters": {"type": "object", "properties": {"room": {"type": "string"}},
      "required": ["room"]}}}],
    "tool_choice": "auto", "stream": false
  }'
```

Resultado obtenido: `HTTP_STATUS:200` y una tool call estructurada:

```json
{
  "role": "assistant",
  "content": "",
  "tool_calls": [
    {
      "id": "call_mpel12c8",
      "type": "function",
      "function": {
        "name": "open_window",
        "arguments": "{\"room\":\"living room\"}"
      }
    }
  ]
}
```

### Paso 7 — Apuntar el experimento al nuevo modelo

Los modelos del barrido se leen de **`config.yaml`** (fuente única de verdad; el
`models:` de `experiment_config.yaml` es solo un fallback que el runner sobrescribe
al cargar):

```yaml
models: ["qwen2.5:7b", "dolphin3-tools:8b"]   # main assistant LLM (swept)
```

Y relanzar (con `--resume` se saltan las celdas de qwen ya completadas):

```bash
python scripts/run_experiment.py --console --summary
```

---

## 5. Reproducir desde cero (resumen)

Requisito previo: tener `qwen2.5:7b` en local (de donde sale la plantilla).

```bash
# 1. Descargar los pesos base de Dolphin
ollama pull dolphin3:8b

# 2. Crear el Modelfile del Paso 3 (ver arriba)

# 3. Construir el modelo con tools
ollama create dolphin3-tools:8b -f Modelfile

# 4. Verificar
ollama show dolphin3-tools:8b   # Capabilities debe incluir "tools"
```

---

## 6. Notas para el TFG

- **Solo cambia la plantilla, no los pesos.** El comportamiento (alineación, tendencia a
  rechazar o no) proviene de los pesos de `dolphin3:8b`, que son idénticos. La plantilla
  solo define el formato del prompt y de las tool-calls.
- **Comparación más limpia:** al usar la misma plantilla ChatML que `qwen2.5:7b`, ambos
  modelos comparten *exactamente* el mismo formato de function calling. Así la única
  variable entre los dos brazos es la alineación del modelo, no el formato de las tools.
- **El modelo es local.** Persiste entre ejecuciones y reinicios, pero **solo en esta
  máquina**. En otro equipo hay que volver a ejecutar el `ollama create`. Por eso esta
  receta (y/o el `Modelfile`) debe versionarse en el repositorio.
- **Recuperación tras el fallback de texto:** el cliente ya sabe recuperar tool-calls
  emitidas como texto (`extract_text_tool_calls` en `src/llm/client.py`), un apaño para
  la conocida rareza Qwen/Ollama. Esto da robustez extra si el modelo alguna vez emite la
  llamada como texto en lugar de en el campo `tool_calls`.

---

## 7. Eliminar el modelo (si hiciera falta)

```bash
ollama rm dolphin3-tools:8b
```
