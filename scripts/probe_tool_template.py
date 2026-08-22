"""Comprueba que el modelo *ve* sus propias tool calls en el historial.

La plantilla de chat que Ollama empaqueta con ``qwen2.5:7b`` (y que
``dolphin3-tools:8b`` heredó) renderiza la rama ``assistant`` con un ``if/else``::

    {{ if .Content }}{{ .Content }}{{- else if .ToolCalls }}<tool_call>...

Como :meth:`llm.client.AssistantMessage.to_openai` manda siempre ``content`` junto
a ``tool_calls``, cualquier turno en el que el modelo narre *mientras* llama a una
herramienta pierde la llamada en el prompt de la inferencia siguiente. La tool se
ejecuta y el log queda bien; lo que se corrompe es el historial que el modelo ve de
sí mismo, y a partir de ahí deja de invocar herramientas (qwen) o se inventa
bloques ``<tool_response>`` (dolphin).

Este probe reproduce ese fallo de forma mínima y determinista: **la misma
conversación, cambiando solo el ``content`` del turno assistant**, preguntándole al
modelo qué función llamó. Con la plantilla rota la segunda variante falla; con los
Modelfiles de ``models/`` las dos pasan.

Resultado medido (temperature 0, seed 0):

    modelo                             content=""      content="Sure, let me ..."
    qwen2.5:7b                  ROTO   get_weather ok  WeatherQuery        FALLA
    dolphin3-tools:8b-pre-...   ROTO   get_weather ok  weather             FALLA
    qwen2.5-tools:7b            FIX    get_weather ok  get_weather         ok
    dolphin3-tools:8b           FIX    get_weather ok  get_weather         ok

Uso:
    python scripts/probe_tool_template.py                    # modelos de config.yaml
    python scripts/probe_tool_template.py qwen2.5:7b         # control: debe FALLAR

Requiere Ollama sirviendo en ``llm.base_url``. Devuelve 0 si todo pasa, 1 si no.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import get_settings  # noqa: E402  (import after sys.path bootstrap)
from llm.client import LLMClient  # noqa: E402

#: Una única herramienta, ajena al dominio del testbed, para que la respuesta
#: correcta no pueda salir del prompt de sistema ni de la memoria del modelo.
WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}

SYSTEM = "You are a helpful assistant with access to tools."
QUESTION = (
    "Which function did you call to obtain that temperature? "
    "Answer with the function name only."
)

#: El texto que el modelo habría narrado junto a la llamada. Es exactamente lo que
#: la rama ``else if`` de la plantilla rota hace desaparecer.
NARRATION = "Sure, let me check that for you."


def _history(assistant_content: str) -> list[dict]:
    """La conversación de prueba, con ``assistant_content`` en el turno assistant.

    Réplica de lo que construye :class:`orchestrator.memory.ShortTermMemory`: turno
    assistant con ``content`` + ``tool_calls`` (formato de
    :meth:`llm.client.AssistantMessage.to_openai`) seguido del mensaje ``tool``
    (formato de :meth:`orchestrator.dispatcher.ToolResult.to_tool_message`).
    """
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "What is the weather in Madrid?"},
        {
            "role": "assistant",
            "content": assistant_content,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "Madrid"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "name": "get_weather",
            "content": '{"ok": true, "result": {"city": "Madrid", "temp_c": 21}}',
        },
        {"role": "user", "content": QUESTION},
    ]


def _client(model: str) -> LLMClient:
    """Un cliente apuntando a ``model``, heredando base_url/timeout de config.yaml."""
    llm = get_settings().llm
    return LLMClient(settings=llm.model_copy(update={"model": model}))


def _asks_which_function(client: LLMClient, assistant_content: str) -> tuple[bool, str]:
    """¿Reconoce el modelo la llamada que hay en su propio historial?

    Se pregunta **sin** pasar ``tools``, y no es un descuido: el esquema de
    herramientas se renderiza en un bloque ``# Tools`` dentro del prompt de sistema,
    así que con él presente el modelo responde ``get_weather`` leyéndolo de ahí, sin
    necesidad de ver su propia llamada — el probe pasaría también sobre la plantilla
    rota (medido). Omitiéndolo, el único sitio del prompt donde aparece el nombre es
    el turno assistant, que es justo lo que la plantilla rota borra.
    """
    reply = client.chat(_history(assistant_content), tools=None,
                        temperature=0, seed=0)
    answer = (reply.content or "").strip()
    return "get_weather" in answer.lower(), answer


def _returns_structured_call(client: LLMClient) -> tuple[bool, str]:
    """¿Sigue Ollama devolviendo ``tool_calls`` estructuradas?

    Ollama deriva el parser de salida de cómo la plantilla renderiza
    ``.ToolCalls``, así que tocarla puede degradar las llamadas a texto suelto. En
    ese caso :func:`llm.client.extract_text_tool_calls` las rescata con un id
    ``text_call_*`` — funciona, pero es el síntoma de que la plantilla nueva rompió
    el parser, y hay que detectarlo aquí y no a mitad de campaña.
    """
    reply = client.chat(
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "What is the weather in Madrid?"},
        ],
        tools=[WEATHER_TOOL],
        temperature=0,
        seed=0,
    )
    if not reply.has_tool_calls:
        return False, f"sin tool_calls (content={reply.content!r})"
    call = reply.tool_calls[0]
    if call.id.startswith("text_call_"):
        return False, f"recuperada del texto por el fallback (id={call.id})"
    return call.name == "get_weather", f"{call.name}(id={call.id})"


def probe(model: str) -> bool:
    """Ejecuta los tres asertos sobre ``model`` e informa por consola."""
    print(f"\n=== {model} ===")
    client = _client(model)
    checks = [
        ("historial con content vacio  ", *_asks_which_function(client, "")),
        ("historial con content + call ", *_asks_which_function(client, NARRATION)),
        ("tool_calls estructuradas     ", *_returns_structured_call(client)),
    ]
    for label, ok, detail in checks:
        print(f"  [{'ok ' if ok else 'FALLA'}] {label} -> {detail}")
    return all(ok for _, ok, _ in checks)


def main(argv: list[str]) -> int:
    models = argv[1:] or get_settings().models
    results = {m: probe(m) for m in models}
    print()
    for model, ok in results.items():
        print(f"{'PASA ' if ok else 'FALLA'} {model}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
