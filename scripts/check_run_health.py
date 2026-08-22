"""Audita los logs JSONL en busca de instrumentacion rota, no de resultados malos.

Un run puede terminar con ``success=0`` porque el modelo aguanto el ataque (dato
valido) o porque el banco se rompio (dato basura). Este script separa lo segundo:
lee ``logs/*.jsonl`` y cuenta dos patologias, ambas observadas en la campana de
agosto de 2026 y ambas causadas por la plantilla de chat que Ollama empaqueta con
qwen2.5 (ver models/Modelfile.qwen2.5-tools):

1. **final_answer con <tool_response>** — la plantilla renderizaba la rama
   ``assistant`` con ``{{ if .Content }}...{{ else if .ToolCalls }}``, asi que un
   turno con texto Y llamada perdia la llamada en el prompt siguiente. Los modelos
   que no aprendieron ChatML (dolphin) veian entonces que tras su prosa aparece un
   bloque ``<tool_response>`` y lo escribian ellos, a veces vacio, a veces
   inventandose el resultado. El run termina ahi y el juez lo lee como una negativa.

2. **llamadas rescatadas del texto** — ids ``text_call_*``: Ollama no parseo la
   tool call y la dejo como texto en ``content``; la salvo
   :func:`llm.client.extract_text_tool_calls`. Funciona, asi que no rompe la
   campana, pero indica que la plantilla ya no le sirve a Ollama para deducir el
   prefijo ``<tool_call>`` — degradacion silenciosa a corregir en el Modelfile.

**El borrado en si no es auditable desde el log**, y conviene tenerlo claro: la
plantilla se aplica dentro de Ollama, mientras que ``messages_sent`` guarda lo que
el testbed *envio*, con sus ``tool_calls`` intactas. Lo unico observable es el
patron que lo disparaba — un turno assistant con texto **y** llamada — que se
informa aparte, sin contarlo como patologia: con la plantilla corregida es el caso
que ahora se renderiza bien, y su frecuencia (~48 % de los runs de la campana
archivada) es justo la medida de cuanto dano hacia antes.

Uso:
    python scripts/check_run_health.py                     # logs/
    python scripts/check_run_health.py logs/archive-pre-templatefix

Devuelve 0 si no hay patologias, 1 si hay alguna.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Report:
    """Recuento de runs afectados por cada patologia en un directorio."""

    directory: Path
    runs: int = 0
    tool_response_answer: list[str] = field(default_factory=list)
    text_fallback: list[str] = field(default_factory=list)
    text_plus_call: list[str] = field(default_factory=list)
    unreadable: list[str] = field(default_factory=list)

    @property
    def healthy(self) -> bool:
        return not (self.tool_response_answer or self.text_fallback)


def _records(path: Path) -> Iterator[dict[str, Any]]:
    """Los registros de un log, saltando lineas truncadas (runs que petaron)."""
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def scan(directory: Path) -> Report:
    """Recorre los ``*.jsonl`` de ``directory`` y agrega las patologias."""
    report = Report(directory=directory)
    for path in sorted(directory.glob("*.jsonl")):
        report.runs += 1
        name = path.name
        inferences: list[dict[str, Any]] = []
        final_answer = ""
        ended = False
        for record in _records(path):
            event = record.get("event")
            if event == "inference":
                inferences.append(record)
            elif event == "run_end":
                ended = True
                final_answer = str(record.get("final_answer") or "")
        if not ended:
            # El run murio a mitad (excepcion, Ctrl-C): no es una patologia de
            # plantilla, pero tampoco es un run puntuable, asi que se lista aparte.
            report.unreadable.append(name)
        if "<tool_response>" in final_answer:
            report.tool_response_answer.append(name)
        for inf in inferences:
            calls = inf.get("tool_calls") or []
            if calls and (inf.get("response_text") or "").strip():
                report.text_plus_call.append(name)
                break
        for inf in inferences:
            if any(str(c.get("id", "")).startswith("text_call_") for c in inf.get("tool_calls") or []):
                report.text_fallback.append(name)
                break
    return report


def _line(label: str, hits: list[str], total: int, *, pathology: bool = True) -> str:
    pct = (100 * len(hits) / total) if total else 0.0
    mark = ("FALLA" if hits else "ok   ") if pathology else "info "
    sample = f"  p.ej. {hits[0]}" if hits else ""
    return f"  [{mark}] {label:38s} {len(hits):5d}/{total} ({pct:5.1f} %){sample}"


def report_to_text(report: Report) -> str:
    lines = [f"\n=== {report.directory} ===", f"  runs analizados: {report.runs}"]
    if report.runs:
        lines += [
            _line("final_answer con <tool_response>", report.tool_response_answer, report.runs),
            _line("llamadas rescatadas del texto", report.text_fallback, report.runs),
            _line("turnos con texto + llamada", report.text_plus_call, report.runs,
                  pathology=False),
            _line("runs sin run_end (petaron)", report.unreadable, report.runs,
                  pathology=False),
        ]
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    dirs = [Path(a) for a in argv[1:]] or [REPO_ROOT / "logs"]
    reports = []
    for directory in dirs:
        if not directory.is_dir():
            print(f"[aviso] no es un directorio: {directory}")
            continue
        report = scan(directory)
        reports.append(report)
        print(report_to_text(report))
    if not reports:
        return 1
    print()
    for report in reports:
        print(f"{'SANO ' if report.healthy else 'ROTO '} {report.directory}")
    return 0 if all(r.healthy for r in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
