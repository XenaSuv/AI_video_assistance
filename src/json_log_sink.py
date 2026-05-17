"""JSONL sink for loguru — one flat JSON object per log record.

Compatible with Datadog, Grafana Loki, CloudWatch, and any aggregator
that parses newline-delimited JSON (no extra configuration required).

Usage (already wired in pipeline_helpers._setup_logging):
    from src.json_log_sink import JsonSink
    logger.add(JsonSink(run_dir / "run.jsonl"), level="DEBUG", filter=scrub)

Output schema — one line per record:
    {
        "ts":       "2026-05-17T10:23:45.678+00:00",  # ISO-8601, ms precision
        "level":    "INFO",
        "msg":      "Pipeline complete! Cost: $0.021",
        "module":   "pipeline_orchestrator",
        "function": "_finalize",
        "line":     874,
        "extra":    {"pipeline": "daily"},             # only when logger.bind() used
        "exc":      "Traceback (most recent call ..."  # only when exception logged
    }

Log aggregator hints:
    Datadog:       set `dd_source=python`, map `msg`→`message`, `ts`→`@timestamp`
    Grafana Loki:  use promtail with json pipeline stage; `level` and `ts` auto-detected
    CloudWatch:    enable structured logging on the log group; no mapping needed
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

_MAX_BYTES = 10 * 1024 * 1024  # 10 MB — matches text log rotation ceiling


class JsonSink:
    """Thread-safe JSONL sink with size-based rotation.

    Pass an instance directly to loguru's ``logger.add()``::

        logger.add(JsonSink(path), level="DEBUG", filter=scrub)

    When the file exceeds *max_bytes* it is renamed to ``<stem>.1.jsonl``
    (overwriting any previous backup) and a fresh file is started.
    """

    def __init__(self, path: Path | str, max_bytes: int = _MAX_BYTES) -> None:
        self._path = Path(path)
        self._max_bytes = max_bytes
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # loguru callable-sink protocol
    # ------------------------------------------------------------------

    def __call__(self, message: Any) -> None:
        record = message.record
        entry: dict[str, Any] = {
            "ts":       record["time"].isoformat(timespec="milliseconds"),
            "level":    record["level"].name,
            "msg":      record["message"],
            "module":   record["module"],
            "function": record["function"],
            "line":     record["line"],
        }
        if record["extra"]:
            entry["extra"] = {k: _safe(v) for k, v in record["extra"].items()}
        if record["exception"] is not None:
            import traceback as _tb
            entry["exc"] = "".join(_tb.format_exception(*record["exception"]))

        line = json.dumps(entry, ensure_ascii=False, default=str) + "\n"

        with self._lock:
            self._maybe_rotate()
            with open(self._path, "a", encoding="utf-8") as fh:
                fh.write(line)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _maybe_rotate(self) -> None:
        """Rename current file to ``<stem>.1.jsonl`` when it exceeds max_bytes."""
        if not (self._path.exists() and self._path.stat().st_size >= self._max_bytes):
            return
        backup = self._path.parent / (self._path.stem + ".1.jsonl")
        if backup.exists():
            backup.unlink()
        self._path.rename(backup)


def _safe(v: Any) -> Any:
    """Return *v* unchanged if it is a JSON-safe scalar, otherwise ``str(v)``."""
    return v if isinstance(v, (str, int, float, bool, type(None))) else str(v)
