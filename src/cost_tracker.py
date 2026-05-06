"""Central cost ledger for all LLM, TTS, and image API calls.

Usage
-----
At pipeline start:
    ledger = reset_ledger()

In any module:
    from src.cost_tracker import get_ledger
    get_ledger().record_llm("script", "gpt-4o-mini", prompt_tokens, completion_tokens)
    get_ledger().record_tts("tts-en", char_count, "eleven_turbo_v2_5")
    get_ledger().record_image("image-scene_01", "dall-e-3")

At pipeline end:
    report = get_ledger().save(run_dir / "cost_report.json")
    logger.info(f"Total cost: ${report['total_usd']:.4f}")
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

# ── Prices ────────────────────────────────────────────────────────────────────
# USD per 1M tokens (LLM), per 1K chars (TTS), per image (DALL-E).
# Override by passing custom_prices= to CostLedger().

_DEFAULT_LLM_PRICES: dict[str, dict[str, float]] = {
    # model → {input, output, cached}  (per 1M tokens)
    "gpt-4o-mini":           {"input": 0.150,  "output": 0.600,  "cached": 0.075},
    "gpt-4o":                {"input": 2.500,  "output": 10.000, "cached": 1.250},
    "gpt-4o-2024-11-20":     {"input": 2.500,  "output": 10.000, "cached": 1.250},
    "gpt-4.1":               {"input": 2.000,  "output": 8.000,  "cached": 0.500},
    "gpt-4.1-mini":          {"input": 0.400,  "output": 1.600,  "cached": 0.100},
    "o3-mini":               {"input": 1.100,  "output": 4.400,  "cached": 0.550},
}

_DEFAULT_TTS_PRICES: dict[str, float] = {
    # model → USD per 1K characters
    "eleven_turbo_v2_5":      0.30,
    "eleven_multilingual_v2": 0.30,
    "eleven_monolingual_v1":  0.30,
}

_DEFAULT_IMAGE_PRICES: dict[str, float] = {
    # model → USD per image
    "dall-e-3":               0.080,   # HD 1024×1024
    "dall-e-2":               0.020,
    "stable-image-core":      0.003,   # Stability AI
}


@dataclass
class _Entry:
    tag: str
    kind: str          # "llm" | "tts" | "image"
    model: str
    usd: float
    ts: float = field(default_factory=time.time)
    # LLM-specific
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    # TTS-specific
    chars: int = 0
    # Image-specific
    n: int = 0


class CostLedger:
    """Accumulates API costs for a single pipeline run."""

    def __init__(self, custom_prices: dict[str, Any] | None = None):
        self._entries: list[_Entry] = []
        self._llm_prices   = {**_DEFAULT_LLM_PRICES,   **(custom_prices or {}).get("llm",   {})}
        self._tts_prices   = {**_DEFAULT_TTS_PRICES,   **(custom_prices or {}).get("tts",   {})}
        self._image_prices = {**_DEFAULT_IMAGE_PRICES, **(custom_prices or {}).get("image", {})}

    # ── record helpers ────────────────────────────────────────────────────────

    def record_llm(
        self,
        tag: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cached_tokens: int = 0,
    ) -> None:
        prices = self._llm_prices.get(model) or self._llm_prices.get("gpt-4o-mini")
        assert prices is not None, f"No price data for model {model!r}"
        usd = (
            (prompt_tokens - cached_tokens) * prices["input"] / 1_000_000
            + cached_tokens                * prices["cached"] / 1_000_000
            + completion_tokens            * prices["output"] / 1_000_000
        )
        self._entries.append(_Entry(
            tag=tag, kind="llm", model=model, usd=round(usd, 6),
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
        ))
        logger.debug(
            f"[cost] {tag} llm/{model} "
            f"in={prompt_tokens}(cached={cached_tokens}) out={completion_tokens} "
            f"${usd:.4f}"
        )

    def record_tts(self, tag: str, chars: int, model: str) -> None:
        price_per_k = self._tts_prices.get(model, 0.30)
        usd = chars * price_per_k / 1000
        self._entries.append(_Entry(
            tag=tag, kind="tts", model=model, usd=round(usd, 6), chars=chars,
        ))
        logger.debug(f"[cost] {tag} tts/{model} chars={chars} ${usd:.4f}")

    def record_image(self, tag: str, model: str, n: int = 1) -> None:
        price_per_img = self._image_prices.get(model, 0.080)
        usd = n * price_per_img
        self._entries.append(_Entry(
            tag=tag, kind="image", model=model, usd=round(usd, 6), n=n,
        ))
        logger.debug(f"[cost] {tag} image/{model} n={n} ${usd:.4f}")

    # ── reporting ─────────────────────────────────────────────────────────────

    def total_usd(self) -> float:
        return round(sum(e.usd for e in self._entries), 4)

    def summary(self) -> dict[str, Any]:
        by_tag: dict[str, dict[str, Any]] = {}
        by_kind: dict[str, dict[str, Any]] = {
            "llm":   {"usd": 0.0, "prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0},
            "tts":   {"usd": 0.0, "chars": 0},
            "image": {"usd": 0.0, "count": 0},
        }

        for e in self._entries:
            t = by_tag.setdefault(e.tag, {"usd": 0.0})
            t["usd"] = round(t["usd"] + e.usd, 6)

            k = by_kind[e.kind]
            k["usd"] = round(k["usd"] + e.usd, 6)
            if e.kind == "llm":
                k["prompt_tokens"]     += e.prompt_tokens
                k["completion_tokens"] += e.completion_tokens
                k["cached_tokens"]     += e.cached_tokens
            elif e.kind == "tts":
                k["chars"] += e.chars
            elif e.kind == "image":
                k["count"] += e.n

        # round by_tag usd
        for t in by_tag.values():
            t["usd"] = round(t["usd"], 4)

        return {
            "total_usd": self.total_usd(),
            "by_tag":    by_tag,
            "by_kind":   {k: v for k, v in by_kind.items() if v["usd"] > 0},
            "entries": [
                {
                    "tag": e.tag, "kind": e.kind, "model": e.model,
                    "usd": e.usd,
                    **({"prompt_tokens": e.prompt_tokens,
                        "completion_tokens": e.completion_tokens,
                        "cached_tokens": e.cached_tokens} if e.kind == "llm" else {}),
                    **({"chars": e.chars} if e.kind == "tts" else {}),
                    **({"n": e.n}         if e.kind == "image" else {}),
                }
                for e in self._entries
            ],
        }

    def save(self, path: Path) -> dict[str, Any]:
        """Write JSON report to path and return it."""
        report = self.summary()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(report, indent=2))
        tmp.replace(path)
        logger.info(
            f"Cost report saved → {path.name}  "
            f"total=${report['total_usd']:.4f}  "
            f"entries={len(self._entries)}"
        )
        return report

    def log_summary(self) -> None:
        """Log a one-line cost breakdown to INFO."""
        s = self.summary()
        parts = []
        if llm := s["by_kind"].get("llm"):
            parts.append(
                f"LLM ${llm['usd']:.4f} "
                f"({llm['prompt_tokens']}in/{llm['completion_tokens']}out"
                f"/{llm['cached_tokens']}cached tokens)"
            )
        if tts := s["by_kind"].get("tts"):
            parts.append(f"TTS ${tts['usd']:.4f} ({tts['chars']} chars)")
        if img := s["by_kind"].get("image"):
            parts.append(f"Images ${img['usd']:.4f} ({img['count']} imgs)")
        logger.info(f"[cost] total=${s['total_usd']:.4f}  " + "  ".join(parts))


# ── module-level singleton ────────────────────────────────────────────────────

_ledger: CostLedger | None = None


def get_ledger() -> CostLedger:
    """Return the current pipeline's ledger, creating one lazily if needed."""
    global _ledger
    if _ledger is None:
        _ledger = CostLedger()
    return _ledger


def reset_ledger(custom_prices: dict[str, Any] | None = None) -> CostLedger:
    """Create a fresh ledger (call at the start of each pipeline run)."""
    global _ledger
    _ledger = CostLedger(custom_prices)
    return _ledger
