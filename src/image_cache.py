"""Cross-run DALL-E image cache using prompt-embedding cosine similarity.

Before generating a new image, find_similar() checks whether any previously
generated image has a visual_prompt within `threshold` cosine distance of the
new prompt.  At 0.95 the two prompts describe nearly the same scene; reusing
the cached image saves ~$0.08 (DALL-E HD) or ~$0.003 (SD) per scene.

Storage (two files inside data_dir):
  dalle_image_cache.json  — list of {prompt, path, ts} dicts
  dalle_image_cache.npy   — float32 matrix, shape (N, 1536)

Embeddings: OpenAI text-embedding-3-small (~$0.000006 / prompt).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from src.retry_utils import make_openai_client

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_JSON_NAME = "dalle_image_cache.json"
_NPY_NAME  = "dalle_image_cache.npy"
_DIM       = 1536   # text-embedding-3-small


# ── storage ──────────────────────────────────────────────────────────────────

def _json_path(data_dir: Path) -> Path:
    return data_dir / _JSON_NAME


def _npy_path(data_dir: Path) -> Path:
    return data_dir / _NPY_NAME


def _load(data_dir: Path) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Return (metadata_list, embedding_matrix).  Prunes missing files."""
    jp  = _json_path(data_dir)
    np_ = _npy_path(data_dir)

    raw_entries = json.loads(jp.read_text()) if jp.exists() else []
    entries: list[dict[str, Any]] = []
    for e in raw_entries:
        if not isinstance(e, dict):
            continue
        path = e.get("path")
        if not isinstance(path, str):
            continue
        entries.append(e)
    matrix = np.load(str(np_)).astype(np.float32) if (np_.exists() and entries) \
             else np.zeros((0, _DIM), dtype=np.float32)

    # Safety: keep lengths in sync
    n = min(len(entries), len(matrix))
    entries, matrix = entries[:n], matrix[:n]

    # Prune stale entries (image file deleted / output dir cleaned)
    keep = [i for i, e in enumerate(entries) if Path(e["path"]).exists()]
    if len(keep) < len(entries):
        logger.debug(f"Image cache: pruned {len(entries)-len(keep)} stale entries")
    entries = [entries[i] for i in keep]
    matrix  = matrix[keep] if keep else np.zeros((0, _DIM), dtype=np.float32)

    return entries, matrix


def _save(data_dir: Path, entries: list[dict[str, Any]], matrix: np.ndarray) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    _json_path(data_dir).write_text(json.dumps(entries, indent=2))
    np.save(str(_npy_path(data_dir)), matrix)


# ── embedding ─────────────────────────────────────────────────────────────────

def _embed(text: str) -> np.ndarray:
    client = make_openai_client()
    resp   = client.embeddings.create(model="text-embedding-3-small", input=text[:2048])
    return np.array(resp.data[0].embedding, dtype=np.float32)


# ── public API ────────────────────────────────────────────────────────────────

def find_similar(
    prompt: str,
    data_dir: Path,
    threshold: float = 0.95,
) -> tuple[Path | None, np.ndarray | None]:
    """Return (cached_image_path | None, prompt_embedding).

    The embedding is returned so callers can pass it directly to add_entry()
    without making a second API call.
    """
    try:
        entries, matrix = _load(data_dir)
        vec = _embed(prompt)

        if len(entries) == 0:
            return None, vec

        norms = np.linalg.norm(matrix, axis=1)
        norms[norms == 0] = 1.0
        sims    = matrix @ vec / (norms * (np.linalg.norm(vec) or 1.0))
        best_i  = int(np.argmax(sims))
        best_sim = float(sims[best_i])

        if best_sim >= threshold:
            cached = Path(entries[best_i]["path"])
            logger.info(
                f"Image cache HIT: similarity={best_sim:.3f} "
                f"→ {cached.name} (saved ~$0.08)"
            )
            return cached, vec

        logger.debug(f"Image cache miss: best_sim={best_sim:.3f} (need ≥{threshold})")
        return None, vec

    except Exception as exc:
        logger.warning(f"Image cache lookup failed (non-fatal): {exc}")
        return None, None


def add_entry(
    prompt: str,
    image_path: Path,
    data_dir: Path,
    embedding: np.ndarray | None = None,
) -> None:
    """Persist a new cache entry.  Pass the embedding from find_similar() to
    avoid a redundant API call."""
    try:
        entries, matrix = _load(data_dir)
        vec = embedding if embedding is not None else _embed(prompt)

        entry = {
            "prompt": prompt[:500],
            "path":   str(image_path.resolve()),
            "ts":     int(time.time()),
        }
        entries.append(entry)
        matrix = np.vstack([matrix, vec[np.newaxis, :]]) \
                 if len(matrix) else vec[np.newaxis, :]

        _save(data_dir, entries, matrix)
        logger.debug(f"Image cache: stored entry #{len(entries)} → {image_path.name}")
    except Exception as exc:
        logger.warning(f"Image cache add_entry failed (non-fatal): {exc}")
