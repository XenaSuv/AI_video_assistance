"""Hook Mutation Engine for evolving high-performing opening hooks."""
from __future__ import annotations

import random
from typing import Any

from loguru import logger

sys_path_insert = __import__("sys").path.insert
import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from config import settings


class HookMutationEngine:
    """Mutates top hooks to create diverse, testable variations."""

    def __init__(self, llm: Any | None = None):
        self.llm = llm
        self.max_mutations = settings.get("hook_mutation_limit", 8) if hasattr(settings, "get") else 8

    def run(self, top_hooks: list[str], context: dict[str, Any]) -> dict[str, Any]:
        """Generate mutated hook candidates from top-performing hooks."""
        mutated: list[str] = []

        base_hooks = self._select_base_hooks(top_hooks)
        logger.info(f"Selected {len(base_hooks)} base hooks for mutation")

        for hook in base_hooks:
            mutated.extend(self._rule_mutations(hook))
            mutated.extend(self._llm_mutations(hook, context))

        mutated = self._deduplicate(mutated)
        mutated = self._diversity_filter(mutated)

        scored = self._score_mutations(mutated, context)
        mutated = [item["hook"] for item in scored][: self.max_mutations]

        logger.info(f"Produced {len(mutated)} mutated hooks")
        return {"mutated_hooks": mutated}

    def _select_base_hooks(self, top_hooks: list[str]) -> list[str]:
        """Select the strongest hooks to mutate."""
        if not top_hooks:
            return []
        return top_hooks[: min(len(top_hooks), 5)]

    def _rule_mutations(self, hook: str) -> list[str]:
        """Fast, stable rule-based variations."""
        lower = hook.lower()
        variations: list[str] = []

        if "problem" in lower:
            variations.append("But here's what no one is saying...")
            variations.append("But this is where things break down...")
            variations.append("But the real problem is hidden here...")

        if "no one" in lower:
            variations.append("Everyone is missing this...")
            variations.append("This part is being ignored...")
            variations.append("No one wants to admit this...")

        if "here's why" in lower or "here's what" in lower:
            variations.append("Here's why it matters more than you think...")
            variations.append("Here's what most people miss...")

        if "but" in lower and "problem" not in lower:
            variations.append("But this is the thing nobody notices...")

        return self._normalize_hooks(variations)

    def _llm_mutations(self, hook: str, context: dict[str, Any]) -> list[str]:
        """Generate hook variations using an LLM if available."""
        if not self.llm:
            return []

        prompt = f"""
Take this hook and generate 3 short variations.

Hook: \"{hook}\"

Context:
- Angle: {context.get('angle', 'unknown')}
- Persona: {context.get('persona', 'neutral')}
- Format: {context.get('format', 'general')}

Rules:
- Keep it short (max 10 words)
- Make it more engaging
- Add curiosity or tension
- Use natural, human phrasing
"""

        try:
            response = self.llm(prompt)
            if isinstance(response, str):
                items = [line.strip("\" ") for line in response.splitlines() if line.strip()]
            elif isinstance(response, list):
                items = [str(item).strip() for item in response if str(item).strip()]
            else:
                items = []
        except Exception as exc:
            logger.warning(f"LLM mutation failed: {exc}")
            items = []

        return self._normalize_hooks(items[:3])

    def _normalize_hooks(self, hooks: list[str]) -> list[str]:
        """Normalize hook text and remove short junk."""
        normalized = []
        for hook in hooks:
            clean = hook.strip()
            if len(clean) < 10:
                continue
            if clean.endswith("."):
                clean = clean[:-1].strip()
            normalized.append(clean)
        return normalized

    def _deduplicate(self, hooks: list[str]) -> list[str]:
        """Remove duplicate hooks while preserving order."""
        seen = set()
        unique = []
        for hook in hooks:
            key = hook.lower().strip()
            if key not in seen:
                seen.add(key)
                unique.append(hook)
        return unique

    def _diversity_filter(self, hooks: list[str]) -> list[str]:
        """Keep only the most diverse hook set."""
        unique = self._deduplicate(hooks)
        if len(unique) <= 5:
            return unique
        random.shuffle(unique)
        return unique[:5]

    def _score_mutations(self, hooks: list[str], context: dict[str, Any]) -> list[dict[str, Any]]:
        """Score mutated hooks for a lightweight ranking."""
        scored = []
        for hook in hooks:
            score = 0.0
            lower = hook.lower()
            score += 0.4 if any(word in lower for word in ["no one", "everyone", "never", "hidden", "secret"]) else 0.0
            score += 0.3 if any(word in lower for word in ["but", "this is", "what", "why", "how"]) else 0.0
            score += 0.2 if len(hook.split()) <= 10 else 0.0
            if context.get("angle") and context["angle"] in lower:
                score += 0.1
            scored.append({"hook": hook, "score": round(score, 2)})
        scored.sort(key=lambda item: item["score"], reverse=True)  # type: ignore[arg-type,return-value]
        return scored


def mutate_hooks(top_hooks: list[str], context: dict[str, Any], llm: Any | None = None) -> dict[str, Any]:
    engine = HookMutationEngine(llm=llm)
    return engine.run(top_hooks, context)
