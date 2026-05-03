"""Micro-hook agent for inserting attention-grabbing hooks throughout the script."""
from __future__ import annotations

import json
import re
import random
from typing import Any

from loguru import logger
from openai import OpenAI

sys_path_insert = __import__("sys").path.insert
from pathlib import Path
from pathlib import Path as _Path

import sys
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from config import settings
from src.hook_optimizer import HookOptimizer
from src.shared_types import InsertedHook, MicroHookResult


class MicroHookAgent:
    """Agent that inserts micro-hooks throughout the script to maintain viewer engagement."""

    HOOK_TYPES = {
        "curiosity": [
            "but here's the weird part...",
            "and this is where it gets interesting",
            "you won't believe what happened next",
            "there's something you need to know",
        ],
        "conflict": [
            "but there's a problem",
            "and not everyone agrees",
            "this creates a real dilemma",
            "the catch is...",
        ],
        "twist": [
            "and then something unexpected happened",
            "this changed everything",
            "the plot thickens",
            "here's the twist",
        ],
        "personal": [
            "if you're building anything with AI, this matters",
            "this might affect you more than you think",
            "for developers, this is crucial",
            "this could impact your work tomorrow",
        ],
        "doubt": [
            "I'm not fully convinced",
            "something feels off here",
            "this raises some questions",
            "I'm skeptical about this",
        ],
    }

    def __init__(self, llm: OpenAI | None = None, hook_frequency: int = 3):
        self.llm = llm or OpenAI(api_key=settings.openai_api_key)
        self.hook_frequency = hook_frequency  # Insert hook every N paragraphs
        self.hook_optimizer = HookOptimizer()

    def run(
        self,
        script: str,
        scene_plan: list[dict[str, Any]],
        persona: dict[str, Any],
    ) -> MicroHookResult:
        """Insert micro-hooks into the script."""
        logger.info("Starting micro-hook insertion")

        editorial_context = {
            "angle": scene_plan[0].get("angle", "unknown") if scene_plan else "unknown",
            "format": "unknown",
            "persona": persona,
        }

        optimized = self.hook_optimizer.run(editorial_context)
        recommended_patterns = optimized.recommended_patterns
        avoid_patterns = optimized.avoid_patterns

        logger.info(f"Using {len(recommended_patterns)} optimized hook patterns")

        # Split script into paragraphs
        paragraphs = self._split_into_paragraphs(script)

        # Determine persona-appropriate hook types
        hook_types = self._get_persona_hook_types(persona)

        inserted_hooks = []
        enhanced_paragraphs = []
        hook_count = 0

        for i, para in enumerate(paragraphs):
            enhanced_paragraphs.append(para)

            # Insert hook every N paragraphs (but not at the very end)
            if (i + 1) % self.hook_frequency == 0 and i < len(paragraphs) - 1:
                hook = self._generate_hook(hook_types, para, scene_plan, recommended_patterns, avoid_patterns)
                if hook:
                    enhanced_paragraphs.append(hook)
                    inserted_hooks.append(InsertedHook(
                        position=len("\n\n".join(enhanced_paragraphs)),
                        type=hook["type"],
                        text=hook["text"],
                        optimized=hook.get("optimized", False),
                    ))
                    hook_count += 1

        final_script = "\n\n".join(enhanced_paragraphs)
        logger.info(f"Inserted {hook_count} micro-hooks")

        return MicroHookResult(
            final_script=final_script,
            inserted_hooks=inserted_hooks,
            optimization=optimized,
        )

    def _split_into_paragraphs(self, script: str) -> list[str]:
        """Split script into paragraphs."""
        # Split on double newlines or periods followed by capital letters
        paragraphs = re.split(r'\n\s*\n', script.strip())
        # Filter out empty paragraphs
        return [p.strip() for p in paragraphs if p.strip()]

    def _get_persona_hook_types(self, persona: dict[str, Any]) -> list[str]:
        """Determine appropriate hook types based on persona."""
        style = persona.get("style", "").lower()

        if "skeptical" in style or "cynical" in style:
            return ["doubt", "conflict", "curiosity"]
        elif "curious" in style or "engineer" in style:
            return ["curiosity", "twist", "personal"]
        elif "explainer" in style:
            return ["personal", "curiosity", "twist"]
        else:
            return ["curiosity", "conflict", "twist", "personal"]

    def _generate_hook(
        self,
        hook_types: list[str],
        context: str,
        scene_plan: list[dict[str, Any]],
        recommended_patterns: list[str],
        avoid_patterns: list[str]
    ) -> dict[str, Any] | None:
        """Generate a contextually appropriate micro-hook with voice annotations."""
        # Use optimized patterns if available
        if recommended_patterns:
            # Try to use a recommended pattern first
            for pattern in recommended_patterns:
                if pattern in self.HOOK_TYPES:
                    hook_type = pattern
                    if hook_type in hook_types:  # Ensure it matches persona
                        templates = self.HOOK_TYPES[hook_type]
                        template = templates[len(context.split()) % len(templates)]
                        annotated_text = f"[PAUSE_SHORT]\n{template}\n[PAUSE_SHORT]"
                        return {
                            "type": hook_type,
                            "text": annotated_text,
                            "optimized": True,
                        }

        # Avoid patterns that have performed poorly
        filtered_hook_types = [ht for ht in hook_types if ht not in avoid_patterns]

        if not filtered_hook_types:
            filtered_hook_types = hook_types  # Fallback if all are avoided

        # Simple rule-based generation
        hook_type = filtered_hook_types[len(context) % len(filtered_hook_types)]

        templates = self.HOOK_TYPES[hook_type]
        # Pick template based on context length for variety
        template = templates[len(context.split()) % len(templates)]

        # Add voice annotations
        annotated_text = f"[PAUSE_SHORT]\n{template}\n[PAUSE_SHORT]"

        return {
            "type": hook_type,
            "text": annotated_text,
            "optimized": False,
        }


def add_micro_hooks(
    script: str,
    scene_plan: list[dict[str, Any]],
    persona: dict[str, Any],
    hook_frequency: int = 3,
) -> dict[str, Any]:
    """Convenience function for adding micro-hooks."""
    agent = MicroHookAgent(hook_frequency=hook_frequency)
