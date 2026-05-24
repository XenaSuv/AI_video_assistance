"""Humanizer agent for making AI-generated scripts sound more human-like."""
from __future__ import annotations

from typing import Any

from loguru import logger
from openai import OpenAI

from src.retry_utils import make_openai_client

sys_path_insert = __import__("sys").path.insert
import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from config import settings
from src.cost_tracker import get_ledger
from src.shared_types import HumanizationResult


class HumanizerAgent:
    """Agent that transforms AI-generated scripts into more human-like narration."""

    def __init__(
        self,
        llm: OpenAI | None = None,
        humanization_level: float = 0.7,
        chaos_level: float = 0.4,
        emotion_level: float = 0.6,
    ):
        self.llm = llm or make_openai_client()
        self.humanization_level = humanization_level
        self.chaos_level = chaos_level
        self.emotion_level = emotion_level

    def run(
        self,
        script: str,
        editorial_plan: dict[str, Any],
        persona: dict[str, Any],
    ) -> HumanizationResult:
        """Humanize the script with multiple passes."""
        logger.info("Starting humanization process")

        # Multi-pass approach for better quality
        changes = []

        # Pass 1: Structure breaker
        script, change1 = self._structure_breaker(script)
        changes.extend(change1)

        # Pass 2: Sentence variator
        script, change2 = self._sentence_variator(script)
        changes.extend(change2)

        # Pass 3: Reaction injector
        script, change3 = self._reaction_injector(script, persona)
        changes.extend(change3)

        # Pass 4: Imperfection layer
        script, change4 = self._imperfection_layer(script)
        changes.extend(change4)

        # Pass 5: Rhythm optimizer
        script, change5 = self._rhythm_optimizer(script)
        changes.extend(change5)

        # Pass 6: Voice tuner
        script, change6 = self._voice_tuner(script, persona, editorial_plan)
        changes.extend(change6)

        logger.info(f"Humanization complete with {len(changes)} changes")
        return HumanizationResult(final_script=script, changes=changes)

    def _structure_breaker(self, script: str) -> tuple[str, list[str]]:
        """Break perfect paragraph structure into shorter, punchier chunks."""
        prompt = f"""
Break the script's perfect structure. Make it feel like spoken thoughts, not written paragraphs.

Rules:
- Split long paragraphs into 1-3 sentence chunks
- Add abrupt transitions sometimes
- Keep all factual content intact
- Make it feel conversational

Original script:
{script}

Return only the rewritten script, no explanations.
"""
        try:
            response = self.llm.chat.completions.create(
                model=settings.openai_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.chaos_level,
                max_tokens=2000,
            )
            new_script = (response.choices[0].message.content or "").strip()
            if response.usage:
                _det = response.usage.prompt_tokens_details
                get_ledger().record_llm(
                    tag="humanize-structure", model=settings.openai_model,
                    prompt_tokens=response.usage.prompt_tokens or 0,
                    completion_tokens=response.usage.completion_tokens or 0,
                    cached_tokens=getattr(_det, "cached_tokens", 0) if _det else 0,
                )
            changes = ["broke perfect paragraph structure into conversational chunks"]
            return new_script, changes
        except Exception as exc:
            logger.warning(f"Structure breaker failed: {exc}")
            return script, []

    def _sentence_variator(self, script: str) -> tuple[str, list[str]]:
        """Vary sentence lengths to create natural rhythm."""
        prompt = f"""
Vary sentence lengths in this script. Mix short punchy sentences with longer explanatory ones.

Rules:
- Short sentences for impact
- Medium sentences for explanation
- Long sentences sparingly for detail
- Keep natural flow

Script:
{script}

Return only the rewritten script.
"""
        try:
            response = self.llm.chat.completions.create(
                model=settings.openai_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.chaos_level * 0.8,
                max_tokens=2000,
            )
            new_script = (response.choices[0].message.content or "").strip()
            if response.usage:
                _det = response.usage.prompt_tokens_details
                get_ledger().record_llm(
                    tag="humanize-sentences", model=settings.openai_model,
                    prompt_tokens=response.usage.prompt_tokens or 0,
                    completion_tokens=response.usage.completion_tokens or 0,
                    cached_tokens=getattr(_det, "cached_tokens", 0) if _det else 0,
                )
            changes = ["varied sentence lengths for natural rhythm"]
            return new_script, changes
        except Exception as exc:
            logger.warning(f"Sentence variator failed: {exc}")
            return script, []

    def _reaction_injector(self, script: str, persona: dict[str, Any]) -> tuple[str, list[str]]:
        """Add emotional reactions and subjective elements."""
        style = persona.get("style", "conversational")
        prompt = f"""
Add human reactions and emotions to make this script feel like a real person speaking.

Persona style: {style}

Add patterns like:
- "Honestly..."
- "This is weird"
- "I didn't expect this"
- "This is where it gets interesting"
- Personal opinions and reactions

Keep all facts intact, just make it more engaging.

Script:
{script}

Return only the rewritten script.
"""
        try:
            response = self.llm.chat.completions.create(
                model=settings.openai_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.emotion_level,
                max_tokens=2000,
            )
            new_script = (response.choices[0].message.content or "").strip()
            if response.usage:
                _det = response.usage.prompt_tokens_details
                get_ledger().record_llm(
                    tag="humanize-reactions", model=settings.openai_model,
                    prompt_tokens=response.usage.prompt_tokens or 0,
                    completion_tokens=response.usage.completion_tokens or 0,
                    cached_tokens=getattr(_det, "cached_tokens", 0) if _det else 0,
                )
            changes = ["injected emotional reactions and subjectivity"]
            return new_script, changes
        except Exception as exc:
            logger.warning(f"Reaction injector failed: {exc}")
            return script, []

    def _imperfection_layer(self, script: str) -> tuple[str, list[str]]:
        """Add slight imperfections like repetitions and conversational fillers."""
        prompt = f"""
Add human imperfections to make this sound less robotic.

Add:
- Slight repetitions for emphasis
- Conversational fillers like "like", "you know"
- Incomplete thoughts that get completed
- Natural hesitations

Don't overdo it - keep it professional but human.

Script:
{script}

Return only the rewritten script.
"""
        try:
            response = self.llm.chat.completions.create(
                model=settings.openai_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.chaos_level * 0.6,
                max_tokens=2000,
            )
            new_script = (response.choices[0].message.content or "").strip()
            if response.usage:
                _det = response.usage.prompt_tokens_details
                get_ledger().record_llm(
                    tag="humanize-imperfection", model=settings.openai_model,
                    prompt_tokens=response.usage.prompt_tokens or 0,
                    completion_tokens=response.usage.completion_tokens or 0,
                    cached_tokens=getattr(_det, "cached_tokens", 0) if _det else 0,
                )
            changes = ["added slight imperfections and conversational elements"]
            return new_script, changes
        except Exception as exc:
            logger.warning(f"Imperfection layer failed: {exc}")
            return script, []

    def _rhythm_optimizer(self, script: str) -> tuple[str, list[str]]:
        """Optimize pacing with text-based pauses and emphasis."""
        prompt = f"""
Optimize the rhythm of this script for spoken delivery.

Add:
- Text pauses with "..."
- Emphasis with italics or CAPS for key words
- Speed up beginnings, slow down important parts
- Natural breathing points

Make it feel like it's being spoken live.

Script:
{script}

Return only the rewritten script.
"""
        try:
            response = self.llm.chat.completions.create(
                model=settings.openai_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.humanization_level * 0.8,
                max_tokens=2000,
            )
            new_script = (response.choices[0].message.content or "").strip()
            if response.usage:
                _det = response.usage.prompt_tokens_details
                get_ledger().record_llm(
                    tag="humanize-rhythm", model=settings.openai_model,
                    prompt_tokens=response.usage.prompt_tokens or 0,
                    completion_tokens=response.usage.completion_tokens or 0,
                    cached_tokens=getattr(_det, "cached_tokens", 0) if _det else 0,
                )
            changes = ["optimized rhythm with pauses and emphasis"]
            return new_script, changes
        except Exception as exc:
            logger.warning(f"Rhythm optimizer failed: {exc}")
            return script, []

    def _voice_tuner(self, script: str, persona: dict[str, Any], editorial_plan: dict[str, Any]) -> tuple[str, list[str]]:
        """Fine-tune voice to match persona and editorial style."""
        style = persona.get("style", "conversational")
        rules = persona.get("rules", [])
        tone = editorial_plan.get("tone", "curious")

        prompt = f"""
Fine-tune this script to perfectly match the persona voice.

Persona style: {style}
Persona rules: {', '.join(rules)}
Editorial tone: {tone}

Adjust language, attitude, and delivery to embody this voice completely.
Make it sound like this specific person is speaking.

Script:
{script}

Return only the rewritten script.
"""
        try:
            response = self.llm.chat.completions.create(
                model=settings.openai_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.humanization_level,
                max_tokens=2000,
            )
            new_script = (response.choices[0].message.content or "").strip()
            if response.usage:
                _det = response.usage.prompt_tokens_details
                get_ledger().record_llm(
                    tag="humanize-voice", model=settings.openai_model,
                    prompt_tokens=response.usage.prompt_tokens or 0,
                    completion_tokens=response.usage.completion_tokens or 0,
                    cached_tokens=getattr(_det, "cached_tokens", 0) if _det else 0,
                )
            changes = ["fine-tuned voice to match persona and editorial tone"]
            return new_script, changes
        except Exception as exc:
            logger.warning(f"Voice tuner failed: {exc}")
            return script, []


def humanize_script(
    script: str,
    editorial_plan: dict[str, Any],
    persona: dict[str, Any],
    humanization_level: float = 0.7,
    chaos_level: float = 0.4,
    emotion_level: float = 0.6,
) -> HumanizationResult:
    """Convenience function for single humanization call."""
    agent = HumanizerAgent(
        humanization_level=humanization_level,
        chaos_level=chaos_level,
        emotion_level=emotion_level,
    )
    return agent.run(script, editorial_plan, persona)
