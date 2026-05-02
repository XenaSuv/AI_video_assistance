"""Editorial decision-making for story selection, angle, hooks, and scene planning."""
from __future__ import annotations

import random
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from loguru import logger
from openai import OpenAI

sys_path_insert = __import__("sys").path.insert
from pathlib import Path
from pathlib import Path as _Path

import sys
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from config import settings
from src.scraper import NewsItem
from src.feedback_analyzer import FeedbackAnalyzer

ANGLE_CANDIDATES = [
    {
        "key": "technical_breakthrough",
        "label": "This looks like a breakthrough, but might be overhyped",
        "conflict": "hype vs reality",
    },
    {
        "key": "industry_impact",
        "label": "This announcement could reshape the AI industry immediately",
        "conflict": "innovation vs adoption",
    },
    {
        "key": "threat_to_jobs",
        "label": "This technology might replace jobs faster than people expect",
        "conflict": "growth vs jobs",
    },
    {
        "key": "overhyped_vs_reality",
        "label": "This is being hyped, but the reality is more complicated",
        "conflict": "hype vs reality",
    },
    {
        "key": "what_this_means_for_you",
        "label": "Here’s what this means for people actually using AI",
        "conflict": "promise vs practical use",
    },
]

CONFLICT_TERMS = {
    "bias", "lawsuit", "ban", "replace", "attack", "failure", "exploit",
    "crash", "restriction", "delay", "shutdown", "policy", "regulation",
    "security", "cost", "ethical", "privacy", "misinformation",
}

AI_AUDIENCE_TERMS = {
    "developer", "devops", "engineer", "ml", "ai", "model", "api", "fine-tune",
    "prompt", "code", "research", "benchmark", "open source", "security",
}

STORY_TYPES = {
    "corporate": ["company", "launch", "business", "partnership", "enterprise", "funding", "investment"],
    "research": ["paper", "research", "study", "benchmark", "algorithm", "model", "dataset"],
    "policy": ["regulation", "law", "privacy", "ban", "government", "policy"],
    "product": ["app", "feature", "release", "update", "tool", "platform"],
}

FORMAT_RULES = {
    "quick_hit": {
        "scene_plan": [
            {"type": "hook", "duration": 5, "style": "fast_cut"},
            {"type": "context", "duration": 12, "style": "visual_explain"},
            {"type": "impact", "duration": 10, "style": "zoom_overlay"},
            {"type": "takeaway", "duration": 8, "style": "callout"},
        ],
    },
    "deep_dive": {
        "scene_plan": [
            {"type": "hook", "duration": 7, "style": "slow_build"},
            {"type": "context", "duration": 15, "style": "visual_explain"},
            {"type": "technical", "duration": 18, "style": "whiteboard"},
            {"type": "implication", "duration": 12, "style": "data_overlay"},
            {"type": "summary", "duration": 8, "style": "calm"},
        ],
    },
    "hot_take": {
        "scene_plan": [
            {"type": "hook", "goal": "shock", "style": "fast_cut"},
            {"type": "conflict", "goal": "show tension", "style": "contrast"},
            {"type": "twist", "goal": "reveal unexpected angle", "style": "stat_card"},
            {"type": "implication", "goal": "make it personal", "style": "dramatic"},
            {"type": "close", "goal": "leave the viewer reacting", "style": "firm"},
        ],
    },
}

@dataclass
class EditorialPlan:
    selected_stories: list[dict[str, Any]]
    editorial_plan: list[dict[str, Any]]
    global_style: dict[str, str]


class EditorialBrain:
    def __init__(self, llm: OpenAI | None = None, config: dict[str, Any] | None = None):
        self.llm = llm or OpenAI(api_key=settings.openai_api_key)
        self.config = config or {}
        self.feedback_analyzer = FeedbackAnalyzer()
        self.angle_performance_cache = {}
        self.format_performance_cache = {}

    def run(
        self,
        stories: list[NewsItem],
        history: list[str] | None = None,
        channel_config: dict[str, Any] | None = None,
        platform: str = "youtube_long",
    ) -> EditorialPlan:
        history = history or []
        channel_config = channel_config or {}
        ranked = self.rank_stories(stories, history, platform)
        selected = ranked[:5]

        variation = random.choice(["controversial", "explainer", "storytelling", "fast_news"])
        plans = []
        for story in selected:
            plan = self._plan_story(story, history, channel_config, platform, variation)
            plans.append(plan)

        style = self._choose_global_style(channel_config, platform)
        return EditorialPlan(
            selected_stories=[self._story_summary(s) for s in selected],
            editorial_plan=plans,
            global_style=style,
        )

    def rank_stories(self, stories: list[NewsItem], history: list[str], platform: str) -> list[NewsItem]:
        scored = []
        for story in stories:
            hype = self._hype_score(story)
            novelty = self._novelty_score(story, history)
            controversy = self._controversy_score(story)
            audience_fit = self._audience_fit(story, platform)
            recency = self._recency_score(story)
            score = (hype * 0.3 + novelty * 0.25 + controversy * 0.2 + audience_fit * 0.15 + recency * 0.1)
            story_meta = {
                "story": story,
                "score": score,
                "hype_score": hype,
                "novelty_score": novelty,
                "controversy_score": controversy,
                "audience_fit": audience_fit,
                "recency": recency,
            }
            scored.append(story_meta)

        scored.sort(key=lambda item: item["score"], reverse=True)
        ranked = [item["story"] for item in scored]
        logger.info(
            "Editorial ranking: %s",
            ", ".join(f"[{i+1}] {s.title[:60]}" for i, s in enumerate(ranked[:5])),
        )
        return ranked

    def _plan_story(
        self,
        story: NewsItem,
        history: list[str],
        channel_config: dict[str, Any],
        platform: str,
        variation: str,
    ) -> dict[str, Any]:
        angle_data = self._pick_angle(story, history)
        angle = angle_data["label"]
        conflict = angle_data["conflict"]
        persona = self._select_persona(story, channel_config)
        tone = self._select_tone(story, angle_data["key"], variation)
        hook_variants = self._generate_hook_variants(story, angle_data["key"])
        hook = self._rank_hooks(hook_variants, story)
        format_data = self._decide_format(story, angle_data["key"], variation)
        fmt = format_data["format"]
        scene_plan = self._director_bridge(format_data, angle_data["key"], variation)

        return {
            "story_id": story.url,
            "angle": angle,
            "conflict": conflict,
            "tone": tone,
            "hook": hook,
            "hook_variants": hook_variants,
            "novelty_score": self._novelty_score(story, history),
            "controversy_score": self._controversy_score(story),
            "format": fmt,
            "persona": persona,
            "scene_plan": scene_plan,
        }

    def _story_summary(self, story: NewsItem) -> dict[str, Any]:
        return {
            "story_id": story.url,
            "title": story.title,
            "source": story.source,
            "published": story.published,
        }

    def _hype_score(self, story: NewsItem) -> float:
        source_boost = {
            "HackerNews": 1.0,
            "HuggingFace": 0.8,
            "arXiv": 0.4,
            "OpenAI": 0.9,
            "Anthropic": 0.9,
            "Google DeepMind": 0.8,
            "Microsoft AI": 0.7,
        }
        base = source_boost.get(story.source, 0.5)
        buzz = 1.0 if re.search(r"\b(launch|announce|breaking|revealed|reveals|trend|viral|buzz|exploded)\b", story.title, re.I) else 0.0
        return min(1.0, base * 0.7 + buzz * 0.3)

    def _novelty_score(self, story: NewsItem, history: list[str]) -> float:
        if not history:
            return 0.8
        title_tokens = self._tokens(story.title)
        best = 0.0
        for past in history:
            overlap = self._jaccard(title_tokens, self._tokens(past))
            best = max(best, overlap)
        return round(1.0 - best, 2)

    def _controversy_score(self, story: NewsItem) -> float:
        text = f"{story.title} {story.summary}".lower()
        matches = sum(1 for term in CONFLICT_TERMS if term in text)
        return min(1.0, matches * 0.2)

    def _audience_fit(self, story: NewsItem, platform: str) -> float:
        text = f"{story.title} {story.summary}".lower()
        hits = sum(1 for term in AI_AUDIENCE_TERMS if term in text)
        base = min(1.0, hits / 6)
        platform_boost = 0.1 if platform in ("youtube_long", "shorts") else 0.0
        return min(1.0, base + platform_boost)

    def _recency_score(self, story: NewsItem) -> float:
        try:
            delta = datetime.now().date() - datetime.fromisoformat(story.published).date()
            days = delta.days
        except Exception:
            days = 3
        return max(0.0, min(1.0, (7 - days) / 7))

    def _pick_angle(self, story: NewsItem, history: list[str]) -> dict[str, str]:
        scored = []
        for candidate in ANGLE_CANDIDATES:
            score = self._angle_score(story, candidate["key"])
            
            # Boost score based on past performance
            perf = self.feedback_analyzer.get_angle_performance(candidate["key"])
            if perf and perf.get("avg_hook_score", 0) > 0.7:
                score *= 1.2  # Boost high-performing angles
                logger.debug(f"Boosting {candidate['key']} based on past performance: {perf['avg_hook_score']}")
            
            scored.append((score, candidate))
        
        chosen = max(scored, key=lambda item: item[0])[1]
        logger.debug(
            "Angle scores for %s: %s",
            story.title,
            {candidate['key']: self._angle_score(story, candidate['key']) for candidate in ANGLE_CANDIDATES},
        )
        return chosen

    def _angle_score(self, story: NewsItem, angle: str) -> float:
        text = f"{story.title} {story.summary}".lower()
        if angle == "technical_breakthrough":
            return 0.8 if re.search(r"\b(model|paper|benchmark|parameters|training|accuracy|latency)\b", text) else 0.4
        if angle == "industry_impact":
            return 0.75 if re.search(r"\b(launch|enterprise|partnership|investment|adoption|platform|business)\b", text) else 0.5
        if angle == "threat_to_jobs":
            return 0.8 if re.search(r"\b(replace|automation|jobs|workers|hiring|layoff|productivity)\b", text) else 0.2
        if angle == "overhyped_vs_reality":
            return 0.9 if re.search(r"\b(hype|overhyped|promised|but|yet|still|however)\b", text) else 0.3
        if angle == "what_this_means_for_you":
            return 0.85 if re.search(r"\b(users|developers|enterprise|customers|teams|students|workflow)\b", text) else 0.45
        return 0.5

    def _select_persona(self, story: NewsItem, channel_config: dict[str, Any]) -> dict[str, Any]:
        text = f"{story.title} {story.summary}".lower()
        if any(term in text for term in STORY_TYPES["corporate"]):
            style = "sharp, slightly cynical, informal"
            rules = [
                "question claims",
                "avoid corporate language",
                "speak like an insider",
            ]
        elif any(term in text for term in STORY_TYPES["research"]):
            style = "curious engineer, impatient with fluff"
            rules = [
                "explain the mechanics",
                "call out vagueness",
                "keep it grounded in real impact",
            ]
        elif any(term in text for term in STORY_TYPES["policy"]):
            style = "analytic, blunt, no-nonsense"
            rules = [
                "challenge hype",
                "distill the risk",
                "use direct language",
            ]
        else:
            style = "explainer with edge"
            rules = [
                "keep it punchy",
                "make it feel personal",
                "avoid dry textbook language",
            ]
        return {
            "style": style,
            "rules": rules,
        }

    def _select_tone(self, story: NewsItem, angle: str, variation: str) -> str:
        if angle == "hot_take":
            return "provocative"
        if angle == "threat_to_jobs":
            return "urgent"
        if angle == "technical_breakthrough":
            return "informed"
        if variation == "controversial":
            return "provocative"
        if variation == "storytelling":
            return "personal"
        if variation == "fast_news":
            return "direct"
        return "curious"

    def _generate_hook_variants(self, story: NewsItem, angle: str) -> list[str]:
        texts = []
        base = story.title.rstrip(".")
        conflict = self._conflict_phrase(story)
        if angle == "technical_breakthrough":
            texts = [
                f"Everyone is excited about {base}, but the real leap is the hidden tradeoff",
                f"This breakthrough sounds huge — until you see the data behind it",
                f"No one is talking about the one detail that changes everything",
            ]
        elif angle == "industry_impact":
            texts = [
                f"This move could reshape the AI industry faster than anyone predicted",
                f"How {base} changes the tools developers will use every day",
                f"The biggest industry shift in AI this month is hiding in plain sight",
            ]
        elif angle == "threat_to_jobs":
            texts = [
                f"This tech might replace your job sooner than you think",
                f"One announcement could make entire teams obsolete",
                f"If you're building AI tools, this is the one signal you can't ignore",
            ]
        elif angle == "overhyped_vs_reality":
            texts = [
                f"Everyone is hyping this, but something about it feels off",
                f"This looks like a breakthrough — until you test the fine print",
                f"No one is talking about the one weak point in this story",
            ]
        else:
            texts = [
                f"Here's what {base} really means for your work tomorrow",
                f"The one practical lesson from {base} every AI team should know",
                f"What this announcement means for your next project",
            ]

        variants = [self._normalize_hook(t) for t in texts]
        return variants[:3]

    def _rank_hooks(self, hooks: list[str], story: NewsItem) -> str:
        scored = []
        for hook in hooks:
            curiosity = self._hook_curiosity(hook)
            negativity = self._hook_negativity(hook)
            surprise = self._hook_surprise(hook)
            score = curiosity * 0.45 + negativity * 0.3 + surprise * 0.25
            scored.append((score, hook))
        chosen = max(scored, key=lambda item: item[0])[1]
        logger.debug(
            "Hook scores for %s: %s",
            story.title,
            {hook: round(score, 2) for score, hook in scored},
        )
        return chosen

    def _hook_curiosity(self, hook: str) -> float:
        return min(1.0, len(re.findall(r"\b(what|why|how|could|might|if|think)\b", hook.lower())) * 0.3 + 0.2)

    def _hook_negativity(self, hook: str) -> float:
        negatives = len(re.findall(r"\b(but|unless|if|yet|still|not|avoid|problem|fail|issue)\b", hook.lower()))
        return min(1.0, negatives * 0.25)

    def _hook_surprise(self, hook: str) -> float:
        return min(1.0, len(re.findall(r"\b(never|nobody|no one|surprising|unexpected|hidden|secret|quietly)\b", hook.lower())) * 0.3 + 0.1)

    def _decide_format(self, story: NewsItem, angle: str, variation: str) -> dict[str, Any]:
        if self._is_breaking(story):
            return {"format": "quick_hit", "scenes": 5, "pacing": "fast"}
        complexity = self._complexity_score(story)
        controversy = self._controversy_score(story)
        
        # Check format performance history
        base_format = None
        if complexity > 0.7:
            base_format = "deep_dive"
        elif controversy > 0.55:
            base_format = "hot_take"
        elif angle == "technical_breakthrough":
            base_format = "deep_dive"
        else:
            base_format = "quick_hit"
        
        # Boost format based on past performance
        fmt_perf = self.feedback_analyzer.get_format_performance(base_format)
        if fmt_perf and fmt_perf.get("avg_hook_score", 0) > 0.7:
            logger.debug(f"Format {base_format} performing well (score: {fmt_perf['avg_hook_score']})")
        
        if base_format == "deep_dive":
            return {"format": base_format, "scenes": 10, "pacing": "slow"}
        elif base_format == "hot_take":
            return {"format": base_format, "scenes": 4, "pacing": "fast"}
        elif variation == "fast_news":
            return {"format": "quick_hit", "scenes": 5, "pacing": "fast"}
        elif variation == "storytelling":
            return {"format": "deep_dive", "scenes": 9, "pacing": "measured"}
        return {"format": "quick_hit", "scenes": 6, "pacing": "medium"}

    def _director_bridge(self, format_data: dict[str, Any], angle: str, variation: str) -> list[dict[str, Any]]:
        scene_plan = FORMAT_RULES.get(format_data["format"], FORMAT_RULES["quick_hit"])["scene_plan"]
        plan = [dict(step, angle=angle) for step in scene_plan]
        if variation == "storytelling" and format_data["format"] == "deep_dive":
            plan.insert(2, {"type": "story", "goal": "show the human angle", "style": "narrative", "angle": angle})
        if variation == "controversial" and format_data["format"] == "hot_take":
            plan.append({"type": "challenge", "goal": "call out the weak spot", "style": "sharp", "angle": angle})
        return plan

    def _choose_global_style(self, channel_config: dict[str, Any], platform: str) -> dict[str, str]:
        persona = channel_config.get("persona", "seasoned AI editor")
        if platform == "shorts":
            return {"persona": persona, "energy_level": "very high", "pacing": "dynamic"}
        if platform == "youtube_long":
            return {"persona": persona, "energy_level": "high", "pacing": "purposeful"}
        return {"persona": persona, "energy_level": "medium", "pacing": "balanced"}

    def _complexity_score(self, story: NewsItem) -> float:
        text = f"{story.title} {story.summary}".lower()
        terms = ["benchmark", "architectur", "parameters", "fine-tun", "training", "dataset", "latency", "throughput", "model size"]
        hits = sum(1 for term in terms if term in text)
        return min(1.0, hits / 4)

    def _conflict_phrase(self, story: NewsItem) -> str:
        text = f"{story.title} {story.summary}".lower()
        matches = [term for term in CONFLICT_TERMS if term in text]
        if matches:
            return f"the {matches[0]} at the heart of this story"
        return "the one detail that changes the story"

    def _tokens(self, value: str) -> set[str]:
        return {token for token in re.findall(r"\w{3,}", value.lower())}

    def _jaccard(self, a: set[str], b: set[str]) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    def _normalize_hook(self, text: str) -> str:
        result = re.sub(r"\s+"," ", text).strip()
        return result if len(result.split()) <= 12 else " ".join(result.split()[:12])


def editorial_plan_to_prompt(plan: EditorialPlan) -> str:
    lines = ["=== EDITORIAL PLAN ==="]
    lines.append("Global style:")
    lines.append(f"persona: {plan.global_style['persona']}")
    lines.append(f"energy_level: {plan.global_style['energy_level']}")
    lines.append(f"pacing: {plan.global_style['pacing']}")
    lines.append("")
    lines.append("Selected stories:")
    for story in plan.selected_stories:
        lines.append(f"- {story['title']} ({story['source']})")
    lines.append("")
    for item in plan.editorial_plan:
        lines.append(f"Story ID: {item['story_id']}")
        lines.append(f"  angle: {item['angle']}")
        lines.append(f"  conflict: {item.get('conflict', 'make the tension explicit')}")
        persona = item['persona']
        if isinstance(persona, dict):
            lines.append(f"  persona style: {persona.get('style')}")
            lines.append("  persona rules:")
            for rule in persona.get('rules', []):
                lines.append(f"    - {rule}")
        else:
            lines.append(f"  persona: {persona}")
        lines.append(f"  tone: {item['tone']}")
        lines.append(f"  hook: {item['hook']}")
        lines.append(f"  format: {item['format']}")
        lines.append(f"  scene_plan: {item['scene_plan']}")
        lines.append("")
    return "\n".join(lines)
