"""Drop Predictor — pre-emptive retention risk scoring per scene.

Instead of learning from drops *after* a video is published, this module
predicts P(drop at scene_i) *before* generation, enabling pre-emptive
correction during the script-to-video pipeline.

Architecture
------------
                   Script / Scene objects
                           ↓
              FeatureEncoder.from_scene()
                           ↓
                     SceneFeatures
                           ↓
              DropPredictor.predict()
            ┌──────────────┴──────────────┐
            │ sklearn                     │ no sklearn
            │ RandomForestClassifier      │ _rule_based_predict()
            └──────────────┬─────────────┘
                           ↓
                    DropPrediction
                    (prob, risk, factors)
                           ↓
           apply_preemptive_corrections()
                           ↓
                 Modified Scene objects

Feature vector
--------------
13 features extracted from each Scene:
    duration, position, has_hook, has_avatar, complexity,
    sentence_length, is_image, is_avatar, is_diagram,
    is_text_overlay, is_explain, is_hook, is_transition

Rule-based heuristics (no sklearn)
-----------------------------------
Strong risk signals (additive, clamped to [0, 1]):
    intent==explain AND duration>45s  → +0.35  (long lecture)
    has_hook==False                   → +0.20  (nothing to hold attention)
    scene_type==image AND no hook     → +0.15  (static boring visual)
    complexity > 0.7                  → +0.15  (too dense narration)
    position <= 1                     → +0.10  (early scenes, high stakes)
    sentence_length > 35 words        → +0.10  (wall of text)
    intent==transition                → +0.05  (pacing break → viewers leave)

Pre-emptive corrections
-----------------------
When predicted risk >= HIGH_RISK_THRESHOLD (0.6):
    intent==explain → SHORTEN_AND_ADD_HOOK + duration *= 0.7
    scene_type==image → REPLACE_WITH_AVATAR
    fallback → ADD_MICRO_HOOK (prepend "[HOOK] " to visual_prompt)

Training pipeline
-----------------
After video is published and analytics arrive::

    dataset = TrainingDataset(data_dir)
    dataset.build_from_video(scenes, retention_data, scene_map)
    predictor.train(dataset)

sklearn integration
-------------------
When sklearn is available the predictor trains a RandomForestClassifier
(n_estimators=50, random_state=42) and persists it via joblib/pickle.
When sklearn is absent the rule-based predictor is used automatically.
"""
from __future__ import annotations

import json
import pickle
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings

# Optional sklearn import — rule-based fallback when absent
try:
    from sklearn.ensemble import RandomForestClassifier as _RFC
    _SKLEARN_AVAILABLE = True
except ImportError:
    _RFC = None                    # type: ignore[assignment,misc]
    _SKLEARN_AVAILABLE = False


# ── Constants ─────────────────────────────────────────────────────────────────

HIGH_RISK_THRESHOLD   = 0.60
MEDIUM_RISK_THRESHOLD = 0.40
DROP_LABEL_THRESHOLD  = 0.15   # retention delta that counts as a "drop" label

_HOOK_PHRASES = frozenset([
    "but", "wait", "here's", "the truth", "nobody", "you won't", "actually",
    "shocking", "secret", "warning", "breaking",
])


# ── Data types ────────────────────────────────────────────────────────────────

class DropRisk(str, Enum):
    LOW    = "low"
    MEDIUM = "medium"
    HIGH   = "high"


@dataclass
class SceneFeatures:
    """Extracted features for one scene — ready for encoding or rule scoring."""
    scene_type:      str   = "image"
    intent:          str   = "explain"
    duration:        float = 30.0
    position:        int   = 0
    has_hook:        bool  = False
    has_avatar:      bool  = False
    sentence_length: int   = 20
    complexity:      float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_type":      self.scene_type,
            "intent":          self.intent,
            "duration":        self.duration,
            "position":        self.position,
            "has_hook":        self.has_hook,
            "has_avatar":      self.has_avatar,
            "sentence_length": self.sentence_length,
            "complexity":      self.complexity,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SceneFeatures":
        return cls(
            scene_type      = d.get("scene_type",      "image"),
            intent          = d.get("intent",          "explain"),
            duration        = float(d.get("duration",        30.0)),
            position        = int(d.get("position",          0)),
            has_hook        = bool(d.get("has_hook",         False)),
            has_avatar      = bool(d.get("has_avatar",       False)),
            sentence_length = int(d.get("sentence_length",   20)),
            complexity      = float(d.get("complexity",      0.5)),
        )


@dataclass
class DropPrediction:
    """Predicted drop risk for one scene."""
    scene_idx:   int
    probability: float
    risk:        DropRisk
    top_factors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_idx":   self.scene_idx,
            "probability": round(self.probability, 4),
            "risk":        self.risk.value,
            "top_factors": self.top_factors,
        }


# ── Feature encoding ──────────────────────────────────────────────────────────

FEATURE_NAMES = [
    "duration", "position", "has_hook", "has_avatar", "complexity",
    "sentence_length", "is_image", "is_avatar", "is_diagram",
    "is_text_overlay", "is_explain", "is_hook_intent", "is_transition",
]


class FeatureEncoder:
    """Converts Scene objects or SceneFeatures to numeric feature vectors."""

    def encode(self, features: SceneFeatures) -> list[float]:
        """Return a 13-element numeric vector for sklearn."""
        return [
            features.duration,
            float(features.position),
            1.0 if features.has_hook   else 0.0,
            1.0 if features.has_avatar else 0.0,
            features.complexity,
            float(features.sentence_length),
            1.0 if features.scene_type == "image"        else 0.0,
            1.0 if features.scene_type == "avatar"       else 0.0,
            1.0 if features.scene_type == "diagram"      else 0.0,
            1.0 if features.scene_type == "text_overlay" else 0.0,
            1.0 if features.intent     == "explain"      else 0.0,
            1.0 if features.intent     == "hook"         else 0.0,
            1.0 if features.intent     == "transition"   else 0.0,
        ]

    def from_scene(self, scene: Any, position: int | None = None) -> SceneFeatures:
        """Extract SceneFeatures from a Scene (src.script_generator.Scene) object."""
        narration       = getattr(scene, "narration", "") or ""
        words           = narration.split()
        sentence_length = len(words)
        duration        = float(getattr(scene, "duration_sec", 30) or 30)
        complexity      = self._compute_complexity(words, duration)

        scene_type = getattr(scene, "scene_type",   "image")
        intent     = getattr(scene, "scene_intent", "explain")
        scene_idx  = getattr(scene, "idx",          0)
        pos        = position if position is not None else scene_idx

        has_avatar = scene_type == "avatar"
        has_hook   = (
            intent == "hook"
            or any(p in narration.lower() for p in _HOOK_PHRASES)
        )

        return SceneFeatures(
            scene_type      = scene_type,
            intent          = intent,
            duration        = duration,
            position        = pos,
            has_hook        = has_hook,
            has_avatar      = has_avatar,
            sentence_length = sentence_length,
            complexity      = complexity,
        )

    @staticmethod
    def _compute_complexity(words: list[str], duration: float) -> float:
        """Normalised narration density: words-per-second capped at 1.0."""
        if not words or duration <= 0:
            return 0.0
        wps = len(words) / duration
        return round(min(1.0, wps / 3.0), 4)   # 3 wps = maximum complexity


# ── Rule-based predictor ──────────────────────────────────────────────────────

def _rule_based_predict(features: SceneFeatures) -> tuple[float, list[str]]:
    """Return (probability, top_factors) using additive heuristic scoring.

    Used when sklearn is unavailable or when the model has not been trained.
    """
    score: float      = 0.0
    factors: list[str] = []

    if features.intent == "explain" and features.duration > 45:
        score += 0.35
        factors.append("long_explain_scene")

    if not features.has_hook:
        score += 0.25
        factors.append("no_hook")

    if features.scene_type == "image" and not features.has_hook:
        score += 0.20
        factors.append("static_image_no_hook")

    if features.complexity > 0.7:
        score += 0.15
        factors.append("high_complexity")

    if features.position <= 1:
        score += 0.10
        factors.append("early_position")

    if features.sentence_length > 35:
        score += 0.10
        factors.append("long_sentence")

    if features.intent == "transition":
        score += 0.05
        factors.append("transition_scene")

    return min(1.0, round(score, 4)), factors


def _risk_level(probability: float) -> DropRisk:
    if probability >= HIGH_RISK_THRESHOLD:
        return DropRisk.HIGH
    if probability >= MEDIUM_RISK_THRESHOLD:
        return DropRisk.MEDIUM
    return DropRisk.LOW


# ── Training dataset ──────────────────────────────────────────────────────────

class TrainingDataset:
    """JSON-backed collection of (features, label) training examples.

    label = 1 if a retention drop > DROP_LABEL_THRESHOLD was detected
            at the scene's timestamp, 0 otherwise.
    """

    def __init__(self, data_dir: Path | None = None) -> None:
        self._path    = (data_dir or settings.data_dir) / "drop_predictor_dataset.json"
        self._records = self._load()

    def _load(self) -> list[dict[str, Any]]:
        try:
            if self._path.exists():
                return json.loads(self._path.read_text())
        except Exception as exc:
            logger.warning(f"TrainingDataset: could not load ({exc}), starting fresh")
        return []

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._records, indent=2))

    def add_example(self, features: SceneFeatures, label: int) -> None:
        """Add one (features, label) pair.  label must be 0 or 1."""
        self._records.append({"features": features.to_dict(), "label": int(label)})
        self._save()

    def build_from_video(
        self,
        scenes:         list[Any],
        retention_data: dict[str, Any],
        scene_map:      dict[int, dict[str, Any]],
        encoder:        FeatureEncoder | None = None,
    ) -> int:
        """Derive training examples from a completed video's retention curve.

        For each scene, label = 1 if any mapped timestamp showed a retention
        drop > DROP_LABEL_THRESHOLD, else 0.

        Returns the number of examples added.
        """
        enc   = encoder or FeatureEncoder()
        curve = retention_data.get("curve", [])
        if not curve:
            return 0

        # Build drop set and covered set from scene_map
        drop_scene_idxs:    set[int] = set()
        covered_scene_idxs: set[int] = set()   # scenes we have timestamp data for
        for i in range(1, len(curve)):
            info = scene_map.get(i)
            if info is None:
                continue
            scene_idx = info.get("scene_idx", i)
            covered_scene_idxs.add(scene_idx)
            delta = curve[i - 1] - curve[i]
            if delta > DROP_LABEL_THRESHOLD:
                drop_scene_idxs.add(scene_idx)

        added = 0
        for pos, scene in enumerate(scenes):
            scene_idx = getattr(scene, "idx", pos)
            if scene_idx not in covered_scene_idxs:
                # No timestamp data for this scene → skip rather than mislabel
                continue
            label = 1 if scene_idx in drop_scene_idxs else 0
            self.add_example(enc.from_scene(scene, position=pos), label)
            added += 1

        logger.info(
            f"TrainingDataset: added {added} examples "
            f"({len(drop_scene_idxs)} positive labels)"
        )
        return added

    def get_xy(self) -> tuple[list[list[float]], list[int]]:
        """Return (X, y) for sklearn model fitting."""
        enc = FeatureEncoder()
        X   = [enc.encode(SceneFeatures.from_dict(r["features"])) for r in self._records]
        y   = [r["label"] for r in self._records]
        return X, y

    def __len__(self) -> int:
        return len(self._records)

    def all_records(self) -> list[dict[str, Any]]:
        return list(self._records)


# ── DropPredictor ──────────────────────────────────────────────────────────────

class DropPredictor:
    """Predicts P(drop) for each scene before video generation.

    When sklearn is available and training data exists, uses a
    RandomForestClassifier.  Falls back to rule-based heuristics otherwise.

    Typical lifecycle::

        predictor = DropPredictor(data_dir=run_dir)

        # Before generating video:
        predictions = predictor.predict_scenes(script.scenes)
        corrections = apply_preemptive_corrections(script.scenes, predictions)
        adjustments = suggest_strategy_adjustments(predictions)

        # After video is published and analytics arrive:
        dataset = predictor.dataset
        dataset.build_from_video(script.scenes, retention_data, scene_map)
        predictor.train()
    """

    _MODEL_FILENAME = "drop_predictor_model.pkl"
    MIN_TRAIN_SIZE  = 10   # minimum examples before sklearn model is used

    def __init__(
        self,
        data_dir: Path | None = None,
        model:    Any         = None,    # injectable for testing
    ) -> None:
        self._data_dir = data_dir or settings.data_dir
        self._encoder  = FeatureEncoder()
        self.dataset   = TrainingDataset(data_dir)
        self._model    = model
        if self._model is None:
            self._model = self._load_model()

    # ── Prediction ─────────────────────────────────────────────────────────────

    def predict(self, features: SceneFeatures, scene_idx: int = 0) -> DropPrediction:
        """Return a DropPrediction for a single scene's features."""
        if self._model is not None and _SKLEARN_AVAILABLE:
            prob    = self._sklearn_predict(features)
            factors = self._sklearn_factors(features)
        else:
            prob, factors = _rule_based_predict(features)

        return DropPrediction(
            scene_idx   = scene_idx,
            probability = prob,
            risk        = _risk_level(prob),
            top_factors = factors,
        )

    def predict_scenes(self, scenes: list[Any]) -> list[DropPrediction]:
        """Predict drop probability for every scene in a script.

        Returns predictions in the same order as *scenes*.
        """
        predictions = []
        for pos, scene in enumerate(scenes):
            feats = self._encoder.from_scene(scene, position=pos)
            pred  = self.predict(feats, scene_idx=getattr(scene, "idx", pos))
            predictions.append(pred)
            if pred.risk != DropRisk.LOW:
                logger.info(
                    f"DropPredictor: scene {pred.scene_idx} risk={pred.risk.value} "
                    f"prob={pred.probability:.2f} factors={pred.top_factors}"
                )
        return predictions

    # ── Training ───────────────────────────────────────────────────────────────

    def train(self, dataset: TrainingDataset | None = None) -> bool:
        """Fit sklearn model from accumulated examples.

        Returns True if training succeeded, False otherwise (e.g. sklearn
        unavailable or insufficient data).
        """
        if not _SKLEARN_AVAILABLE:
            logger.warning("DropPredictor.train: sklearn not available, skipping")
            return False

        ds = dataset or self.dataset
        if len(ds) < self.MIN_TRAIN_SIZE:
            logger.warning(
                f"DropPredictor.train: only {len(ds)} examples "
                f"(need {self.MIN_TRAIN_SIZE}), skipping"
            )
            return False

        X, y = ds.get_xy()
        self._model = _RFC(n_estimators=50, random_state=42)
        self._model.fit(X, y)
        self._save_model()
        logger.info(f"DropPredictor: trained on {len(X)} examples")
        return True

    # ── Sklearn helpers ────────────────────────────────────────────────────────

    def _sklearn_predict(self, features: SceneFeatures) -> float:
        vec   = self._encoder.encode(features)
        proba = self._model.predict_proba([vec])[0]
        # proba shape: (n_classes,); class 1 = drop
        classes = list(self._model.classes_)
        idx_1   = classes.index(1) if 1 in classes else -1
        return float(proba[idx_1]) if idx_1 >= 0 else 0.0

    def _sklearn_factors(self, features: SceneFeatures) -> list[str]:
        """Return names of the top-3 features by importance for this prediction."""
        if not hasattr(self._model, "feature_importances_"):
            return []
        importances = self._model.feature_importances_
        top_idxs    = sorted(range(len(importances)), key=lambda i: importances[i], reverse=True)
        return [FEATURE_NAMES[i] for i in top_idxs[:3] if i < len(FEATURE_NAMES)]

    # ── Model persistence ──────────────────────────────────────────────────────

    def _model_path(self) -> Path:
        return self._data_dir / self._MODEL_FILENAME

    def _save_model(self) -> None:
        if self._model is None:
            return
        try:
            path = self._model_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump(self._model, f)
            logger.debug(f"DropPredictor: model saved to {path}")
        except Exception as exc:
            logger.warning(f"DropPredictor: could not save model ({exc})")

    def _load_model(self) -> Any:
        if not _SKLEARN_AVAILABLE:
            return None
        path = self._model_path()
        if not path.exists():
            return None
        try:
            with open(path, "rb") as f:
                model = pickle.load(f)
            logger.info(f"DropPredictor: loaded model from {path}")
            return model
        except Exception as exc:
            logger.warning(f"DropPredictor: could not load model ({exc}), will retrain")
            return None


# ── Pre-emptive corrections ───────────────────────────────────────────────────

_HOOK_PREFIX = "[HOOK] "


def apply_preemptive_corrections(
    scenes:      list[Any],
    predictions: list[DropPrediction],
    threshold:   float = HIGH_RISK_THRESHOLD,
) -> list[dict[str, Any]]:
    """Modify high-risk Scene objects in place before video generation.

    Returns a list of applied-correction records (for logging / learning).
    """
    scene_by_idx = {getattr(s, "idx", i): s for i, s in enumerate(scenes)}
    applied: list[dict[str, Any]] = []

    for pred in predictions:
        if pred.probability < threshold:
            continue

        scene = scene_by_idx.get(pred.scene_idx)
        if scene is None:
            continue

        fix = _choose_preemptive_fix(scene)
        _apply_preemptive_fix(scene, fix)
        applied.append({
            "scene_idx":   pred.scene_idx,
            "fix":         fix,
            "probability": round(pred.probability, 4),
            "risk":        pred.risk.value,
        })
        logger.info(
            f"DropPredictor: pre-emptive fix {fix!r} on scene {pred.scene_idx} "
            f"(prob={pred.probability:.2f})"
        )

    return applied


def _choose_preemptive_fix(scene: Any) -> str:
    intent     = getattr(scene, "scene_intent", "explain")
    scene_type = getattr(scene, "scene_type",   "image")

    if intent == "explain":
        return "shorten_and_add_hook"
    if scene_type == "image":
        return "replace_with_avatar"
    return "add_micro_hook"


def _apply_preemptive_fix(scene: Any, fix: str) -> None:
    dur = getattr(scene, "duration_sec", 0)

    if fix == "shorten_and_add_hook":
        if dur > 0:
            scene.duration_sec = int(dur * 0.7)
        scene.visual_prompt = _HOOK_PREFIX + getattr(scene, "visual_prompt", "")
    elif fix == "replace_with_avatar":
        scene.scene_type = "avatar"
    elif fix == "add_micro_hook":
        scene.visual_prompt = _HOOK_PREFIX + getattr(scene, "visual_prompt", "")


# ── Strategy adjustments ──────────────────────────────────────────────────────

def suggest_strategy_adjustments(predictions: list[DropPrediction]) -> dict[str, Any]:
    """Derive global strategy hints for DecisionEngineV2 from risk predictions.

    Rules:
    - Any high-risk scene in position 0–2  → pace="fast", +0.20 hook_aggressiveness
    - >= 3 high-risk scenes total           → +0.10 hook_aggressiveness
    - High-risk scene at position 0         → force_avatar_intro=True
    - No high-risk scenes                  → empty dict
    """
    high_risk = [p for p in predictions if p.risk == DropRisk.HIGH]
    if not high_risk:
        return {}

    adjustments: dict[str, Any] = {}
    hook_delta = 0.0

    early_high = [p for p in high_risk if p.scene_idx <= 2]
    if early_high:
        adjustments["pace"] = "fast"
        hook_delta          += 0.20

    if any(p.scene_idx == 0 for p in high_risk):
        adjustments["force_avatar_intro"] = True

    if len(high_risk) >= 3:
        hook_delta += 0.10

    if hook_delta:
        adjustments["hook_aggressiveness_delta"] = round(hook_delta, 2)

    adjustments["high_risk_scene_count"]  = len(high_risk)
    adjustments["early_high_risk_count"] = len(early_high)

    logger.info(f"DropPredictor: strategy adjustments → {adjustments}")
    return adjustments
