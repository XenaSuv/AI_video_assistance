# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased] — Architecture & Quality Sprint

### Added
- `_run_language_variant()` in `src/language_variant.py` refactored to run two pairs of
  steps concurrently via `asyncio.gather + loop.run_in_executor`:
  - Steps 3+4: `generate_subtitles` and `assemble_video` overlap (both need only the audio
    produced in step 2; neither writes to paths the other reads)
  - Steps 5+6: `build_short` and `generate_thumbnail` overlap (both need only `long_video`
    produced in step 4)
  - Eliminates two sequential blocking-I/O waits from the RU variant wall-clock time
- `_assemble_one_scene()` in `src/video_generator.py`: per-scene assembly extracted from
  `assemble_video` loop body (A/V merge, title overlay, chyron); safe for concurrent
  execution because every intermediate file is uniquely named by scene index
- `_async_assemble_scenes()` in `src/video_generator.py`: assembles all scenes concurrently
  via `asyncio.gather + loop.run_in_executor`; replaces the sequential `for scene in script.scenes`
  loop in `assemble_video`, cutting wall-clock assembly time from O(N) to O(1) per
  scene-count
- `_step_shorts_and_ru()` in `PipelineOrchestrator`: Shorts experiments and RU upload
  now run concurrently via `asyncio.gather + loop.run_in_executor`, mirroring the
  `_voice_and_images` pattern already used in `_MediaBuilder`
- `scripts/check_new_module_coverage.py`: per-file coverage gate (≥80%) for newly written
  modules; runs as a separate CI step after pytest
- `src/config_validator.py`: startup validator expanded from 20 to 45 checked fields,
  covering all YouTube/TikTok/HeyGen/D-ID/budget/music settings
- SQLite schema versioning in `src/deduplicator.py` via `PRAGMA user_version`; schema
  upgrades apply automatically on first open after a change
- `HEYGEN_ENABLED` flag: DID_API_KEY check now correctly skipped when HeyGen is the
  active presenter (PR #115)

### Changed
- `src/pipeline_orchestrator.py` refactored: 43 → 24 imports; orchestration split into
  three focused classes — `_ScriptStep`, `_MediaBuilder`, `_PublishStep`
- `pyproject.toml`: `mypy --strict` now enforced globally; `src/media_builder.py`,
  `src/publish_step.py`, `src/pipeline_orchestrator.py`, `src/script_step.py`,
  `src/deferred_feedback.py`, `src/config_validator.py` all pass strict checks;
  legacy modules isolated in `[[tool.mypy.overrides]]` with `ignore_errors = true`
- CI pipeline now has four steps: ruff → mypy → pytest (≥75% overall) → coverage gate
  (≥80% new modules)
- `src/media_builder.py` and `src/publish_step.py` promoted out of `ignore_errors`
  override; bare `list`/`dict` annotations tightened to `list[Any]`/`dict[str, Any]`
- `src/shorts_generator.py` formally deprecated; `src/shorts_engine_v2.py`,
  `src/shorts_pipeline.py`, `src/shorts_experiment_engine.py` declared primary

### Fixed
- `ElevenLabs convert()` called with `VoiceSettings` object instead of raw dict
- All 23 bare `except:` clauses now log via `logger.debug/warning` (were silent)
- `_setup_logging` import paths corrected in `topic_main.py` and `breaking_main.py`

### Tests
- `test_publish_step.py`: 78% → 99% — added `TestRunShortsExperimentsExecution` (7 tests)
  covering main experiment loop, and `TestUploadEnOptionalEngines` (5 tests) covering
  all five post-upload optional engine except-handlers
- `test_language_variant.py`: 8 → 13 tests; added `TestRunLanguageVariantConcurrency` (5
  tests) verifying both subtitles+video run concurrently, subtitle failure does not block
  video, thumbnail always runs, and subtitle_path is forwarded to publish_episode
- `test_video_generator.py`: 13% → 97% — 45 smoke tests; `_FfmpegMocks` context manager
  patching all `src.ffmpeg_utils.*` functions; added `TestAssembleOneScene` (8 tests)
  and `TestAsyncAssembleScenes` (3 tests) covering the new parallel assembly functions
- `test_script_step.py`: 15% → 98% — 42 tests covering `__init__`, `run()` cached/fresh
  paths, `_humanize_script`, `_apply_micro_hooks`, `_apply_thompson_hook`, `_run_new()`
- `test_deferred_feedback.py`: new — 30 tests, 100% coverage of `_DeferredFeedbackCollector`
- `test_pipeline_orchestrator.py`: +4 tests for `_step_shorts_and_ru` concurrency
- `test_weekly_script_generator.py`: +48 tests
- `test_shorts_pipeline.py`: +47 tests
- `test_media_builder.py` / `test_publish_step.py` (initial): +56 tests for `_MediaBuilder`
  and `_PublishStep` after extraction from orchestrator
- Total test suite: 4 034 tests, 88% overall coverage

---

## [0.5.0] — 2026-05-17 — Observability & Infrastructure

### Added
- `src/live_state.py`: real-time pipeline state written to JSON; tracks step progress,
  cost, and events consumable by external dashboards
- `src/json_log_sink.py`: JSONL structured log sink for log aggregator integration
  (Datadog, Loki, CloudWatch)
- `src/budget_guard.py`: daily/monthly spend limits; triggers Slack alert when exceeded
- `src/latency_tracker.py`: records step timings, computes p95, detects slow steps;
  Slack notification sent when any step exceeds its threshold
- `src/pipeline_observer.py`: per-step timing and structured pipeline trace
- `src/config_validator.py` (initial): startup validation with `ConfigurationError` on
  missing required settings (20 fields)
- `src/checkpoint.py`: resumable pipeline execution — reruns skip already-completed steps
  and reuse cached artifacts from `output/YYYY-MM-DD/`
- `test_pipeline_orchestrator.py`: 34 smoke tests for PipelineOrchestrator control flow

### Changed
- `src/main.py` refactored into `PipelineOrchestrator` class (28.5K → focused orchestration)
- HeyGen per-persona avatar IDs moved from hardcoded values into `config/settings.py`
- FFmpeg timeout constants moved into `settings`; configurable via env vars
- `openai<3`, `elevenlabs<2` upper bounds pinned to prevent silent breaking changes
- Dashboard (`src/dashboard.py`) connected to real performance data and cost ledger

### Fixed
- HeyGen asset upload endpoint corrected to `upload.heygen.com/v1/asset`
- HeyGen hook clip avatar routing now goes through `AvatarEngine` for persona awareness
- FFmpeg `concat()` stream-copy first, `veryfast` re-encode fallback to prevent CI timeout
- JSON-format token handled correctly in pickle migration step
- VoiceSettings passed as object (not dict) to ElevenLabs `convert()`

---

## [0.4.0] — 2026-05-05 — Learning System Completion

### Added
- `src/thompson_bandit.py`: Beta-posterior Thompson Sampling for variant selection;
  state persisted to JSON between runs
- `src/scene_bandit.py`: multi-armed bandit over scene policies; uses Thompson Sampling
- `src/bandit_engine.py`: UCB1 multi-armed bandit for continuous variant optimisation
- `src/ab_testing_engine.py`: sequential A/B testing with CTR × retention scoring
- `src/drop_predictor.py`: pre-emptive retention risk scoring per scene before publish
- `src/retention_correction_engine.py`: drop-point detection and scene-level fix suggestions
- `src/decision_engine_v2.py`: channel-level strategy driving scene composition;
  performance windows, exploration decay, mode selection
- `src/decision_engine_v3.py`: unified signals pipeline — cross-learning + context analysis
- `src/sequence_learning_engine.py`: learns scene order → retention patterns
- `src/cross_learning_engine.py`: combination-level (angle × format) pattern learning
- `src/context_analyzer.py`: content-aware strategy layer; adapts to topic category
- `src/packaging_engine.py`: synchronises hook / title / thumbnail from editorial plan
- `src/system_orchestrator.py`: single control point for coordinating all pipeline types
- `src/dashboard.py`: real-time monitoring dashboard
- `src/debug_dashboard.py`: rich run-history logging

### Changed
- `DecisionEngine` now uses exponential time decay — recent videos count more than old ones
- `SceneVarietyEngine` v2 wired to `DecisionEngineV2` + feedback memory

---

## [0.3.0] — 2026-05-03 — Engagement & Content Engines

### Added
- `src/pacing_auditor.py`: detects flat zones in topic scripts; inserts pacing annotations
- `src/comment_magnet_agent.py`: generates comment-bait lines to boost comment velocity
- `src/open_loop_agent.py`: viewer retention open-loop questions inserted into narration
- `src/screenshot_capturer.py`: dynamic URL screenshot support for topic segments
- `src/scraper_cache/`: per-day HTTP response cache to avoid redundant API calls
- `src/topic_segment_generator.py`: topic deep-dive segment with auto-generated idea queue
- Topic deep-dive pipeline (`topic_main.py`): Tue/Thu 10:00 UTC GitHub Actions workflow
- `src/retry_utils.py`: exponential backoff for all external HTTP and OpenAI API calls
  (replaces 12 inline retry loops)
- `src/cost_tracker.py`: central CostLedger tracking LLM, TTS, and image API spend per run
- `tests/test_scraper.py`: 49 tests, full scraper coverage

### Changed
- `src/presenter.py`: wired into daily pipeline; `http_get/http_post` wrappers replace
  bare `requests` calls
- Background music settings (path, volume) passed through to GitHub Actions workflows
- Shared outro clip reused across all pipelines (reduced asset duplication)
- Daily, breaking, and digest content flows reshaped for editorial consistency

### Fixed
- Daily video audio and shorts assembly (missing audio gate added)
- Breaking news pipeline video + shorts assembly

---

## [0.2.0] — 2026-04-30 — Shorts & Multi-Platform

### Added
- `src/shorts_pipeline.py`: standalone Shorts pipeline; 25-second hook-first format
- `src/shorts_engine_v2.py`: editorial + beat-based hook generation
- `src/shorts_experiment_engine.py`: per-experiment upload + analytics collection
- `src/shorts_beat_engine.py`: 16-second beat execution planner
- `src/tiktok_uploader.py` / `src/tiktok_analytics.py`: TikTok upload and metrics
- `src/thumbnail_ab.py`: thumbnail A/B testing — multiple style variants, bandit selection
- `src/hook_mutation_engine.py`: evolves high-performing hooks into new variants
- `src/hook_pattern_engine.py`: pattern-based hook library with seed-controlled generation
- `src/hook_selector.py`: records hook usage; prevents repetition across videos
- Weekly digest pipeline (`digest_main.py`): Sunday 10:00 UTC
- Breaking news detector (`breaking_detector.py`): polls sources every 30 min

### Changed
- Breaking news pipeline now uses real video clips for Shorts
- CI triggered on push to main and on PRs; mypy + ruff + pytest-cov gate added

---

## [0.1.0] — 2026-04-29 — Initial Production System

### Added
- Daily AI news pipeline: scrape → deduplicate → script → voice → video → thumbnail → upload
- Russian language variant: translate → ru-TTS → reassemble → ru-YouTube
- `src/scraper.py`: arXiv, HuggingFace, HackerNews, official blog RSS aggregation
- `src/editorial_brain.py`: story ranking, angle selection, persona assignment
- `src/viral_selector.py`: engagement-potential scoring and filtering
- `src/script_generator.py`: structured `VideoScript` from editorial plan
- `src/humanizer_agent.py`: breaks rigid AI tone, adds rhythm and reactions
- `src/micro_hook_agent.py`: injects engagement hooks every 20–40 s
- `src/voice_generator.py`: ElevenLabs TTS with SSML pacing and 5-retry backoff
- `src/video_generator.py`: DALL-E + Ken Burns + background music + intro/outro assembly
- `src/image_generator.py`: DALL-E 3 primary, Pexels/Pixabay/Unsplash fallback
- `src/youtube_uploader.py`: YouTube Data API v3 upload + playlist + captions
- `src/feedback_analyzer.py`: retention curve parsing, drop-point detection, hook scoring
- `src/youtube_analytics.py`: real metrics from YouTube Data API v3
- `src/performance_tracker.py`: per-run video metadata → learning loop store
- `src/deduplicator.py`: SQLite 30-day story deduplication
- `src/quality_gate.py`: hard/soft content checks block publish on failure
- `config/settings.py`: frozen dataclass, all config from env vars
- Weekly tutorial pipelines: Claude / ChatGPT / Gemini (Mon/Wed/Fri)
- GitHub Actions: daily (Mon–Sat 08:00 UTC), weekly (Mon/Wed/Fri 09:00 UTC)
- Docker deployment support
- `src/analytics.py`: channel recommendations from performance history
- `src/title_optimizer.py`: GPT-powered title A/B suggestions
- `src/thumbnail_generator.py` / `src/thumbnail_mvp.py`: PIL-based thumbnail generation
- `src/infographic_generator.py`: scene-level infographic overlays
- `src/translator.py`: EN → RU script translation via OpenAI
- `src/subtitle_generator.py`: Whisper-based auto-subtitles (non-fatal)
