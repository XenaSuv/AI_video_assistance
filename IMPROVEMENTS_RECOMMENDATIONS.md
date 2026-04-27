# AI Video Assistance — Code Review & Optimization Recommendations

## Executive Summary

This is a well-structured, complex automation pipeline (~5,711 LOC across 29 modules) orchestrating daily news videos, weekly tutorials, and breaking news content to YouTube + TikTok. The project demonstrates good separation of concerns and caching strategies. However, there are opportunities for robustness, performance, and maintainability improvements.

**Risk Level: Medium** — Production system with external API dependencies (OpenAI, ElevenLabs, YouTube) and no test coverage.

---

## 🔴 Critical Issues (Fix First)

### 1. **No Test Coverage**
- **Issue**: Zero unit tests or integration tests. No `pytest` or `unittest` in dependencies.
- **Risk**: Breaking changes go undetected; regressions in scraper, script generation, or video assembly can fail silently.
- **Impact**: High — pipeline can break with no early warning.
- **Recommendation**:
  ```bash
  # Add to requirements.txt:
  pytest>=7.4.0
  pytest-asyncio>=0.21.0
  pytest-mock>=3.11.1
  ```
  - Create `tests/` directory with unit tests for core modules:
    - `tests/test_scraper.py` — validate NewsItem parsing, dedup logic
    - `tests/test_script_generator.py` — verify Scene/VideoScript structure
    - `tests/test_deduplicator.py` — SQLite state transitions
  - Target: 60%+ coverage on deterministic functions (avoid mocking API calls)
  - Run tests in CI/CD pipeline (add to GitHub Actions)

### 2. **Missing Error Recovery in Long-Running Pipelines**
- **Issue**: Video assembly (`video_generator.py`) can fail mid-way (e.g., MoviePy encoding) with no resumption mechanism.
- **Locations**: `main.py:249`, `digest_main.py:115`, `breaking_main.py:82`
- **Recommendation**: Implement **checkpoint-based recovery**:
  ```python
  # Create intermediate cache markers
  (run_dir / ".clip_gen_done").touch()  # after clips generated
  (run_dir / ".video_assembled").touch() # after video.mp4 written
  
  # Skip completed steps on retry
  if not (run_dir / ".clip_gen_done").exists():
      clip_paths_by_scene = {...}
  ```
  - Add explicit logging of which step is failing
  - Document in README how to retry individual pipelines

### 3. **Bare Exception Catching Without Context**
- **Issue**: Multiple instances of `except Exception as e:` without re-raising or proper context capture
- **Locations**: `scraper.py:104, 164, 213, 362`, `thumbnail_ab.py:62`, `breaking_main.py:160`
- **Example**:
  ```python
  except Exception as e:
      logger.error(f"Failed: {e}")  # Too generic — loses traceback
  ```
- **Fix**:
  ```python
  except Exception as e:
      logger.exception(f"Operation failed at {step_name}")  # Includes full traceback
      raise  # Re-raise so caller knows it failed
  ```

### 4. **PIL/Pillow Import Missing (Already Fixed)**
- **Status**: ✓ Fixed in `video_generator.py` line 23
- **Verify**: Ensure `Pillow>=10.4.0` is installed in CI/runner environment

---

## 🟠 High Priority Issues (Implement Next Sprint)

### 5. **Hardcoded Paths and Magic Numbers Scattered Across Code**
- **Issue**: Font paths, video dimensions, timeouts, retry counts are hardcoded.
- **Examples**:
  - `video_generator.py:29`: `_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"` (Linux-only)
  - `video_generator.py:30`: `VIDEO_W, VIDEO_H = 1280, 720` (magic number)
  - `image_generator.py:52`: `timeout=60` (hardcoded)
  - `video_generator.py:251`: `fps=24`, `codec="libx264"` (hardcoded)
- **Impact**: Hard to customize for different environments (Windows, macOS) or adjust for testing.
- **Recommendation**: Add to `config/settings.py`:
  ```python
  # Video encoding
  video_fps: int = int(_env("VIDEO_FPS", "24"))
  video_codec: str = _env("VIDEO_CODEC", "libx264")
  video_bitrate: str = _env("VIDEO_BITRATE", "6M")
  video_width: int = int(_env("VIDEO_WIDTH", "1280"))
  video_height: int = int(_env("VIDEO_HEIGHT", "720"))
  
  # Fonts
  font_path_bold: Path = ROOT / _env("FONT_PATH_BOLD", "fonts/DejaVuSans-Bold.ttf")
  
  # Timeouts
  request_timeout_sec: int = int(_env("REQUEST_TIMEOUT_SEC", "60"))
  ```

### 6. **API Rate Limiting Not Properly Handled**
- **Issue**: 
  - DALL-E retries exist (`image_generator.py:40`) but are limited to 4 attempts
  - ElevenLabs has no retry logic visible
  - OpenAI script generation has no rate limit handling
- **Risk**: Quota exhaustion → pipeline stops with unclear error
- **Recommendation**:
  ```python
  # Add to voice_generator.py & script_generator.py:
  from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential
  
  def _is_retryable_elevenlabs(exc: Exception) -> bool:
      return isinstance(exc, (RateLimitError, ConnectionError, TimeoutError))
  
  @retry(
      retry=retry_if_exception(_is_retryable_elevenlabs),
      wait=wait_exponential(multiplier=2, min=4, max=120),
      stop=stop_after_attempt(5),
      reraise=True,
  )
  def synthesize_scene(scene: Scene, ...):
      # existing code
  ```

### 7. **No Dependency Injection / Configuration Isolation**
- **Issue**: Settings hardcoded as module-level globals (`config.settings`)
- **Impact**: Hard to test with different configs, mock external services
- **Recommendation**: Extract settings into context objects:
  ```python
  # Instead of: from config import settings
  # Use:
  class PipelineContext:
      def __init__(self, settings: Settings, output_dir: Path):
          self.settings = settings
          self.output_dir = output_dir
  
  def run_pipeline(ctx: PipelineContext) -> dict:
      # Access via ctx.settings, ctx.output_dir
  ```

---

## 🟡 Medium Priority Issues (Nice-to-Have, Next 2-3 Sprints)

### 8. **Code Duplication Across Pipelines**
- **Shared Logic** (repeated in `main.py`, `weekly_main.py`, `breaking_main.py`, `digest_main.py`):
  ```
  1. _setup_logging(run_dir)
  2. _load_cached_script(path)
  3. Script generation → Voice → Video → Shorts → Thumbnail → Upload
  4. Russian variant handling (_run_language_variant)
  ```
- **Opportunity**: Extract into reusable base class:
  ```python
  class BasePipeline:
      def __init__(self, run_dir: Path, settings: Settings):
          self.run_dir = run_dir
          self.settings = settings
          self._setup_logging()
      
      def _setup_logging(self) -> None: ...
      def _synthesize_and_cache_voice(self, script: VideoScript) -> dict[int, Path]: ...
      def _build_and_cache_video(self, ...): ...
      def run(self) -> dict: ...
  
  class DailyPipeline(BasePipeline):
      def run(self) -> dict:
          # Scrape, deduplicate, then call super()
  ```
  - **Benefit**: Reduces ~1,200 LOC of duplication

### 9. **Weak Abstraction for Language Variants**
- **Issue**: Russian variant logic embedded in each pipeline via `_run_language_variant()` helper
- **Impact**: Hard to add new languages; code spread across multiple files
- **Recommendation**: Create `LanguageVariant` class:
  ```python
  class LanguageVariant:
      def __init__(self, 
                   lang_code: str,
                   youtube_secrets: Path,
                   youtube_token: Path,
                   elevenlabs_voice_id: str):
          self.lang_code = lang_code
          ...
      
      def translate_script(self, script: VideoScript) -> VideoScript: ...
      def synthesize_voice(self, script: VideoScript, out_dir: Path) -> dict: ...
      def publish(self, ...): ...
  
  # Usage:
  ru = LanguageVariant("ru", settings.ru_youtube_client_secrets, ...)
  ru_summary = ru.publish(...)
  ```

### 10. **No Structured Logging for Performance Monitoring**
- **Issue**: Using `loguru` but not capturing metrics (timing, API costs, retries)
- **Impact**: Can't debug bottlenecks or estimate costs
- **Recommendation**:
  ```python
  # Track call durations
  import time
  from contextlib import contextmanager
  
  @contextmanager
  def track_step(name: str, data: dict | None = None):
      start = time.time()
      try:
          yield
      finally:
          duration = time.time() - start
          logger.info(f"STEP_COMPLETE", step=name, duration_sec=duration, **(data or {}))
  
  # Usage:
  with track_step("generate_voice", {"scenes": len(script.scenes)}):
      synthesize_script(script, run_dir)
  ```

### 11. **No Dry-Run Validation Beyond Scrape**
- **Issue**: `main.py` has `--dry-run` flag but only skips after scrape. Weekly/Breaking have no dry-run.
- **Recommendation**: Extend dry-run to all pipelines:
  ```python
  def run_pipeline(..., dry_run: bool = False):
      if dry_run:
          logger.info("DRY RUN: skipping API calls")
      
      # Dry-run should:
      # 1. Load cached script
      # 2. Validate script structure
      # 3. Skip voice/video/upload
      # 4. Print what WOULD happen
  ```

### 12. **Missing Data Validation at Pipeline Boundaries**
- **Issue**: `NewsItem`, `Scene`, `VideoScript` have no `.validate()` methods
- **Risk**: Invalid data propagates through pipeline, fails late
- **Recommendation**:
  ```python
  @dataclass
  class VideoScript:
      title: str
      hook: str
      scenes: list[Scene]
      
      def validate(self) -> list[str]:
          """Return list of validation errors, empty if valid."""
          errors = []
          if not self.title or len(self.title) > 100:
              errors.append("title must be 1–100 chars")
          if not self.scenes:
              errors.append("at least 1 scene required")
          for i, scene in enumerate(self.scenes):
              if not scene.heading:
                  errors.append(f"scene {i}: heading required")
          return errors
      
      def validate_or_raise(self) -> None:
          if errs := self.validate():
              raise ValueError(f"Invalid VideoScript: {', '.join(errs)}")
  ```

---

## 🟢 Low Priority / Technical Debt (Quality of Life)

### 13. **Large Modules Could Be Split**
- **Largest modules**:
  - `scraper.py` (422 LOC) — 6 separate scrapers + helpers → split into `scrapers/` package
  - `infographic_generator.py` (399 LOC) — merge into `video_generator.py` or `image_generator.py`
  - `thumbnail_ab.py` (365 LOC) — combine with `thumbnail_generator.py`
  - `weekly_script_generator.py` (408 LOC) — share more code with `script_generator.py`

### 14. **Consider Async I/O for Parallel Operations**
- **Current**: Blocking HTTP calls (scraper, API calls)
- **Potential**: Parallel scraping, concurrent voice synthesis for multiple scenes
- **Note**: Would require significant refactor; worth revisiting if pipeline latency becomes issue
- **Possible Tools**: `asyncio` + `aiohttp` for scraping, `concurrent.futures` for voice synthesis

### 15. **Add CLI Progress Bars and ETA Estimation**
- **Tool**: `tqdm` library (not in requirements)
- **Benefit**: User visibility for long operations (video encoding, voice synthesis)
- **Effort**: Low (can be added gradually)

### 16. **Database Schema Versioning**
- **Issue**: `deduplicator.py` uses SQLite but no schema versioning; migrations are manual
- **Recommendation**: Use `alembic` or simple `schema_version` table
- **Benefit**: Easier upgrades without data loss

---

## 🛡️ Security & Environment Concerns

### 17. **Secrets Handling**
- **Current**: `.pickle` files and `.json` for API keys stored locally
- **Recommendation**:
  - ✓ Use environment variables (already done for API keys)
  - Use `python-dotenv` carefully — never commit `.env` file
  - Consider adding `.env.example` with placeholder values
  - Add to `.gitignore`: `*.pickle`, `*.pickle.bak`, `.env`, `config/token*.pickle`

### 18. **Path Traversal Risk**
- **Issue**: User-provided paths (e.g., `--item` in `breaking_main.py`) read directly without validation
- **Fix**:
  ```python
  import json
  from pathlib import Path
  
  def load_item_safely(path_str: str, base_dir: Path = settings.data_dir) -> NewsItem:
      item_path = (base_dir / path_str).resolve()
      if not str(item_path).startswith(str(base_dir)):
          raise ValueError(f"Path traversal attempted: {path_str}")
      return NewsItem(**json.loads(item_path.read_text()))
  ```

### 19. **Unvalidated External Content**
- **Issue**: Scraped content (titles, summaries) fed into scripts without sanitization
- **Risk**: Injection into videos, YouTube metadata, TikTok posts
- **Recommendation**: Add sanitization layer:
  ```python
  def sanitize_title(title: str, max_len: int = 100) -> str:
      # Remove HTML, control chars, URLs
      import html
      title = html.unescape(title)
      title = re.sub(r'[^\w\s\-.,!?]', '', title)  # Keep only safe chars
      return title[:max_len].strip()
  ```

---

## 📊 Performance Optimizations

### 20. **Video Encoding Efficiency**
- **Current**: Single-threaded ffmpeg (`threads=4` in settings)
- **Opportunity**: Use `preset=faster` for lower quality videos (shorts)
- **Recommendation**:
  ```python
  def build_video(..., quality: str = "medium"):
      presets = {"fast": "superfast", "medium": "medium", "high": "slow"}
      final.write_videofile(
          ...,
          preset=presets[quality],
          # ... other params
      )
  ```

### 21. **Image Caching**
- **Issue**: DALL-E images cached, but Ken Burns animation regenerated on each run
- **Recommendation**: Cache rendered video clips:
  ```python
  # In image_generator.py:
  clip_cache = out_dir / f"scene_{scene.idx:02d}_clip.mp4"
  if clip_cache.exists():
      return clip_cache
  # ... generate ...
  ```

### 22. **API Cost Monitoring**
- **Add usage tracking**:
  ```python
  # Track tokens, API calls, estimated costs
  logger.info(f"API_COST",
      dalle_calls=dalle_count,
      dalle_cost_usd=dalle_count * 0.04,
      elevenlabs_chars=total_chars,
      elevenlabs_cost_usd=total_chars * 0.000003,
  )
  ```

---

## 📋 Summary of Actionable Items (Priority Order)

| Priority | Item | Effort | Impact | Owner |
|----------|------|--------|--------|-------|
| 🔴 CRITICAL | Add pytest suite (60%+ coverage) | 2–3 days | High | QA/Dev |
| 🔴 CRITICAL | Implement checkpoint recovery in video assembly | 1 day | High | Dev |
| 🔴 CRITICAL | Fix bare exception handling (re-raise + context) | 4 hours | Medium | Dev |
| 🟠 HIGH | Externalize magic numbers & paths to `settings.py` | 1 day | Medium | Dev |
| 🟠 HIGH | Add rate-limit retry to ElevenLabs & OpenAI | 1 day | Medium | Dev |
| 🟠 HIGH | Extract base pipeline class for DRY | 2 days | Medium | Dev |
| 🟡 MEDIUM | Validate data at pipeline boundaries | 1 day | Low | Dev |
| 🟡 MEDIUM | Add path traversal protection | 2 hours | Medium | SecOps |
| 🟡 MEDIUM | Split large modules (scraper, infographic) | 2 days | Low | Dev |
| 🟢 LOW | Add progress bars (`tqdm`) | 4 hours | Low | Dev |

---

## 🎯 Next Steps

1. **This Sprint**: Fix critical issues (1–3) + high-priority #5 & #6
2. **Next Sprint**: Implement base pipeline class (#8) + validation (#12)
3. **Backlog**: Async I/O, expanded test suite, schema versioning

---

## 📚 Additional Resources

- **Python Testing**: [pytest docs](https://docs.pytest.org/)
- **Error Handling**: [PEP 3134 — Exception Chaining](https://peps.python.org/pep-3134/)
- **Logging**: [loguru best practices](https://loguru.readthedocs.io/)
- **MoviePy Tuning**: [MoviePy encoding parameters](https://zulko.github.io/moviepy/ref/VideoClip/VideoClip.write_videofile.html)

---

**Report Generated**: 2026-04-27  
**Review Scope**: Full codebase (5,711 LOC, 29 modules)
