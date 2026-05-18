"""Coverage gate for newly-added modules.

Reads coverage.json produced by pytest-cov --cov-report=json and exits with
status 1 if any listed module falls below THRESHOLD percent.

Run after the main test step:
    python scripts/check_new_module_coverage.py
"""
import json
import sys
from pathlib import Path

THRESHOLD = 80

# Modules added or significantly expanded in this feature branch.
# Do NOT add legacy modules here — fix their coverage instead.
NEW_MODULES = [
    "src/config_validator.py",
    "src/deferred_feedback.py",
    "src/pipeline_orchestrator.py",
    "src/script_step.py",
]


def main() -> int:
    coverage_file = Path("coverage.json")
    if not coverage_file.exists():
        print("coverage.json not found — run pytest with --cov-report=json first")
        return 1

    with coverage_file.open() as f:
        data: dict = json.load(f)

    files: dict = data.get("files", {})
    failed: list[str] = []

    for module in NEW_MODULES:
        if module not in files:
            print(f"MISSING  {module}  (not found in coverage.json)")
            failed.append(module)
            continue
        pct: float = files[module]["summary"]["percent_covered"]
        mark = "PASS" if pct >= THRESHOLD else "FAIL"
        print(f"{mark}  {pct:5.1f}%  {module}")
        if pct < THRESHOLD:
            failed.append(module)

    if failed:
        print(f"\nCoverage below {THRESHOLD}%: {', '.join(failed)}")
        return 1

    print(f"\nAll new modules meet >={THRESHOLD}% coverage requirement.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
