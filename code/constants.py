# ============================================================
# constants.py — Centralised absolute paths & defaults
# ============================================================
from pathlib import Path

# --- Base project root (folder containing this file) ---
PROJECT_ROOT = Path(__file__).resolve().parent

# --- Core directories (absolute) ---
ROOT_DIR = PROJECT_ROOT / "ValenceSphere"
GLOBAL_DIR = ROOT_DIR / "_global"
LOG_DIR = ROOT_DIR / "_adjudication_logs"

# --- Ensure existence ---
for d in [ROOT_DIR, GLOBAL_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# --- Model defaults ---
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"

# --- Utility (optional) ---
def safe_concept_name(name: str) -> str:
    """Return filesystem-safe version of a concept name."""
    import re
    return re.sub(r'[:<>"/\\|?*]+', "_", name.strip())
