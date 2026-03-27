"""
FortifAI Admin – Server configuration constants.
Edit this file to change server, database, and FL parameters.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Network ────────────────────────────────────────────────────────────────────
SERVER_HOST: str = "0.0.0.0"
SERVER_PORT: int = 9999

# ── PostgreSQL ─────────────────────────────────────────────────────────────────
DB_CONFIG: dict = {
    "host":     os.getenv("DB_HOST",     "localhost"),
    "database": os.getenv("DB_NAME",     "fortifai_db"),
    "user":     os.getenv("DB_USER",     "postgres"),
    "password": os.getenv("DB_PASSWORD", "postgres"),
    "port":     int(os.getenv("DB_PORT", "5432")),
}

# ── Federated Learning (canonical source: shared/detection_config.py) ─────────
from shared.detection_config import (
    FL_SERVER_CLIP_BOUND as CLIP_BOUND,
    FL_TRIM_FRAC as TRIM_FRAC,
    FL_DP_NOISE_SCALE as DP_NOISE_SCALE,
    FL_VALIDATION_AUC_DROP as VALIDATION_AUC_DROP_THRESHOLD,
    FL_LEARNING_RATE,
    FL_MOMENTUM,
    FL_MU,
)

# ── AI Assistant ───────────────────────────────────────────────────────────────
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
# Override with GEMINI_MODEL in .env (e.g. gemini-2.5-flash for stable, gemini-2.5-pro for depth).
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")

# ── GUI Refresh ────────────────────────────────────────────────────────────────
GUI_REFRESH_MS: int = 5_000        # Dashboard auto-refresh (ms)
FL_AUTO_AGGREGATE_MS: int = 120_000  # Auto-aggregation interval (ms)