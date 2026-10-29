"""Configuration. Everything tunable lives here, loaded from env with safe defaults."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

try:  # optional: .env is a developer convenience, not a requirement
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv absent is fine
    pass


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "")) if os.getenv(name) else default
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "")) if os.getenv(name) else default
    except ValueError:
        return default


PROJECT_ROOT = Path(__file__).resolve().parent.parent
def data_dir() -> Path:
    """Resolve at settings creation so tests and explicit overrides stay isolated."""
    return Path(os.getenv("INTERVIEWTRACK_DATA_DIR", PROJECT_ROOT / "data"))


DATA_DIR = data_dir()  # compatibility; defaults below resolve dynamically


class MatchWeights(BaseModel):
    """Weights for the hybrid match score.

    These are a transparent weighted sum, NOT a calibrated probability. They were
    tuned against the labeled synthetic fixtures (see scripts/evaluate.py).
    """

    model_config = ConfigDict(extra="forbid")

    requisition_match: float = 0.45
    employer_match: float = 0.34
    role_similarity: float = 0.26
    shared_recruiter: float = 0.12
    #: A sender already LEARNED as belonging to this application. The core
    #: recruiter-anchoring signal: the strongest link short of a thread or
    #: requisition, but still never decisive on its own — employer and role decide.
    known_contact: float = 0.34
    same_gmail_thread: float = 0.60
    thread_reference_link: float = 0.16
    semantic_similarity: float = 0.22
    temporal_proximity: float = 0.06

    #: Semantic similarity below this contributes nothing (noise floor).
    semantic_floor: float = 0.45
    #: Cap on how much semantics can contribute when employer identity is unknown.
    semantic_only_cap: float = 0.34


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # --- storage ---
    db_path: Path = Field(default_factory=lambda: data_dir() / "interviewtrack.db")

    # --- embedding model ---
    embedding_model: str = Field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    )
    embedding_fallback_model: str = Field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_FALLBACK_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )
    embeddings_enabled: bool = Field(default_factory=lambda: _env_bool("EMBEDDINGS_ENABLED", True))

    # --- matching thresholds (tuned on labeled fixtures) ---
    weights: MatchWeights = Field(default_factory=MatchWeights)
    auto_match_threshold: float = Field(
        default_factory=lambda: _env_float("AUTO_MATCH_THRESHOLD", 0.62)
    )
    review_threshold: float = Field(default_factory=lambda: _env_float("REVIEW_THRESHOLD", 0.34))

    # --- Gmail retrieval ---
    lookback_days: int = Field(default_factory=lambda: _env_int("GMAIL_LOOKBACK_DAYS", 92))
    max_messages: int = Field(default_factory=lambda: _env_int("GMAIL_MAX_MESSAGES", 1500))
    gmail_page_size: int = Field(default_factory=lambda: _env_int("GMAIL_PAGE_SIZE", 100))

    # --- Gmail Mode: Desktop OAuth (single user, read-only) ---
    client_secret_file: str = Field(
        default_factory=lambda: os.getenv("GOOGLE_CLIENT_SECRET_FILE",
                                          str(data_dir() / "client_secret.json"))
    )
    # Per-account token storage. The OAuth *app* (client_secret.json) is shared —
    # that is not account data — but each authorized account gets its OWN token file
    # under this directory, so a second account never overwrites the first.
    tokens_dir: str = Field(
        default_factory=lambda: os.getenv("GOOGLE_TOKENS_DIR",
                                          str(data_dir() / "tokens"))
    )
    # Legacy single-token path, still honoured for a one-account setup.
    token_file: str = Field(
        default_factory=lambda: os.getenv("GOOGLE_TOKEN_FILE",
                                          str(data_dir() / "token.json"))
    )
    # OPTIONAL expected account. Set locally in .env; NEVER hardcoded in logic. When
    # set, authorization and sync warn if the connected account differs.
    gmail_expected_account: str = Field(
        default_factory=lambda: os.getenv("GMAIL_EXPECTED_ACCOUNT", "").strip()
    )

    # --- legacy Web OAuth (kept for reference; Gmail Mode uses Desktop) ---
    google_client_id: str = Field(default_factory=lambda: os.getenv("GOOGLE_CLIENT_ID", ""))
    google_client_secret: str = Field(default_factory=lambda: os.getenv("GOOGLE_CLIENT_SECRET", ""))
    oauth_redirect_uri: str = Field(
        default_factory=lambda: os.getenv("OAUTH_REDIRECT_URI", "http://localhost:8501")
    )

    # --- optional external LLM (OFF by default: emails contain personal data) ---
    llm_review_enabled: bool = Field(default_factory=lambda: _env_bool("LLM_REVIEW_ENABLED", False))
    anthropic_api_key: str = Field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    llm_model: str = Field(default_factory=lambda: os.getenv("LLM_MODEL", "claude-sonnet-5-5"))
    llm_max_candidates: int = Field(default_factory=lambda: _env_int("LLM_MAX_CANDIDATES", 3))

    @property
    def google_configured(self) -> bool:
        return bool(self.google_client_id and self.google_client_secret)

    @property
    def llm_configured(self) -> bool:
        """Note: Claude Code's own auth does NOT grant API access — a real key is required."""
        return self.llm_review_enabled and bool(self.anthropic_api_key)

    #: Gmail scopes. Read-only by design; no write/delete/modify is ever requested.
    @property
    def google_scopes(self) -> list[str]:
        return [
            "openid",
            "https://www.googleapis.com/auth/userinfo.email",
            "https://www.googleapis.com/auth/userinfo.profile",
            "https://www.googleapis.com/auth/gmail.readonly",
        ]


_settings: Settings | None = None


def get_settings(refresh: bool = False) -> Settings:
    """Process-wide settings singleton."""
    global _settings
    if _settings is None or refresh:
        data_dir().mkdir(parents=True, exist_ok=True)
        _settings = Settings()
    return _settings
