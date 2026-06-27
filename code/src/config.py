import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    vision_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o"))
    text_model: str = field(default_factory=lambda: os.getenv("OPENAI_TEXT_MODEL", "gpt-4o-mini"))
    max_retries: int = field(default_factory=lambda: int(os.getenv("MAX_RETRIES", "3")))
    request_delay_seconds: float = field(
        default_factory=lambda: float(os.getenv("REQUEST_DELAY_SECONDS", "0.5"))
    )
    cache_dir: Path = field(default_factory=lambda: Path(os.getenv("CACHE_DIR", ".cache")))

    def validate(self) -> None:
        if not self.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Copy code/.env.example to code/.env and add your key."
            )


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def dataset_dir() -> Path:
    return repo_root() / "dataset"
