from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Ollama (OpenAI-compatible endpoint)
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_api_key: str = "ollama"

    # Models
    victim_model: str = "llama3.1:8b"
    attacker_model: str = "qwen2.5:7b"

    # Paths
    db_path: Path = Path("./data/results.db")
    log_dir: Path = Path("./logs")

    # Request catcher
    catcher_host: str = "localhost"
    catcher_port: int = 5001
    catcher_url: str = "http://localhost:5001"

    # Experiments
    max_iterations: int = 40
    log_level: str = "INFO"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
