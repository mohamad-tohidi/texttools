from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM API settings
    opanai_api_key: str
    base_url: str
    model: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
