from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _build_ws_url(base: str) -> str:
    if base.startswith('https://'):
        return base.replace('https://', 'wss://', 1).rstrip('/')
    if base.startswith('http://'):
        return base.replace('http://', 'ws://', 1).rstrip('/')
    return base.rstrip('/')


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')

    backend_host: str = Field(default='0.0.0.0', env='BACKEND_HOST')
    backend_port: int = Field(default=8001, env='BACKEND_PORT')

    raw_backend_url: "str | None" = Field(default=None, env='BACKEND_URL')

    http_timeout: int = Field(default=120, env='HTTP_TIMEOUT')

    @computed_field
    @property
    def backend_url(self) -> str:
        """Return explicit BACKEND_URL if set, otherwise build from host/port."""
        if self.raw_backend_url:
            return self.raw_backend_url.rstrip('/')
        host = self.backend_host.rstrip('/')

        if host.startswith('http://') or host.startswith('https://'):
            return f"{host}:{self.backend_port}"
        return f"http://{host}:{self.backend_port}"

    @computed_field
    @property
    def backend_ws_url(self) -> str:
        return _build_ws_url(self.backend_url)


settings = Settings()