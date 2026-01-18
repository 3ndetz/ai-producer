import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):    
    # API
    google_api_key: Optional[str] = None
    openrouter_api_key: Optional[str] = None
    mistral_api_key: Optional[str] = None
    
    # Models
    image_gen_model: str = "gemini-2.0-flash-exp-image-generation"

    vlm_model_local: str = "mlx-community/Qwen2-VL-2B-Instruct-4bit"
    vlm_model_api: str = "gemini-2.0-flash-exp-image-generation"
    vlm_model_type: str = "api"

    llm_model_openrouter: str = "deepseek/deepseek-chat-v3.1:free"
    llm_model_mistral: str = "mistral-small-latest"
    llm_model_local: str = "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
    llm_model_type: str = "mistral"
    
    # Параметры итераций
    max_iterations: int = 5
    min_quality_score: float = 8.0
    default_iterations_without_feedback: int = 1
    
    # Максимальное количество токенов
    planner_max_tokens: int = 500
    generator_max_tokens: int = 500
    
    output_dir: str = "./data/images"
    
    # Web сервер
    web_host: str = "0.0.0.0"
    web_port: int = 8008
    
    # Backend сервер
    backend_host: str = "0.0.0.0"
    backend_port: int = 8011
    
    # Режим тестирования с заглушками
    use_vlm_stubs: bool = True
    use_llm_stubs: bool = False
    use_image_gen_stubs: bool = True
    

    image_gen_style: str = "descriptive"
    image_gen_provider: str = "comfyui"  
    # gemini | comfyui

    comfyui_base_url: str = "http://localhost:8188"
    comfyui_workflow_path: str = "data/workflows/z_image_turbo_simple.json"
    comfyui_timeout: int = 300

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "allow"


settings = Settings()

os.makedirs(settings.output_dir, exist_ok=True)
