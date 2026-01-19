import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):    
    # API
    google_api_key: Optional[str] = None
    openrouter_api_key: Optional[str] = None
    mistral_api_key: Optional[str] = None
    
    # Models
    image_generation_backend: str = "comfy"
    image_gen_model: str = "gemini-2.0-flash-exp-image-generation"
    comfy_url: str = "http://127.0.0.1:8000/"
    comfy_workflow_path: Optional[str] = "./data/workflows/z_image_turbo_gguf.json"
    comfy_save_node_id: str = "11"
    comfy_poll_interval: float = 1.5
    comfy_poll_timeout: float = 600.0
    comfy_unet_name: str = "z_image_turbo-Q3_K_S.gguf"
    comfy_clip_name: str = "Qwen3-4B-Q3_K_S.gguf"
    comfy_clip_type: str = "qwen_image"
    comfy_vae_name: str = "ae.safetensors"
    comfy_negative_prompt: str = "blurry ugly bad"
    comfy_width: int = 128
    comfy_height: int = 128
    comfy_batch_size: int = 1
    comfy_steps: int = 2
    comfy_cfg: float = 1.0
    comfy_sampler_name: str = "euler"
    comfy_scheduler: str = "simple"
    comfy_denoise: float = 1.0
    comfy_seed: int = 526213216603105

    vlm_model_type: str = "api"
    vlm_model_local: str = "mlx-community/Qwen2-VL-2B-Instruct-4bit"
    vlm_model_api: str = "gemini-2.0-flash-exp-image-generation"

    llm_model_type: str = "mistral"
    llm_model_openrouter: str = "deepseek/deepseek-chat-v3.1:free"
    llm_model_mistral: str = "mistral-small-latest"
    llm_model_local: str = "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
    
    # Параметры итераций
    max_iterations: int = 5
    min_quality_score: float = 8.0
    default_iterations_without_feedback: int = 1
    
    # Максимальное количество токенов
    planner_max_tokens: int = 500
    generator_max_tokens: int = 500
    vlm_max_tokens: int = 512
    critic_requirements_max_tokens: int = 600
    critic_comparison_max_tokens: int = 600
    
    output_dir: str = "./data/images"
    
    # Web сервер
    web_host: str = "0.0.0.0"
    web_port: int = 8008
    
    # Backend сервер
    backend_host: str = "0.0.0.0"
    backend_port: int = 8001
    
    # Режим тестирования с заглушками
    use_vlm_stubs: bool = True
    use_llm_stubs: bool = False
    use_image_gen_stubs: bool = True
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "allow"


settings = Settings()

os.makedirs(settings.output_dir, exist_ok=True)
