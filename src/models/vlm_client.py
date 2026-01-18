import re
from PIL import Image
from loguru import logger
from src.core.config import settings
from typing import Optional, Dict, Any, Literal

try:
    from mlx_vlm import load, generate
    from mlx_vlm.utils import load_config
    from mlx_vlm.prompt_utils import apply_chat_template
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

class VLMClient:    
    def __init__(
        self, 
        model_name: Optional[str] = None,
        mode: Literal["local", "api"] = settings.vlm_model_type,
        api_key: Optional[str] = None
    ):
        from core.config import settings
        
        self.mode = mode
        if self.mode == "api":
            self.model_name = model_name or settings.vlm_model_api
        else:
            self.model_name = model_name or settings.vlm_model_local

        self.api_key = api_key or settings.google_api_key
        
        self.model = None
        self.processor = None
        self.genai_client = None
        
        if self.mode == "local":
            if not MLX_AVAILABLE:
                raise ImportError("mlx_vlm is not installed. Install it or use mode='api'")
            self._load_mlx_model()
        elif self.mode == "api":
            if not GENAI_AVAILABLE:
                raise ImportError("google-genai is not installed. Install it with: pip install google-genai")
            self._init_genai_client()
        else:
            raise ValueError(f"Invalid mode: {mode}. Use 'local' or 'api'")
    
    def _load_mlx_model(self):
        try:
            self.model, self.processor = load(self.model_name)
            self.config = load_config(self.model_name)
            
            logger.info(f"VLM модель '{self.model_name}' успешно загружена (MLX)")
            
        except Exception as e:
            logger.warning(f"Не удалось загрузить VLM модель '{self.model_name}': {e}")
            self.model = None
    
    def _init_genai_client(self):
        try:
            self.genai_client = genai.Client(api_key=self.api_key)
            logger.info("Google Gemini API (VLM) клиент успешно инициализирован")
        except Exception as e:
            logger.error(f"Ошибка инициализации Google Gemini API клиента: {e}")
            raise
    
    async def analyze_image(
        self,
        image_path: str,
        description_prompt: str
    ) -> str:
        try:
            if settings.use_vlm_stubs:
                import asyncio
                await asyncio.sleep(1)
                return "Stub visual description"

            image = Image.open(image_path)
            description = await self._run_vlm_inference(image, description_prompt)
            return description
        except Exception as e:
            error_msg = f"Error analyzing image: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)

    
    async def _run_vlm_inference(self, image: Image.Image, prompt: str) -> str:
        if self.mode == "local":
            return await self._run_mlx_inference(image, prompt)
        else:
            return await self._run_genai_inference(image, prompt)
    
    async def _run_mlx_inference(self, image: Image.Image, prompt: str) -> str:
        formatted_prompt = apply_chat_template(
            self.processor, self.config, prompt, num_images=1
        )
        output = generate(
            self.model, self.processor, formatted_prompt, 
            image, max_tokens=settings.vlm_max_tokens, verbose=False
        )
        return output.text
    
    async def _run_genai_inference(self, image: Image.Image, prompt: str) -> str:
        import io
        
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        image_bytes = img_byte_arr.getvalue()
        
        response_text = ""
        for chunk in self.genai_client.models.generate_content_stream(
            model=self.model_name,
            contents=[
                prompt,
                types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
            ],
        ):
            if chunk.text:
                response_text += chunk.text
        
        return response_text
       
    
    @staticmethod
    def _extract_quality_score(critique_text: str) -> float:
        patterns = [
            r"Quality\s*Score.*?:\D*([\d.,]+)",
            r"Score.*?:\D*([\d.,]+)",
            r"([\d.,]+)\s*/\s*10"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, critique_text, re.IGNORECASE)
            if match:
                score_str = match.group(1).replace(',', '.')
                try:
                    score = float(score_str)
                    logger.debug(f"Extracted quality score: {score}")
                    return min(max(score, 0.0), 10.0)
                except ValueError:
                    pass

        return 5.0
    
    async def is_available(self) -> bool:
        if self.mode == "local":
            available = self.model is not None
            if not available:
                logger.warning("VLM local model is not loaded")
            return available
        elif self.mode == "api":
            available = self.genai_client is not None
            if not available:
                logger.warning("VLM API client is not initialized")
            return available
        return False
