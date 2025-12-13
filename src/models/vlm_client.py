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
        critique_prompt: str
    ) -> Dict[str, Any]:
        try:
            if settings.use_vlm_stubs:
                import asyncio
                await asyncio.sleep(1)
                
                return {
                    "description": "Add more details",
                    "correct_elements": "Stub correct elements",
                    "incorrect_elements": "Stub incorrect elements",
                    "quality_score": 5.0
                }
            
            image = Image.open(image_path)
            critique_text = await self._run_vlm_inference(image, critique_prompt)
            
            parsed_analysis = self._parse_critique_response(critique_text)

            return parsed_analysis
        
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
            image, max_tokens=300, verbose=False
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
       
    
    def _extract_quality_score(critique_text: str) -> float:
        patterns = [
            r"Quality Score:\s*([\d.,]+)",
            r"Score:\s*([\d.,]+)",
            r"([\d.,]+)\s*/\s*10"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, critique_text, re.IGNORECASE)
            if match:
                score_str = match.group(1).replace(',', '.')
                try:
                    score = float(score_str)
                    return min(max(score, 0.0), 10.0)
                except ValueError:
                    pass

        return 5.0
    
    def _parse_critique_response(self, response_text: str) -> Dict[str, Any]:    
        description_match = re.search(r"\*\*Description:\*\*\s*(.*?)(?=\*\*|$)", response_text, re.DOTALL)
        correct_match = re.search(r"\*\*Correct Elements:\*\*\s*(.*?)(?=\*\*|$)", response_text, re.DOTALL)
        incorrect_match = re.search(r"\*\*Incorrect Elements:\*\*\s*(.*?)(?=\*\*|$)", response_text, re.DOTALL)
        
        description = description_match.group(1).strip() if description_match else "No description provided"
        correct_elements = correct_match.group(1).strip() if correct_match else "No correct elements identified"
        incorrect_elements = incorrect_match.group(1).strip() if incorrect_match else "No incorrect elements identified"
        
        quality_score = self._extract_quality_score(response_text)
        
        return {
            "description": description,
            "correct_elements": correct_elements,
            "incorrect_elements": incorrect_elements,
            "quality_score": quality_score
        }
    
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
