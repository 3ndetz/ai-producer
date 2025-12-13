from google import genai
from google.genai import types
from loguru import logger

import os
from PIL import Image
from io import BytesIO
from typing import Optional
from datetime import datetime
from src.core.config import settings


class ImageGenerationClient:
    def __init__(self, api_key: Optional[str] = None, 
                       model: Optional[str] = None):
        self.api_key = api_key or settings.google_api_key
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = model or settings.image_gen_model
        logger.info("Google Gemini API (Gen Image) клиент успешно инициализирован")
        
    async def generate_image(
        self,
        prompt: str,
        output_dir: Optional[str] = None,
        iteration: int = 0
    ) -> str:
        try:
            if settings.use_image_gen_stubs:
                import asyncio
                await asyncio.sleep(2)
                return "data/images/generated_image.png"
            
            output_dir = output_dir or settings.output_dir
            os.makedirs(output_dir, exist_ok=True)
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[prompt],
                config=types.GenerateContentConfig(response_modalities=["IMAGE","TEXT"])
            )
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"image_iter_{timestamp}_{iteration}.png"
            filepath = os.path.join(output_dir, filename)
            
            image_saved = False
            for part in response.candidates[0].content.parts:
                if part.inline_data is not None:
                    image = Image.open(BytesIO(part.inline_data.data))
                    image.save(filepath)
                    image_saved = True
                    break
            
            if not image_saved:
                raise Exception("No image data found in response")
        
            logger.info(f"Image generated successfully and saved to: {filepath}")
            return filepath
            
        except Exception as e:
            error_msg = f"Error generating image: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)
    
    def is_available(self) -> bool:
        try:
            available_models = self.client.models.list()
            if self.model_name not in available_models:
                logger.warning(f"Модель {self.model_name} недоступна")
                return False

            logger.info(f"Модель {self.model_name} доступна по API")
            return True
        
        except Exception as e:
            logger.error(f"Ошибка при проверке модели: {str(e)}")
            return False