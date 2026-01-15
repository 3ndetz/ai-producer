from loguru import logger
from typing import Optional
from src.core.config import settings

from src.models.comfyui_client import ComfyUIClient

from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
import os
from datetime import datetime


class ImageGenerationClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None
    ):
        self.provider = settings.image_gen_provider

        if self.provider == "gemini":
            self.api_key = api_key or settings.google_api_key
            self.client = genai.Client(api_key=self.api_key)
            self.model_name = model or settings.image_gen_model
            logger.info("ImageGen provider: Gemini")

        elif self.provider == "comfyui":
            self.client = ComfyUIClient()
            logger.info("ImageGen provider: ComfyUI")

        else:
            raise ValueError(f"Unknown image_gen_provider: {self.provider}")

    async def generate_image(
        self,
        prompt: str,
        output_dir: Optional[str] = None,
        iteration: int = 0
    ) -> str:

        if settings.use_image_gen_stubs:
            import asyncio
            await asyncio.sleep(1)
            return "data/images/generated_image.png"

        output_dir = output_dir or settings.output_dir

        if self.provider == "comfyui":
            return await self.client.generate_image(
                prompt=prompt,
                output_dir=output_dir,
                iteration=iteration
            )

        # ---- Gemini fallback ----
        os.makedirs(output_dir, exist_ok=True)

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[prompt],
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"]
            )
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"image_iter_{timestamp}_{iteration}.png"
        filepath = os.path.join(output_dir, filename)

        for part in response.candidates[0].content.parts:
            if part.inline_data:
                image = Image.open(BytesIO(part.inline_data.data))
                image.save(filepath)
                logger.info(f"Image saved: {filepath}")
                return filepath

        raise RuntimeError("Gemini: image not found")

    def is_available(self) -> bool:
        if self.provider == "comfyui":
            return True  # health-check можно добавить позже

        try:
            models = self.client.models.list()
            return self.model_name in models
        except Exception:
            return False
