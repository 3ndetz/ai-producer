from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from src.core.config import settings


class ImageGenerationClientInterface(ABC):
    """
    An abstract interface for image generation clients.
    """

    @abstractmethod
    async def generate_image(
        self,
        prompt: str,
        output_dir: Optional[str] = None,
        iteration: int = 0,
        thread_id: Optional[str] = None
    ) -> str:
        ...


def build_image_generation_client() -> ImageGenerationClientInterface:
    backend = settings.image_generation_backend.lower()
    if backend == "comfy":
        from src.models.image_gen_comfy_client import ComfyImageGenerationClient

        return ComfyImageGenerationClient(
            base_url=settings.comfy_url,
            workflow_template_path=settings.comfy_workflow_path,
            save_node_id=settings.comfy_save_node_id,
            poll_interval=settings.comfy_poll_interval,
            poll_timeout=settings.comfy_poll_timeout
        )

    from src.models.image_gen_api_client import ImageGenerationClient

    return ImageGenerationClient()