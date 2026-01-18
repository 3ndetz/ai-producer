from loguru import logger
from typing import Callable, Optional

from src.core.config import settings
from src.core.state import WorkflowState
from src.models.llm_client import LLMClient
from src.models.image_generation_client import ImageGenerationClient
from src.prompts.generator_prompts import GENERATOR_SYSTEM_PROMPT, GENERATOR_SYSTEM_PROMPT_SDXL, format_generator_prompt


class GeneratorAgent:
    def __init__(self, imagen_client: ImageGenerationClient, llm_client: LLMClient):
        self.imagen_client = imagen_client
        self.llm_client = llm_client
        self.name = "Generator"
        self.status_callback: Optional[Callable] = None
    
    def set_status_callback(self, callback: Callable):
        self.status_callback = callback

    async def execute(self, state: WorkflowState) -> WorkflowState:
        logger.debug(f"\n{'='*60}")
        logger.debug(f"🎨 {self.name} Agent - Iteration {state['current_iteration']}")
        logger.debug(f"{'='*60}")
        
        try:
            generator_prompt = format_generator_prompt(
                user_request=state['user_request'],
                user_feedback=state.get('current_user_feedback'),
                current_prompt=state['current_prompt'],
                critique=state.get('current_critique'),
                recommendations=state.get('current_recommendations')
            )

            if self.status_callback:  # STATUS_CALLBACK: prompt_generation
                await self.status_callback("prompt_generation", {
                    "thread_id": state['thread_id'],
                    "prompt": generator_prompt,
                    "current_iteration": state['current_iteration']
                })            
            
            system_prompt = GENERATOR_SYSTEM_PROMPT
            if settings.image_gen_style == 'sdxl':
                system_prompt = GENERATOR_SYSTEM_PROMPT_SDXL
            optimized_prompt = await self.llm_client.generate(
                prompt=generator_prompt,
                system_prompt=system_prompt,
                max_tokens=settings.generator_max_tokens,
                temperature=0.7
            )
            
            logger.debug(f"Optimized prompt: {optimized_prompt[:150]}...")
            
            if self.status_callback:  # STATUS_CALLBACK: prompt_ready
                await self.status_callback("prompt_ready", {
                    "thread_id": state['thread_id'],
                    "prompt": optimized_prompt,
                    "current_iteration": state['current_iteration']
                })            

            if self.status_callback:  # STATUS_CALLBACK: image_generation
                await self.status_callback("image_generation", {
                    "thread_id": state['thread_id'],
                    "current_iteration": state['current_iteration']
                })
            
            image_path = await self.imagen_client.generate_image(
                prompt=optimized_prompt,
                iteration=state['current_iteration'] 
            )
            
            if self.status_callback:  # STATUS_CALLBACK: image_ready
                await self.status_callback("image_ready", {
                    "thread_id": state['thread_id'],
                    "image_path": image_path,
                    "current_iteration": state['current_iteration']
                })
            
            state.update({
                "current_prompt": optimized_prompt,
                "current_image_path": image_path,
                "prompt_history": state["prompt_history"] + [optimized_prompt]
            })
            
            return state
            
        except Exception as e:
            error_msg = f"Image generation failed: {type(e).__name__}: {str(e)}"
            logger.error(f"{error_msg}\nPrompt: {state['current_prompt'][:100]}...")
            
            if self.status_callback:  # STATUS_CALLBACK: error
                await self.status_callback("error", {
                    "thread_id": state['thread_id'],
                    "error_message": error_msg,
                    "agent": "Generator",
                    "current_iteration": state['current_iteration'],
                    "exception_type": type(e).__name__
                })
            
            state.update({
                "error_message": error_msg
            })
            return state
    
    def __str__(self):
        return f"GeneratorAgent(imagen_client={self.imagen_client.model_name}, llm_client={self.llm_client.model})"
