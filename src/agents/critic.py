from loguru import logger
from datetime import datetime
from typing import Callable, Optional

from src.models.vlm_client import VLMClient
from src.core.state import WorkflowState, IterationStep
from src.prompts.critic_prompts import format_critic_prompt


class CriticAgent:    
    def __init__(self, vlm_client: VLMClient):
        self.vlm_client = vlm_client
        self.name = "Critic"
        self.status_callback: Optional[Callable] = None
    
    def set_status_callback(self, callback: Callable):
        self.status_callback = callback
    
    async def execute(self, state: WorkflowState) -> WorkflowState:
        logger.debug(f"\n{'='*60}")
        logger.debug(f"🔍 {self.name} Agent - Iteration {state['current_iteration']}")
        logger.debug(f"{'='*60}")
        
        if not state.get('current_image_path'):
            error_msg = "No image to analyze"
            logger.error(error_msg)
            return {"error_message": error_msg}
        
        if self.status_callback:  # STATUS_CALLBACK: analyzing_image
            await self.status_callback("analyzing_image", {
                "thread_id": state['thread_id'],
                "current_iteration": state['current_iteration']
            })
        
        try:
            critique_prompt = format_critic_prompt(
                state['user_request'], 
                state.get('current_user_feedback'), 
                state['current_prompt']
            )
            
            analysis = await self.vlm_client.analyze_image(
                image_path=state['current_image_path'],
                critique_prompt=critique_prompt
            )
            
            description = analysis['description']
            correct_elements = analysis['correct_elements']
            incorrect_elements = analysis['incorrect_elements']
            quality_score = analysis['quality_score']
            
            critique = f"**Description:** {description}\n\n**Correct Elements:** {correct_elements}\n\n**Incorrect Elements:** {incorrect_elements}"
            
            logger.debug(f"Quality Score: {quality_score:.1f}/10")
            logger.debug(f"\nCritique Preview:")
            logger.debug(critique[:200] + "..." if len(critique) > 200 else critique)


            if self.status_callback:  # STATUS_CALLBACK: analysis_ready
                await self.status_callback("analysis_ready", {
                    "thread_id": state['thread_id'],
                    "critique": critique,
                    "quality_score": quality_score,
                    "current_iteration": state['current_iteration']
                })
            
            iteration_record = IterationStep(
                iteration=state['current_iteration'],
                user_feedback=state['current_user_feedback'],
                recommendations=state["current_recommendations"],
                prompt=state['current_prompt'],
                image_path=state['current_image_path'],
                critique=critique,
                quality_score=quality_score,
                timestamp=datetime.now().isoformat()
            )
            
            state.update({
                    "current_critique": critique,
                    "current_iteration": state['current_iteration'] + 1,
                    "current_quality_score": quality_score,
                    "iterations": state["iterations"] + [iteration_record],
                    "iterations_without_feedback": state["iterations_without_feedback"] - 1
                })        

            return state
            
        except Exception as e:
            error_msg = f"Image analysis failed: {type(e).__name__}: {str(e)}"
            logger.error(f"{error_msg}\nImage path: {state.get('current_image_path', 'N/A')}")
            
            if self.status_callback:  # STATUS_CALLBACK: error
                await self.status_callback("error", {
                    "thread_id": state['thread_id'],
                    "error_message": error_msg,
                    "agent": "Critic",
                    "current_iteration": state['current_iteration'],
                    "exception_type": type(e).__name__
                })
            
            return {
                "error_message": error_msg,
            }
    
    
    def __str__(self):
        return f"CriticAgent(vlm_client={self.vlm_client.model_name})"
