from typing import TypedDict, List, Optional

from src.core.config import settings


class IterationStep(TypedDict):
    iteration: int
    user_feedback: str
    recommendations: Optional[str]
    prompt: str
    image_path: Optional[str]
    critique: Optional[str]
    quality_score: Optional[float]
    timestamp: str



class WorkflowState(TypedDict):
    thread_id: str
    user_request: str
    current_iteration: int
    iterations_without_feedback: int  

    current_prompt: str    
    current_user_feedback: Optional[str]
    current_image_path: Optional[str]
    current_critique: Optional[str]
    current_quality_score: Optional[float]
    current_recommendations: Optional[str]

    iterations: List[IterationStep]
    error_message: Optional[str]
    prompt_history: List[str]
    continue_generation: bool



def create_initial_state(user_request: str, thread_id: str) -> WorkflowState:
    return WorkflowState(
        thread_id=thread_id,
        user_request=user_request,
        current_iteration=1,
        iterations_without_feedback=settings.default_iterations_without_feedback,

        current_prompt=user_request,
        current_user_feedback=None,
        current_image_path=None,
        current_critique=None,
        current_quality_score=None,
        current_recommendations=None,

        iterations=[],
        error_message=None, 
        prompt_history=[user_request],
        continue_generation=True,
    )
