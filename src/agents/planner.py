import os
import re
import sys
import json
from loguru import logger
from typing import Callable, Optional
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.config import settings
from src.core.state import WorkflowState
from src.models.llm_client import LLMClient
from src.prompts.planner_prompts import PLANNER_SYSTEM_PROMPT, format_planner_prompt


class PlannerAgent:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.name = "Planner"
        self.status_callback: Optional[Callable] = None
    
    def set_status_callback(self, callback: Callable):
        self.status_callback = callback
    
    async def execute(self, state: WorkflowState) -> WorkflowState:
        logger.debug(f"\n{'='*60}")
        logger.debug(f"🧠 {self.name} Agent - Iteration {state['current_iteration']}")
        logger.debug(f"{'='*60}")
        
        if self.status_callback:  # STATUS_CALLBACK: planning_continuation
            await self.status_callback("planning_continuation", {
                "thread_id": state['thread_id'],
                "current_iteration": state['current_iteration']
            })
        
        try:
            critique_history = "\n".join([f"Iteration {step['iteration']}: {step['critique']}" for step in state['iterations'] if step['critique']])
            feedback_history = "\n".join([f"Iteration {step['iteration']}: {step['user_feedback']}" for step in state['iterations'] if step['user_feedback']])
            feedback_history += f"\n Iteration {state['current_iteration']}: {state['current_user_feedback']}" if state['current_user_feedback'] else ""

            planner_prompt = format_planner_prompt(
                user_request=state['user_request'],
                current_user_feedback=state['current_user_feedback'],
                current_prompt=state['current_prompt'],
                current_critique=state['current_critique'],
                quality_score=state['current_quality_score'],
                critique_history=critique_history,
                feedback_history=feedback_history
            )
            
            decision_response = await self.llm_client.generate(
                prompt=planner_prompt,
                system_prompt=PLANNER_SYSTEM_PROMPT,
                max_tokens=settings.planner_max_tokens,
                temperature=0.7
            )

            try:
                data = json.loads(decision_response)
            except json.JSONDecodeError:
                data = {}
                cont_match = re.search(r'"continue"\s*:\s*(true|false)', decision_response, re.IGNORECASE)
                data['continue'] = cont_match.group(1).lower() == 'true' if cont_match else None

                rec_match = re.search(r'"recommendations"\s*:\s*"([^"]*)"', decision_response)
                data['recommendations'] = rec_match.group(1) if rec_match else None
            
            logger.debug(f"Decision: continue={data['continue']}, recommendations={data['recommendations'][:100]}...")
            
            if self.status_callback:  # STATUS_CALLBACK: decision_about_continuation
                await self.status_callback("decision_about_continuation", {
                    "thread_id": state['thread_id'],
                    "continue_generation": data['continue'],
                    "current_recommendations": data['recommendations'],
                    "current_iteration": state['current_iteration']
                })
            
            state.update({
                "continue_generation": data['continue'],
                "current_recommendations": data['recommendations']
            })
                        
            return state
                    
        except Exception as e:
            error_msg = f"Prompt planning failed: {type(e).__name__}: {str(e)}"
            logger.error(f"{error_msg}\nCurrent iteration: {state['current_iteration']}")
            
            if self.status_callback:  # STATUS_CALLBACK: error
                await self.status_callback("error", {
                    "thread_id": state['thread_id'],
                    "error_message": error_msg,
                    "agent": "Planner",
                    "current_iteration": state['current_iteration'],
                    "exception_type": type(e).__name__
                })
            
            state.update({"error_message": error_msg})
            return state
    
    def __str__(self):
        return f"PlannerAgent(llm_client={self.llm_client.model})"
