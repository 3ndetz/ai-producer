from datetime import datetime
from typing import Any, Callable, Dict, Optional, Type, TypeVar
from loguru import logger
from pydantic import BaseModel, ValidationError

from src.agents.critic_schemas import (
    CriticComparisonSchema,
    CriticRequirementsSchema,
)
from src.core.config import settings
from src.core.state import IterationStep, WorkflowState
from src.models.llm_client import LLMClient
from src.models.vlm_client import VLMClient
from src.prompts.critic_prompts import (
    CRITIC_COMPARISON_SYSTEM_PROMPT,
    CRITIC_REQUIREMENTS_SYSTEM_PROMPT,
    format_critic_comparison_prompt,
    format_requirements_prompt,
    format_vlm_description_prompt,
)

TModel = TypeVar("TModel", bound=BaseModel)


class CriticAgent:
    def __init__(self, vlm_client: VLMClient, llm_client: LLMClient):
        self.vlm_client = vlm_client
        self.llm_client = llm_client
        self.name = "Critic"
        self.status_callback: Optional[Callable] = None

    def set_status_callback(self, callback: Callable):
        self.status_callback = callback

    async def execute(self, state: WorkflowState) -> WorkflowState:
        logger.debug(f"\n{'=' * 60}")
        logger.debug(f"🔍 {self.name} Agent - Iteration {state['current_iteration']}")
        logger.debug(f"{'=' * 60}")

        if not state.get("current_image_path"):
            error_msg = "No image to analyze"
            logger.error(error_msg)
            return {"error_message": error_msg}

        try:
            if self.status_callback:  # STATUS_CALLBACK: critic_requirements_start
                await self.status_callback(
                    "critic_requirements_start",
                    {
                        "thread_id": state["thread_id"],
                        "current_iteration": state["current_iteration"],
                    },
                )

            requirements = await self._build_requirements_spec(state)
            logger.debug("Image requirements: " + str(requirements))

            if self.status_callback:  # STATUS_CALLBACK: critic_requirements_ready
                await self.status_callback(
                    "critic_requirements_ready",
                    {
                        "thread_id": state["thread_id"],
                        "current_iteration": state["current_iteration"],
                        "requirements": CriticAgent.object_to_markdown_text(requirements),
                        "key_points": requirements.key_points,
                        "structured_payload": requirements.model_dump(),
                    },
                )

            vlm_description = await self._describe_image_with_vlm(state, requirements)
            logger.debug("VLM Description: " + vlm_description)

            if self.status_callback:  # STATUS_CALLBACK: critic_visual_analysis_ready
                await self.status_callback(
                    "critic_visual_analysis_ready",
                    {
                        "thread_id": state["thread_id"],
                        "current_iteration": state["current_iteration"],
                        "analysis": vlm_description,
                    },
                )

            critique_payload = await self._compare_and_score(requirements, vlm_description)
            logger.debug("Critique Payload: " + str(critique_payload))

            critique = self._build_markdown_critique(critique_payload)
            quality_score = max(0.0, min(10.0, float(critique_payload.quality_score)))

            logger.debug(f"Quality Score: {quality_score:.1f}/10")
            logger.debug("\nCritique Preview:")
            logger.debug(critique[:200] + "..." if len(critique) > 200 else critique)

            if self.status_callback:  # STATUS_CALLBACK: critic_analysis_ready
                await self.status_callback(
                    "critic_analysis_ready",
                    {
                        "thread_id": state["thread_id"],
                        "critique": critique,
                        "quality_score": quality_score,
                        "current_iteration": state["current_iteration"],
                        "structured_payload": critique_payload.model_dump(),
                    },
                )

            iteration_record = IterationStep(
                iteration=state['current_iteration'],
                user_feedback=state['current_user_feedback'],
                recommendations=state["current_recommendations"],
                prompt=state['current_prompt'],
                image_path=state['current_image_path'],
                critique=critique,
                quality_score=quality_score,
                timestamp=datetime.now().isoformat(),
            )
            
            state.update({
                    "current_critique": critique,
                    "current_iteration": state['current_iteration'] + 1,
                    "current_quality_score": quality_score,
                    "iterations": state["iterations"] + [iteration_record],
                    "iterations_without_feedback": max(
                        0, state["iterations_without_feedback"] - 1
                    ),
                }
            )

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

    async def _build_requirements_spec(self, state: WorkflowState) -> CriticRequirementsSchema:
        feedback_history = "\n".join([f"Iteration {step['iteration']}: {step['user_feedback']}" for step in state['iterations'] if step['user_feedback']])

        prompt = format_requirements_prompt(
            state["user_request"],
            state.get("current_user_feedback"),
            state["current_prompt"],
            feedback_history,
        )

        response = await self.llm_client.generate(
            prompt=prompt,
            system_prompt=CRITIC_REQUIREMENTS_SYSTEM_PROMPT,
            max_tokens=settings.critic_requirements_max_tokens,
            temperature=0.2,
        )

        logger.debug("Raw requirements response: " + response)

        requirements = self._parse_with_schema(
            raw_text=response,
            schema=CriticRequirementsSchema,
            context="critic requirements",
        )
        
        logger.debug("Structured requirements extracted")
        return requirements

    async def _describe_image_with_vlm(
        self, state: WorkflowState, requirements: CriticRequirementsSchema
    ) -> str:
        
        vlm_prompt = format_vlm_description_prompt(requirements.model_dump())
        description = await self.vlm_client.analyze_image(
            image_path=state["current_image_path"],
            description_prompt=vlm_prompt,
        )
        logger.debug("VLM description generated")
        return description.strip()

    async def _compare_and_score(
        self, requirements: CriticRequirementsSchema, vlm_description: str
    ) -> CriticComparisonSchema:
        comparison_prompt = format_critic_comparison_prompt(
            requirements.model_dump(), vlm_description
        )
        response = await self.llm_client.generate(
            prompt=comparison_prompt,
            system_prompt=CRITIC_COMPARISON_SYSTEM_PROMPT,
            max_tokens=settings.critic_comparison_max_tokens,
            temperature=0.3,
        )
        critique_data = self._parse_with_schema(
            raw_text=response,
            schema=CriticComparisonSchema,
            context="critic comparison",
        )
        logger.debug("Critique comparison completed")
        return critique_data


    @staticmethod
    def object_to_markdown_text(obj: Any, indent: int = 0) -> str:
        md = ""
        prefix = " " * indent
        if isinstance(obj, BaseModel):
            obj_dict = obj.dict()
        elif isinstance(obj, dict):
            obj_dict = obj
        else:
            return f"{prefix}{obj}\n"

        for k, v in obj_dict.items():
            if isinstance(v, list):
                md += f"{prefix}- **{k}**:\n"
                for item in v:
                    md += CriticAgent.object_to_markdown_text(item, indent + 2)
            elif isinstance(v, BaseModel):
                md += f"{prefix}- **{k}**:\n"
                md += CriticAgent.object_to_markdown_text(v, indent + 2)
            else:
                md += f"{prefix}- **{k}**: {v}\n"
        return md


    def _build_markdown_critique(self, payload: CriticComparisonSchema) -> str:
        description = payload.description.strip() or "No description provided"
        correct = self._format_list_section(payload.correct_elements)
        incorrect = self._format_list_section(payload.incorrect_elements)
        improvements = self._format_list_section(payload.improvements)

        sections = [
            f"**Description:** {description}",
            f"**Correct Elements:**\n{correct}",
            f"**Incorrect Elements:**\n{incorrect}",
        ]

        if improvements.strip() and improvements.strip() != "None":
            sections.append(f"**Improvements:**\n{improvements}")

        return "\n\n".join(sections)

    @staticmethod
    def _format_list_section(values: Any) -> str:
        if isinstance(values, list):
            items = [str(value).strip() for value in values if str(value).strip()]
            if not items:
                return "None"
            return "\n".join(f"- {item}" for item in items)
        if isinstance(values, str):
            stripped = values.strip()
            return stripped if stripped else "None"
        return "None"

    def _parse_with_schema(
        self, raw_text: str, schema: Type[TModel], context: str
    ) -> TModel:
        if not raw_text:
            logger.warning(f"Empty response while parsing {context}")
            return schema()

        candidates = [raw_text]
        snippet = self._extract_json_block(raw_text)
        if snippet and snippet != raw_text:
            candidates.append(snippet)

        for candidate in candidates:
            try:
                return schema.model_validate_json(candidate)
            except ValidationError as err:
                logger.warning(f"Failed to parse {context}: {err}")

        logger.warning(f"Falling back to defaults for {context}")
        return schema()

    @staticmethod
    def _extract_json_block(text: str) -> Optional[str]:
        if not text:
            return None
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        return text[start : end + 1]

    def __str__(self):
        llm_model = getattr(self.llm_client, "model", "unknown")
        return f"CriticAgent(vlm_client={self.vlm_client.model_name}, llm_client={llm_model})"
