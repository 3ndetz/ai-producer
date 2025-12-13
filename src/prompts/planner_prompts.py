PLANNER_SYSTEM_PROMPT = """You are an expert prompt engineer specializing in image generation.
Your role is to evaluate the current state based on critique feedback and decide whether to continue the generation process.

Guidelines:
- Assess if the issues mentioned in the critique are solvable
- Check if problems repeat across iterations
- Decide whether to continue generation or stop
- Provide specific recommendations for important details to focus on in generation
- Return a JSON object with 'continue': boolean and 'recommendations': string
"""


PLANNER_REFINEMENT_TEMPLATE = """Evaluate the current state and decide on continuing image generation.

**Original User Request:**
{user_request}

**User Feedback:** {current_user_feedback}

**Current Prompt for Generation:**
{current_prompt}

**Current Critique:**
{current_critique}

**Quality Score:** {quality_score}/10

**Critique History:**
{critique_history}

**Feedback History:**
{feedback_history}

**Task:**
- Assess if the problems are solvable and if they repeat.
- Decide whether to continue generation.
- Pay attention to what frequently recurres in Critique in each iteration.
- Provide recommendations on what to pay special attention to during generation to accurately follow the user's original Request and Feedback.

Return ONLY a JSON object in this format:
{{
  "continue": true/false,
  "recommendations": "Detailed recommendations here"
}}
"""


def format_planner_prompt(
    user_request: str,
    current_user_feedback: str,
    current_prompt: str,
    current_critique: str,
    quality_score: float,
    critique_history: str,
    feedback_history: str
) -> str:
    return PLANNER_REFINEMENT_TEMPLATE.format(
        user_request=user_request,
        current_user_feedback=current_user_feedback,
        current_prompt=current_prompt,
        current_critique=current_critique,
        quality_score=quality_score,
        critique_history=critique_history,
        feedback_history=feedback_history
    )
