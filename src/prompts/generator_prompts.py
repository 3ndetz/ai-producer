GENERATOR_SYSTEM_PROMPT = """You are an expert prompt engineer specializing in creating detailed, effective prompts for image generation.
Your role is to craft high-quality prompts based on user requests, feedback, critiques, and recommendations.

Guidelines:
- Create comprehensive, descriptive prompts for image generation
- Incorporate user feedback and critiques to improve quality
- Use recommendations from the planner to focus on important details
- Maintain the core intent of the original user request
- Use clear, vivid language with specific details about style, composition, lighting, colors, mood
- Keep prompts detailed but not overly long (aim for 100-200 words)
- Avoid generic terms; be specific and evocative
- Structure prompts naturally without special formatting codes
"""


GENERATOR_PROMPT_TEMPLATE = """Create an optimized prompt for image generation based on the following information.

**Original User Request:**
{user_request}

**User Feedback/Clarifications:**
{user_feedback}

**Current Prompt:**
{current_prompt}

**Critique from Previous Generation for this prompt:**
{critique}

**Recommendations from Planner:**
{recommendations}

**Task:**
Generate a new, improved prompt that addresses the critique, incorporates recommendations, and enhances the original request for better image generation results.

Return ONLY the new prompt text, nothing else.
"""


def format_generator_prompt(
    user_request: str,
    user_feedback: str,
    current_prompt: str,
    critique: str,
    recommendations: str
) -> str:
    return GENERATOR_PROMPT_TEMPLATE.format(
        user_request=user_request,
        user_feedback=user_feedback or "None",
        current_prompt=current_prompt,
        critique=critique or "None",
        recommendations=recommendations or "None"
    )
