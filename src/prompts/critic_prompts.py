CRITIC_ANALYSIS_TEMPLATE = """You are an expert image quality analyst and art critic. 
Your task is to analyze the generated image and provide detailed, structured feedback. 
Follow ALL the rules below strictly.

Your evaluation MUST include:

1. Alignment with the original user request and current prompt
2. Accuracy of depicted elements compared to the text prompts
3. Overall quality, composition, clarity, and coherence of the image
4. A clear identification of what matches the prompt and what does NOT
5. A final quality score from 0 to 10 (10 = perfect alignment and quality)

Analyze this image that was generated from the following prompts:

**Original User Request:** {user_request}

**User Feedback/Clarifications:** {user_feedback}

**Current Generation Prompt:** {current_prompt}

Provide your answer STRICTLY in the following format:

**Description:** [Detailed description of the generated image]

**Correct Elements:** [What was depicted correctly according to the prompts]

**Incorrect Elements:** [What was depicted incorrectly or is missing]

**Quality Score:** [Score from 0-10]

Do NOT add anything outside of this structure.
"""


def format_critic_prompt(user_request: str, user_feedback: str, current_prompt: str) -> str:
    return CRITIC_ANALYSIS_TEMPLATE.format(
        user_request=user_request,
        user_feedback=user_feedback or "None",
        current_prompt=current_prompt
    )
