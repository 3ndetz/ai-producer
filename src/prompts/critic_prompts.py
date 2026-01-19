import json
from textwrap import dedent
from typing import Any, Dict
from src.agents.critic_schemas import CriticRequirementsSchema, CriticComparisonSchema

CRITIC_REQUIREMENTS_SYSTEM_PROMPT = (
    "You are an art director who extracts structured quality checklists from prompts. "
    "Return only JSON. Do not wrap the JSON in Markdown or code fences (no ``` or ```json)."
)

CRITIC_COMPARISON_SYSTEM_PROMPT = (
    "You are a meticulous visual QA lead. Compare desired requirements with the observed image description. "
    "Highlight matches and mismatches, then assign a 0-10 quality score. Return only JSON. Do not wrap the JSON in Markdown or code fences (no ``` or ```json)." 
)


def format_requirements_prompt(
    user_request: str,
    user_feedback: str,
    current_prompt: str,
    feedback_history: str
) -> str:
    feedback = user_feedback or "None"
    schema_json = json.dumps(CriticRequirementsSchema.model_json_schema())

    return dedent(
        f"""
        You will decompose the creative brief into a structured checklist. Use ONLY the information provided in the context below. 
        Following context:

        # User request
        {user_request}

        # Latest user feedback or clarifications
        {feedback}

        # Prompt sent to the image generator
        {current_prompt}

        # Previous user feedback history
        {feedback_history}


        STRICT OUTPUT FORMAT:
        - Return only the JSON value that conforms to the schema. Do not include any additional text, explanations, headings, or separators.
        - Do not wrap the JSON in Markdown or code fences (no ``` or ```json).
        - Do not prepend or append any text (e.g., do not write "Here is the JSON:").
        - The response must be a single top-level JSON value exactly as required by the schema (object/array/etc.), with no trailing commas or comments.

        The output should be formatted as a JSON instance that conforms to the JSON schema below.

        Here is the output schema (shown for readability only — do not include any backticks or Markdown in your output):

        {schema_json}
        """
    ).strip()

def _format_requirements_prompt(structured_requirements: Dict[str, Any]) -> str:
    parts = []

    schema_properties = CriticRequirementsSchema.model_json_schema()["properties"]
    for key in schema_properties.keys():
        parts.append(f"- {key.capitalize()}")


    objs = structured_requirements.get("objects", [])
    if objs:
        parts.append("- Objects:")
        for obj in objs:
            parts.append(f"  - {obj.get('name', '')}")
    
    
    txts = structured_requirements.get("text_elements", [])
    if txts:
        parts.append("- Text elements:")
        for te in txts:
            content = te.get("content", "").strip()
            parts.append(f"  - \'{content}\' ")
    else:
        parts.append("- Text elements: None specified")

    return "\n".join(parts)
    
    

def format_vlm_description_prompt(structured_requirements: Dict[str, Any]) -> str:
    requirements = _format_requirements_prompt(structured_requirements)

    return dedent(
        f"""
        You are describing an image that was generated for the checklist below.
        Use ONLY the information provided and be concrete about what you actually see.

        # Target checklist
        {requirements}

        STRICT OUTPUT FORMAT:
        - **Overall Description**: 2 brief sentences summarizing what is visible.
        - **Style & Lighting**: short bullets mentioning art style, medium, camera, and lighting.
        - **Objects & Characters**: for each expected object from the checklist, add a bullet with:
            -- Name of object or item: what you actually see
            -- Details: (colors, pose, count, relative positions)
        - **Composition**: 1-2 sentences about viewpoint and layout.
        - **Colors & Textures**: bullets for dominant palette and materials.
        - **Text Elements**: quote exact text found or say 'None visible'.
        - **Emotions / Mood**: 1-2 sentences about atmosphere.
        - **Detailed decription**: thorough paragraph covering all notable aspects image.
        

        If an item is missing or unclear, explicitly state 'Missing: <item>' or 'Unclear: <item>'.
        Keep tone factual and concise. Do not invent details; if unsure, say 'Unclear' with what would confirm it.
        """
    ).strip()


def format_critic_comparison_prompt(
    structured_requirements: Dict[str, Any],
    vlm_description: str) -> str:
    requirements_json = json.dumps(structured_requirements, ensure_ascii=False,)
    description_block = vlm_description.strip()
    schema_json = json.dumps(CriticComparisonSchema.model_json_schema())
    return dedent(
        f"""
        Compare the desired requirements with the actual visual description.
        Determine where the image matches the brief and where it fails.
        Use ONLY the information provided in the context below. 

        # Desired requirements (JSON)
        {requirements_json}

        # Observed image description
        {description_block}

        
        STRICT OUTPUT FORMAT:
        - Return only the JSON value that conforms to the schema. Do not include any additional text, explanations, headings, or separators.
        - Do not wrap the JSON in Markdown or code fences (no ``` or ```json).
        - Do not prepend or append any text (e.g., do not write "Here is the JSON:").
        - The response must be a single top-level JSON value exactly as required by the schema (object/array/etc.), with no trailing commas or comments.

        The output should be formatted as a JSON instance that conforms to the JSON schema below.

        Here is the output schema (shown for readability only — do not include any backticks or Markdown in your output):

        {schema_json}
        """
    ).strip()
