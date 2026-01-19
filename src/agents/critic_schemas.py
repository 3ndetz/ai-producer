from typing import List, Literal
from pydantic import BaseModel, Field


class ObjectRequirement(BaseModel):
    name: str = Field(
        default="",
        description="Name of the object or entity that should appear in the image (e.g., 'Elephant', 'Laptop')."
    )
    description: str = Field(
        default="",
        description="Detailed expected attributes of the object: pose, color, count, size, accessories, relative position, and other distinguishing details."
    )
    priority: Literal["high", "medium", "low"] = Field(
        default="medium",
        description="Priority assigned to this object; 'high' items are critical and should heavily affect the quality score if missing."
    )


class TextElementRequirement(BaseModel):
    content: str = Field(
        default="",
        description="Exact text expected to appear in the image (quote the text verbatim if known)."
    )
    placement: str = Field(
        default="",
        description="Expected placement or region for the text (e.g., 'top-left on banner', 'on laptop screen', or 'none' if not required')."
    )


class CriticRequirementsSchema(BaseModel):
    style: str = Field(
        default="",
        description="Overall art direction and medium"
    )
    composition: str = Field(
        default="",
        description="Viewpoint, framing, and layout cues"
    )
    colors: str = Field(
        default="",
        description="Dominant color palette and contrast cues"
    )
    lighting: str = Field(
        default="",
        description="Lighting type and overall lighting mood"
    )
    emotions: str = Field(
        default="",
        description="Desired atmosphere or emotions conveyed by the image"
    )
    objects: List[ObjectRequirement] = Field(
        default_factory=list,
        description=(
            "List of objects and details that should appear in the image: "
            "{ name: string, description: string, priority: high|medium|low }."
        )
    )
    text_elements: List[TextElementRequirement] = Field(
        default_factory=list,
        description=(
            "List of text elements that should be displayed in the image: "
            "{ content: string, placement: string }."
        )
    )
    additional_requirements: str = Field(
        default="",
        description="Other constraints such as rendering quality, camera settings, or visual effects"
    )
    key_points: List[str] = Field(
        default_factory=list,
        description="Ordered list of 3–6 must-have checkpoints that are critical for evaluation"
    )



class CriticComparisonSchema(BaseModel):
    description: str = Field(
        default="No description provided",
        description="2-sentence summary of what the image actually shows"
    )
    correct_elements: List[str] = Field(
        default_factory=list,
        description="Bullet-style sentences for elements that match the desired requirements"
    )
    incorrect_elements: List[str] = Field(
        default_factory=list,
        description="Bullet-style sentences for mismatches or missing items"
    )
    improvements: List[str] = Field(
        default_factory=list,
        description="Actionable guidance to fix the issues"
    )
    quality_score: float = Field(
        default=5.0,
        ge=0.0,
        le=10.0,
        description="Quality score from 0 to 10, considering priority from the checklist"
    )