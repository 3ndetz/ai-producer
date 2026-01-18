from .prompt_generator import PromptGeneratorNode

NODE_CLASS_MAPPINGS = {
    "PromptGenerator": PromptGeneratorNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptGenerator": "Prompt Generator",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']