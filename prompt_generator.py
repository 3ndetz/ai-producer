import requests
import dotenv
import os

dotenv.load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY", "")


class PromptGeneratorNode:
    def __init__(self):
        self.type = "output"
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "user_request": ("STRING", {
                    "multiline": True, 
                    "default": "a beautiful landscape with mountains and lake"
                }),
                "prompt_style": ([
                    "sdxl", "descriptive"
                ],),
                "detail_level": ([
                    "minimal", "normal", "detailed", "very detailed"
                ],),
                "api_base_url": ("STRING", {
                    "multiline": False,
                    "default": "https://api.mistral.ai/v1"
                }),
                "api_key": ("STRING", {
                    "multiline": False, 
                    "default": api_key,
                    "password": True
                }),
                "model": ([
                    "mistral-medium", 
                    "mistral-large", 
                    "mistral-small",
                    "open-mistral-7b",
                    "open-mixtral-8x7b"
                ],),
            },
            "optional": {
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "blurry, low quality, worst quality, bad anatomy"
                }),
                "custom_system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Optional custom system prompt. Leave empty for default."
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("positive_prompt", "negative_prompt")
    FUNCTION = "generate"
    CATEGORY = "prompt"
    
    def generate(self, user_request, prompt_style, detail_level, api_base_url, api_key, model, negative_prompt="", custom_system_prompt=""):
        # Генерация позитивного промпта
        positive = self._generate_with_mistral(
            user_request, prompt_style, detail_level, api_base_url, api_key, model, custom_system_prompt
        )
        
        # Стандартный негативный промпт если не задан
        if not negative_prompt.strip():
            negative_prompt = "blurry, low quality, worst quality, bad anatomy, ugly, deformed, poorly drawn"
        
        return (positive, negative_prompt)
    
    def _generate_with_mistral(self, user_request, prompt_style, detail_level, api_base_url, api_key, model, custom_system_prompt):
        """Генерация через Mistral API"""
        
        # Если API ключ не настроен, используем fallback
        if not api_key or api_key == "your-mistral-api-key-here":
            return user_request
        
        try:
            # Строим системный промпт
            system_prompt = custom_system_prompt if custom_system_prompt else self._build_system_prompt(prompt_style, detail_level)
            
            # Подготавливаем запрос для Mistral API
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_request}
                ],
                "temperature": 0.7,
                "max_tokens": 500,
                "top_p": 0.8
            }
            
            # Отправляем запрос
            response = requests.post(
                f"{api_base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                prompt = result["choices"][0]["message"]["content"].strip()
                return self._clean_prompt(prompt)
            else:
                print(f"Mistral API error: {response.status_code} - {response.text}")
                return user_request
                
        except Exception as e:
            print(f"Mistral generation failed: {e}")
            return user_request
    
    def _build_system_prompt(self, prompt_style, detail_level):
        """Строит системный промпт для Mistral"""
        
        detail_instructions = {
            "minimal": "25-40 words, concise and direct",
            "normal": "40-70 words, good detail level", 
            "detailed": "70-100 words, high detail with specific elements",
            "very detailed": "100-150 words, extremely detailed with textures, lighting, and atmosphere"
        }
        

        prompts = {
            "sdxl": 
f"""You are an expert SDXL prompt generator for AI image generation.

CRITICAL RULES:
1. PRESERVE CORE CONCEPT: Faithfully maintain ALL key elements from user's original idea
2. Output format: SDXL weighted prompt syntax (concept:weight, concept:weight)
3. Base weight: 1.0 (default, not shown)
4. Weights range: 0.1 to 2.0 for emphasis/de-emphasis
5. Detail level: {detail_instructions[detail_level]}
6. NO explanations, NO markdown, NO prefixes

SDXL SPECIFIC RULES:
- Separate concepts with commas
- Use weights to prioritize important elements from user input
- Structure: (main_subject:1.5), details, environment, artistic_qualities
- Higher weights for user-specified key elements
- Technical terms after style, lower weights for subtle effects

CONCEPT PRESERVATION:
- Identify 2-3 core elements from user input (must be included)
- Assign higher weights (1.3-1.8) to these core elements
- Add supporting details with moderate weights (0.8-1.2)
- Artistic/style elements: consistent but secondary to core concept

Example output: "(dragon:1.7), (iridescent scales:1.3), (mountain peak:1.2), (sunrise:1.4), swirling mists, (epic fantasy art:1.5), (highly detailed:1.3), (cinematic lighting:1.2), (golden hour:1.1), digital painting"

Now generate the SDXL prompt based on this concept:""",
            
            "descriptive" : 
f"""You are an expert prompt generator for AI image generation.

CRITICAL RULES:
1. PRESERVE CORE CONCEPT: Faithfully maintain the user's original intent and main subject
2. Output ONLY the prompt text, no explanations or markdown
3. Language: Natural, descriptive English
4. Detail level: {detail_instructions[detail_level]}
5. Structure: Subject + environment + composition + lighting + mood + artistic details
6. NO prefixes like "Prompt:" or quotes

CONCEPT PRESERVATION RULES:
- Identify and retain the core subject from user input
- Enhance but never replace or contradict the original concept
- Add complementary details that support the main idea
- If uncertain about details, prioritize maintaining the core subject

Example output: "A majestic dragon with iridescent scales perched on a mountain peak at sunrise, surrounded by swirling mists. Epic fantasy art, highly detailed, cinematic lighting, golden hour atmosphere."

Now generate the prompt based on this concept:"""
        }

        return prompts[prompt_style]
    
    def _clean_prompt(self, prompt):
        """Очищает промпт от нежелательного форматирования"""
        # Удаляем возможные префиксы и форматирование
        unwanted_prefixes = [
            "Prompt:", "SDXL:", "Image:", "Here is the prompt:", 
            "```", "```json", "**", "###", "---"
        ]
        
        for prefix in unwanted_prefixes:
            if prompt.startswith(prefix):
                prompt = prompt[len(prefix):].strip()
        
        # Удаляем кавычки и лишние пробелы
        prompt = prompt.replace('"', '').replace("'", "")
        prompt = ' '.join(prompt.split())  # Нормализуем пробелы
        
        return prompt.strip()
