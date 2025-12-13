import httpx
from loguru import logger
from typing import Optional
from src.core.config import settings

try:
    from mlx_lm import load, generate
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

class LLMClient:    
    def __init__(self, provider: str = settings.llm_model_type, 
                       api_key: Optional[str] = None,
                       model: Optional[str] = None):
        self.provider = provider
        
        if provider == "openrouter":
            self.api_key = api_key or settings.openrouter_api_key
            self.base_url = "https://openrouter.ai/api/v1"
            self.model = model or settings.llm_model_openrouter
            logger.info("OpenRouter API (LLM) клиент успешно инициализирован")
        elif provider == "mistral":
            self.api_key = api_key or settings.mistral_api_key
            self.base_url = "https://api.mistral.ai/v1"
            self.model = model or settings.llm_model_mistral
            logger.info("Mistral API (LLM) клиент успешно инициализирован")
        elif provider == "local":
            if not MLX_AVAILABLE:
                raise ImportError("mlx_lm is not installed. Install it or use mode='openrouter'/'mistral'")
            self.model, self.tokenizer = load(model or settings.llm_model_local)
            self.api_key = None
            self.base_url = None
            logger.info(f"LLM модель '{model or settings.llm_model_local}' успешно загружена (MLX)")
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 500,
        temperature: float = 0.7
    ) -> str:
        if self.provider == "local":
            return await self._generate_local(prompt, system_prompt, max_tokens)
        else:
            return await self._generate_api(prompt, system_prompt, max_tokens, temperature)
    
    async def _generate_api(
        self,
        prompt: str,
        system_prompt: Optional[str],
        max_tokens: int,
        temperature: float
    ) -> str:
        try:
            if settings.use_llm_stubs:
                import asyncio
                await asyncio.sleep(1)
                return "Stub generated text"
        
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                
                result = response.json()
                generated_text = result["choices"][0]["message"]["content"]
                
                return generated_text.strip()
                
        except Exception as e:
            error_msg = f"Error generating with LLM: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)
    
    async def _generate_local(
        self,
        prompt: str,
        system_prompt: Optional[str],
        max_tokens: int
    ) -> str:
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            chat_prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True
            )
            result = generate(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    prompt=chat_prompt,
                    max_tokens=max_tokens,
                    verbose=False
                )

            return result["text"].strip()
            
        except Exception as e:
            error_msg = f"Error with local LLM: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)
    
    
    async def is_available(self) -> bool:
        if self.provider == "local":
            return self.model is not None and self.tokenizer is not None
        else:
            try:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.get(
                        f"{self.base_url}/models",
                        headers=headers
                    )
                    if response.status_code == 200:
                        available_models = response.json().get("data", [])
                        model_names = [m["id"] for m in available_models]
                        if self.model in model_names:
                            return True
                        else:
                            logger.warning(f"Модель {self.model} недоступна у провайдера {self.provider}")
                            return False
                    else:
                        logger.warning(f"Ошибка при проверке модели: {response.status_code}")
                        return False
            except Exception as e:
                logger.warning(f"LLM availability check failed: {str(e)}")
                return False
