import httpx
from typing import Any, Dict, Optional

from src.web.config import settings

class BackendClient:
    def __init__(self):
        self.base_url = settings.backend_url
        self.timeout = settings.http_timeout

    async def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout) as client:
            response = await client.post(path, json=payload)
            response.raise_for_status()
            return response.json()

    async def _get(self, path: str) -> Dict[str, Any]:
        async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout) as client:
            response = await client.get(path)
            response.raise_for_status()
            return response.json()

    async def start_workflow(self, prompt: str, thread_id: str) -> Dict[str, Any]:
        return await self._post('/workflow/start', {'prompt': prompt, 'thread_id': thread_id})

    async def continue_workflow(self, thread_id: str, user_feedback: Optional[str] = None) -> Dict[str, Any]:
        return await self._post(f'/workflow/{thread_id}/continue', {'user_feedback': user_feedback})

    async def stop_workflow(self, thread_id: str) -> Dict[str, Any]:
        return await self._post(f'/workflow/{thread_id}/stop', {})

    async def get_workflow_state(self, thread_id: str) -> Dict[str, Any]:
        try:
            return await self._get(f'/workflow/{thread_id}/state')
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                return {'thread_id': thread_id, 'state': {}}
            raise