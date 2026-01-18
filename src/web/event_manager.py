import asyncio
import contextlib
import json
import websockets
from typing import Dict, Optional
from loguru import logger

from src.web.config import settings


class EventManager:
    def __init__(self):
        self.listener_tasks: Dict[str, asyncio.Task] = {}

    async def _stop_event_listener(self, thread_id: Optional[str]) -> None:
        if not thread_id:
            return
        task = self.listener_tasks.pop(thread_id, None)
        if not task:
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    async def _event_listener(self, thread_id: str, status_callback) -> None:
        ws_url = f"{settings.backend_ws_url}/workflow/{thread_id}/ws"
        retry_delay = 1
        while True:
            try:
                async with websockets.connect(ws_url) as websocket:
                    async for message in websocket:
                        payload = json.loads(message)
                        event_name = payload.get('event')
                        data = payload.get('data') or {}
                        await status_callback(event_name, data)
                retry_delay = 1
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning(f"WebSocket for {thread_id} failed: {exc}")
                await asyncio.sleep(retry_delay)
                retry_delay = min(30, retry_delay * 2)

    async def _ensure_event_listener(self, thread_id: str, status_callback) -> None:
        task = self.listener_tasks.get(thread_id)
        if task and not task.done():
            return
        task = asyncio.create_task(self._event_listener(thread_id, status_callback))
        self.listener_tasks[thread_id] = task