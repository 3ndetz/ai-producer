import asyncio
import json
import uuid
import uvicorn
from loguru import logger
from pydantic import BaseModel
from src.core.config import settings
from typing import Any, Dict, List, Optional
from sse_starlette.sse import EventSourceResponse
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect

from src.core.workflow import ImageGenerationWorkflow, create_workflow


class WorkflowEventBroker:
    def __init__(self) -> None:
        self._queues: Dict[str, List[asyncio.Queue]] = {}
        self._history: Dict[str, List[Dict[str, Any]]] = {}
        self._lock = asyncio.Lock()

    async def publish(self, thread_id: str, event: Dict[str, Any]) -> None:
        """Публикует событие и сохраняет последнее состояние для новых подписчиков."""
        history = self._history.setdefault(thread_id, [])
        history.append(event)
        if len(history) > 100:
            history.pop(0)

        async with self._lock:
            queues = list(self._queues.get(thread_id, []))

        for queue in queues:
            try:
                await queue.put(event)
            except asyncio.CancelledError:
                await self.unregister(thread_id, queue)

    async def unregister(self, thread_id: str, queue: asyncio.Queue) -> None:
        """Удаляет очередь по завершению SSE-подписки."""
        async with self._lock:
            queues = self._queues.get(thread_id)
            if not queues:
                return
            if queue in queues:
                queues.remove(queue)
            if not queues:
                self._queues.pop(thread_id, None)

    async def register(self, thread_id: str) -> asyncio.Queue:
        """Регистрирует новую очередь для thread_id и возвращает её."""
        queue: asyncio.Queue = asyncio.Queue()
        async with self._lock:
            self._queues.setdefault(thread_id, []).append(queue)
        return queue

    def get_history(self, thread_id: str) -> List[Dict[str, Any]]:
        return list(self._history.get(thread_id, []))

    def get_thread_ids(self) -> List[str]:
        return list(self._history.keys())


class WorkflowService:
    def __init__(self) -> None:
        self.workflow: Optional[ImageGenerationWorkflow] = None
        self.event_broker = WorkflowEventBroker()
        self._init_lock = asyncio.Lock()
        self._tasks: Dict[str, asyncio.Task] = {}

    async def initialize(self) -> None:
        if self.workflow is not None:
            return
        async with self._init_lock:
            if self.workflow is not None:
                return
            workflow = await create_workflow()
            workflow.set_status_callback(self._status_callback)
            self.workflow = workflow
            logger.info("Workflow initialized")

    async def _status_callback(self, status_type: str, data: Dict[str, Any]) -> None:
        thread_id = data.get("thread_id") or data.get("configurable", {}).get("thread_id") or "default"
        event = {"status": status_type, "payload": data}
        logger.info (f"Publishing event: {event}")
        await self.event_broker.publish(thread_id, event)

    def is_thread_active(self, thread_id: str) -> bool:
        task = self._tasks.get(thread_id)
        return bool(task and not task.done())

    def get_active_thread_ids(self) -> List[str]:
        return [thread_id for thread_id, task in self._tasks.items() if task and not task.done()]

    async def launch_workflow(self, prompt: str, thread_id: str) -> None:
        if self.workflow is None:
            raise RuntimeError("Workflow not initialized")

        async def _runner() -> None:
            try:
                await self.workflow.start_generation(prompt, thread_id)
            except asyncio.CancelledError:
                logger.info(f"Workflow task for {thread_id} cancelled")
                raise
            except Exception as exc:
                logger.exception(f"Workflow task for {thread_id} failed: {exc}")
                await self._emit_service_error(thread_id, exc)
                raise

        task = asyncio.create_task(_runner(), name=f"workflow-{thread_id}")
        self._tasks[thread_id] = task

        def _cleanup(fut: asyncio.Task, *, thread: str = thread_id) -> None:
            self._tasks.pop(thread, None)
            if fut.cancelled():
                return
            exc = fut.exception()
            if exc:
                logger.error(f"Workflow task for {thread} finished with error: {exc}")

        task.add_done_callback(_cleanup)

    async def _emit_service_error(self, thread_id: str, exc: Exception) -> None:
        if not self.workflow or not self.workflow.status_callback:
            return
        try:
            await self.workflow.status_callback(
                "error",
                {
                    "thread_id": thread_id,
                    "agent": "Workflow",
                    "error_message": str(exc),
                },
            )
        except Exception as callback_exc:
            logger.error(f"Failed to emit workflow error for {thread_id}: {callback_exc}")

    def get_workflow(self) -> ImageGenerationWorkflow:
        if self.workflow is None:
            raise RuntimeError("Workflow not initialized")
        return self.workflow


class StartRequest(BaseModel):
    prompt: str
    thread_id: Optional[str] = None


class ContinueRequest(BaseModel):
    user_feedback: Optional[str] = None


app = FastAPI(title="AI Producer Workflow API")
service = WorkflowService()


@app.on_event("startup")
async def _startup() -> None:
    await service.initialize()


@app.get("/health")
async def health_check() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/workflow/start")
async def start_workflow(request: StartRequest) -> Dict[str, Any]:
    await service.initialize()
    thread_id = request.thread_id or str(uuid.uuid4())
    if service.is_thread_active(thread_id):
        raise HTTPException(status_code=409, detail="Workflow already running for this thread")
    logger.debug(f"Scheduling workflow for thread {thread_id}")
    await service.launch_workflow(request.prompt, thread_id)
    await service._status_callback( # STATUS_CALLBACK: starting
        "starting",
        {
            "thread_id": thread_id,
            "prompt": request.prompt
        },
    )
    return {"thread_id": thread_id, "status": "started"}


@app.post("/workflow/{thread_id}/continue")
async def continue_workflow(thread_id: str, request: ContinueRequest) -> Dict[str, Any]:
    logger.debug(f"Continuing workflow for thread {thread_id}")
    workflow = service.get_workflow()
    result = await workflow.continue_generation(thread_id, request.user_feedback)
    return {"thread_id": thread_id, "state": result}


@app.post("/workflow/{thread_id}/stop")
async def stop_workflow(thread_id: str) -> Dict[str, str]:
    logger.debug(f"Stopping workflow for thread {thread_id}")
    workflow = service.get_workflow()
    await workflow.stop_generation(thread_id)
    return {"thread_id": thread_id, "status": "stopped"}


@app.get("/workflow/{thread_id}/state")
async def get_state(thread_id: str) -> Dict[str, Any]:
    workflow = service.get_workflow()
    state = await workflow.get_current_state(thread_id)
    if not state:
        raise HTTPException(status_code=404, detail="Thread not found")
    return {"thread_id": thread_id, "state": state}


@app.get("/workflow/{thread_id}/history")
async def get_event_history(thread_id: str) -> Dict[str, Any]:
    history = service.event_broker.get_history(thread_id)
    return {"thread_id": thread_id, "events": history}


@app.get("/workflow/threads")
async def get_threads() -> Dict[str, List[str]]:
    threads = set(service.event_broker.get_thread_ids())
    threads.update(service.get_active_thread_ids())
    return {"threads": sorted(threads)}


@app.get("/workflow/{thread_id}/events")
async def stream_events(thread_id: str) -> EventSourceResponse:
    queue = await service.event_broker.register(thread_id)

    async def event_generator() -> Any:
        history = service.event_broker.get_history(thread_id)
        for event in history:
            yield {"event": event["status"], "data": json.dumps(event["payload"])}
        try:
            while True:
                event = await queue.get()
                yield {"event": event["status"], "data": json.dumps(event["payload"])}
        finally:
            await service.event_broker.unregister(thread_id, queue)

    return EventSourceResponse(event_generator())


@app.websocket("/workflow/{thread_id}/ws")
async def websocket_events(thread_id: str, websocket: WebSocket) -> None:
    await websocket.accept()
    queue = await service.event_broker.register(thread_id)

    try:
        history = service.event_broker.get_history(thread_id)
        for event in history:
            await websocket.send_json({"event": event["status"], "data": event["payload"]})

        while True:
            event = await queue.get()
            await websocket.send_json({"event": event["status"], "data": event["payload"]})
    except WebSocketDisconnect:
        logger.debug(f"WebSocket disconnected for thread {thread_id}")
    finally:
        await service.event_broker.unregister(thread_id, queue)


if __name__ == "__main__":
    logger.info(f"Backend host: {settings.backend_host}, port: {settings.backend_port}")
    uvicorn.run(app, host=settings.backend_host, port=settings.backend_port, log_level="info")
