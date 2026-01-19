import os
import copy
import json
import uuid
import asyncio
import contextlib
from typing import Any, Dict, Optional, Callable, Awaitable

import httpx
from loguru import logger
import websockets

from src.core.config import settings
from src.models.image_gen_interface import ImageGenerationClientInterface
from urllib.parse import urlparse


class ComfyImageGenerationClient(ImageGenerationClientInterface):
    def __init__(
        self,
        base_url: str,
        workflow_template_path: str,
        save_node_id: str = "11",
        poll_interval: float = 1.5,
        poll_timeout: float = 600
    ):
        if not workflow_template_path:
            raise ValueError("workflow_template_path is required for the Comfy client")

        self.base_url = base_url.rstrip("/")
        self.workflow_template_path = workflow_template_path
        self.save_node_id = save_node_id
        self.poll_interval = poll_interval
        self.poll_timeout = poll_timeout
        self.workflow_template = self._load_workflow_template(workflow_template_path)
        self.workflow_nodes_names = self._get_nodes_names(self.workflow_template)
        self.model_name = "comfy"
        self._timeout = httpx.Timeout(15.0)
        self.status_callback: Optional[Callable[[str, Dict[str, Any]], Awaitable[None]]] = None

        logger.info("Comfy (Gen Image) клиент успешно инициализирован")

    async def generate_image(
        self,
        prompt: str,
        output_dir: Optional[str] = None,
        iteration: int = 0,
        thread_id: Optional[str] = None
    ) -> str:
        if settings.use_image_gen_stubs:
            await asyncio.sleep(2)
            return "data/images/generated_image.png"

        output_dir = output_dir or settings.output_dir
        os.makedirs(output_dir, exist_ok=True)

        workflow_payload = self._prepare_workflow(prompt)
        client_id = str(uuid.uuid4())

        logger.debug(f"Queueing prompt to Comfy for client_id={client_id}")
        response = await self._queue_prompt({"prompt": workflow_payload, "client_id": client_id})
        prompt_id = response.get("prompt_id")
        if not prompt_id:
            raise Exception("Comfy response did not include prompt_id")

        monitor_task = asyncio.create_task(self._monitor_progress(client_id, iteration, thread_id))
        try:
            history_entry = await self._wait_for_completion(prompt_id)
        finally:
            monitor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await monitor_task
        images_info = self._extract_images(history_entry)
        if not images_info:
            raise Exception("No images returned from Comfy workflow")

        image_info = images_info[0]
        raw_image = await self._fetch_image(image_info)
        saved_path = self._persist_image(image_info, raw_image, output_dir, iteration)

        logger.info(f"Comfy image saved to {saved_path}")
        return saved_path

    def set_status_callback(self, callback: Optional[Callable[[str, Dict[str, Any]], Awaitable[None]]]) -> None:
        self.status_callback = callback

    async def _queue_prompt(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}/prompt"
        print (payload)
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            return response.json()

    async def _get_history(self, prompt_id: str) -> Dict[str, Any]:
        url = f"{self.base_url}/history/{prompt_id}"
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.json()

    async def _wait_for_completion(self, prompt_id: str) -> Dict[str, Any]:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.poll_timeout
        while True:
            history = await self._get_history(prompt_id)
            entry = history.get(prompt_id)
            if entry and entry.get("status", {}).get("completed"):
                return entry

            if loop.time() >= deadline:
                raise TimeoutError("Timed out waiting for Comfy workflow to complete")

            await asyncio.sleep(self.poll_interval)

    async def _monitor_progress(
        self,
        client_id: str,
        iteration: int,
        thread_id: Optional[str]
    ) -> None:
        ws_url = self._build_ws_url(client_id)
        retry_delay = 1.0
        while True:
            try:
                async with websockets.connect(ws_url) as websocket:
                    async for message in websocket:
                        payload = json.loads(message)
                        await self._emit_progress(payload, iteration, thread_id)
                break
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning(f"Comfy websocket error: {exc}")
                await asyncio.sleep(retry_delay)
                retry_delay = min(5.0, retry_delay * 2)

    async def _emit_progress(
        self,
        payload: Dict[str, Any],
        iteration: int,
        thread_id: Optional[str]
    ) -> None:
        if not self.status_callback:
            return
        
        if payload.get("type") == "progress" or payload.get("type") == "executing":
            await self.status_callback("comfy_progress", {
                "thread_id": thread_id,
                "current_iteration": iteration,
                "payload": payload,
                "node_name": self.workflow_nodes_names.get(payload.get("data").get("node"), "unknown_node"),
            })

    def _build_ws_url(self, client_id: str) -> str:
        parsed = urlparse(self.base_url)
        scheme = "wss" if parsed.scheme == "https" else "ws"
        netloc = parsed.netloc
        path = parsed.path.rstrip("/")
        ws_path = f"{path}/ws" if path else "/ws"
        return f"{scheme}://{netloc}{ws_path}?clientId={client_id}"

    async def _fetch_image(self, image_info: Dict[str, Any]) -> bytes:
        params = {
            "filename": image_info.get("filename"),
            "subfolder": image_info.get("subfolder"),
            "type": image_info.get("type")
        }
        url = f"{self.base_url}/view"
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            return response.content

    def _persist_image(
        self,
        image_info: Dict[str, Any],
        data: bytes,
        output_dir: str,
        iteration: int
    ) -> str:
        subfolder = image_info.get("subfolder") or ""
        target_dir = os.path.join(output_dir, subfolder)
        os.makedirs(target_dir, exist_ok=True)

        base_name = image_info.get("filename") or f"comfy_iter_{iteration}.png"
        filename = f"image_iter_{iteration}_{base_name}"
        filepath = os.path.join(target_dir, filename)

        with open(filepath, "wb") as f:
            f.write(data)

        return filepath

    def _extract_images(self, history_entry: Dict[str, Any]) -> list[Dict[str, Any]]:
        outputs = history_entry.get("outputs", {})
        node_data = outputs.get(self.save_node_id, {})
        images = node_data.get("images", [])

        if images:
            return images

        for node in outputs.values():
            node_images = node.get("images")
            if node_images:
                return node_images

        return []


    def _prepare_workflow(self, prompt: str) -> Dict[str, Any]:
        workflow = copy.deepcopy(self.workflow_template)
        nodes = workflow.get("prompt", {})

        node_settings = {
            "1": {"unet_name": settings.comfy_unet_name},
            "2": {"clip_name": settings.comfy_clip_name, "type": settings.comfy_clip_type},
            "10": {"vae_name": settings.comfy_vae_name},
            "12": {"width": settings.comfy_width, "height": settings.comfy_height, "batch_size": settings.comfy_batch_size},
            "8": {
                "steps": settings.comfy_steps,
                "cfg": settings.comfy_cfg,
                "sampler_name": settings.comfy_sampler_name,
                "scheduler": settings.comfy_scheduler,
                "denoise": settings.comfy_denoise,
                "seed": settings.comfy_seed,
            },
            "6": {"text": prompt},
            "7": {"text": settings.comfy_negative_prompt},
        }

        for node_id, inputs in node_settings.items():
            for key, value in inputs.items():
                self._set_node_input(nodes, node_id, key, value)

        return nodes

    def _set_node_input(self, nodes: Dict[str, Any], node_id: str, key: str, value: Any) -> None:
        node = nodes.get(node_id)
        if not node:
            return

        inputs = node.setdefault("inputs", {})
        inputs[key] = value

    def _load_workflow_template(self, path: str) -> Dict[str, Any]:
        absolute_path = os.path.abspath(os.path.expanduser(path))
        if not os.path.exists(absolute_path):
            raise FileNotFoundError(f"Comfy workflow template not found: {absolute_path}")

        with open(absolute_path, "r", encoding="utf-8") as f:
            return json.load(f)
        
    def _get_nodes_names(self, workflow_template) -> Dict[str, str]:
        workflow_nodes_names = {}
        for node_id, node_data in workflow_template.get("prompt", {}).items(): 
            name = node_data.get("class_type", f"node_{node_id}")
            workflow_nodes_names[node_id] = name
        return workflow_nodes_names