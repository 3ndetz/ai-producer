import aiohttp
import json
import os
from datetime import datetime
from loguru import logger
from typing import Dict
from src.core.config import settings
import time
import asyncio


def ui_workflow_to_api_prompt(ui: dict) -> dict:
    nodes = {n["id"]: n for n in ui["nodes"]}
    links = ui["links"]

    api = {}

    def find_link(link_id):
        for l in links:
            if l[0] == link_id:
                return str(l[1]), l[2]
        return None

    for node_id, node in nodes.items():
        ctype = node["type"]
        api_node = {"class_type": ctype, "inputs": {}}

        # 1. inputs from links
        for inp in node.get("inputs", []):
            if inp.get("link") is not None:
                src = find_link(inp["link"])
                if src:
                    api_node["inputs"][inp["name"]] = src

        w = node.get("widgets_values", [])

        # 2. widgets → inputs (ПО ТИПУ УЗЛА)
        if ctype == "CLIPTextEncode":
            api_node["inputs"]["text"] = w[0] if w else ""

        elif ctype == "UNETLoader":
            api_node["inputs"]["unet_name"] = w[0]
            api_node["inputs"]["weight_dtype"] = w[1]

        elif ctype == "VAELoader":
            api_node["inputs"]["vae_name"] = w[0]

        elif ctype == "CLIPLoader":
            api_node["inputs"]["clip_name"] = w[0]
            api_node["inputs"]["type"] = w[1]
            api_node["inputs"]["device"] = w[2]

        elif ctype == "ModelSamplingAuraFlow":
            api_node["inputs"]["shift"] = w[0]

        elif ctype == "EmptySD3LatentImage":
            api_node["inputs"]["width"] = w[0]
            api_node["inputs"]["height"] = w[1]
            api_node["inputs"]["batch_size"] = w[2]

        elif ctype == "KSampler":
            api_node["inputs"].update({
                "seed": w[0],
                "steps": w[2],
                "cfg": w[3],
                "sampler_name": w[4],
                "scheduler": w[5],
                "denoise": w[6],
            })

        elif ctype == "SaveImage":
            api_node["inputs"]["filename_prefix"] = w[0]

        api[str(node_id)] = api_node

    return api




class ComfyUIClient:
    def __init__(self):
        self.base_url = settings.comfyui_base_url.rstrip("/")
        self.workflow_path = settings.comfyui_workflow_path

        with open(self.workflow_path, "r", encoding="utf-8") as f:
            self.workflow = json.load(f)

        logger.info("ComfyUI клиент инициализирован")

    def _inject_prompt(
        self,
        prompt: str,
        negative_prompt: str | None = None,
        seed: int | None = None,
        steps: int | None = None,
        cfg: float | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> dict:
        workflow = json.loads(json.dumps(self.workflow))  # deep copy

        for node in workflow.get("nodes", []):
            ntype = node.get("type")
            title = (node.get("title") or "").lower()

            # ---- PROMPTS via widgets_values ----
            if ntype == "CLIPTextEncode":
                if "positive" in title:
                    node.setdefault("widgets_values", [""])
                    node["widgets_values"][0] = prompt
                elif "negative" in title and negative_prompt is not None:
                    node.setdefault("widgets_values", [""])
                    node["widgets_values"][0] = negative_prompt

            # ---- KSampler params via widgets_values (позиционно!) ----
            elif ntype == "KSampler":
                w = node.setdefault("widgets_values", [])
                # у тебя сейчас widgets_values = [seed, "fixed", steps, cfg, sampler, scheduler, denoise]
                # индексы:
                # 0 seed, 2 steps, 3 cfg, 6 denoise
                if len(w) < 7:
                    # если вдруг сломанный workflow — не молча, а явно:
                    raise ValueError(f"KSampler widgets_values too short: {w}")
                if seed is not None:
                    w[0] = seed
                if steps is not None:
                    w[2] = steps
                if cfg is not None:
                    w[3] = cfg

            # ---- SIZE via widgets_values ----
            elif ntype == "EmptySD3LatentImage":
                w = node.setdefault("widgets_values", [])
                # widgets_values = [width, height, batch_size]
                if len(w) < 3:
                    raise ValueError(f"EmptySD3LatentImage widgets_values too short: {w}")
                if width is not None:
                    w[0] = width
                if height is not None:
                    w[1] = height

        return workflow



    async def generate_image(
        self,
        prompt: str,
        output_dir: str,
        iteration: int
    ) -> str:

        os.makedirs(output_dir, exist_ok=True)
        workflow = self._inject_prompt(prompt)

        api_prompt = ui_workflow_to_api_prompt(workflow)

        async with aiohttp.ClientSession() as session:
            # 1. отправляем workflow
            async with session.post(
                f"{self.base_url}/prompt",
                json={"prompt": api_prompt},
            ) as resp:
                data = await resp.json()

            if "prompt_id" not in data:
                raise RuntimeError(
                    f"ComfyUI rejected prompt. Response: {json.dumps(data, indent=2, ensure_ascii=False)}"
                )

            prompt_id = data["prompt_id"]


            logger.info(f"ComfyUI prompt_id={prompt_id}")

            # 2. ждём выполнения
            start = time.time()
            while True:
                async with session.get(f"{self.base_url}/history/{prompt_id}") as r:
                    history = await r.json()

                if prompt_id in history:
                    break

                if time.time() - start > settings.comfyui_timeout:
                    raise TimeoutError("ComfyUI generation timeout")

                await asyncio.sleep(0.5)


            outputs = history[prompt_id]["outputs"]

            # 3. ищем изображение
            for node_output in outputs.values():
                if "images" in node_output:
                    image_info = node_output["images"][0]

                    async with session.get(
                        f"{self.base_url}/view",
                        params=image_info
                    ) as img_resp:
                        img_bytes = await img_resp.read()

                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"image_iter_{timestamp}_{iteration}.png"
                    filepath = os.path.join(output_dir, filename)

                    with open(filepath, "wb") as f:
                        f.write(img_bytes)

                    logger.info(f"Image saved: {filepath}")
                    return filepath

        raise RuntimeError("ComfyUI: image not found in outputs")
