import asyncio
import contextlib
import json
import os
import sys
import uuid
from typing import Any, Dict, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import chainlit as cl
from chainlit import Image

import httpx
import websockets
from loguru import logger
from prompts.web_messages import (
    AGENT_ERROR_MESSAGE_TEMPLATE,
    INITIALIZING_MESSAGE,
    SYSTEM_READY_MESSAGE,
)

BACKEND_URL = os.getenv('BACKEND_URL', 'http://localhost:8011')


def _build_ws_url(base: str) -> str:
    if base.startswith('https://'):
        return base.replace('https://', 'wss://', 1).rstrip('/')
    if base.startswith('http://'):
        return base.replace('http://', 'ws://', 1).rstrip('/')
    return base.rstrip('/')


BACKEND_WS_URL = _build_ws_url(BACKEND_URL)
HTTP_TIMEOUT = 120

listener_tasks: Dict[str, asyncio.Task] = {}


async def _stop_event_listener(thread_id: Optional[str]) -> None:
    if not thread_id:
        return
    task = listener_tasks.pop(thread_id, None)
    if not task:
        return
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


async def status_callback(status_type: str, data: Dict[str, Any]) -> None:
    thread_id = (
        data.get('thread_id')
        or data.get('configurable', {}).get('thread_id')
        or getattr(cl.user_session, 'thread_id', None)
    )

    iteration = (data.get('current_iteration') or 0)

    logger.info(f"status_callback: {cl.user_session.get('id')}, status_type={status_type}, iteration={iteration}, thread_id={thread_id}")

    if not hasattr(cl.user_session, 'current_steps'):
        cl.user_session.current_steps = {}
    if not hasattr(cl.user_session, 'iteration_messages'):
        cl.user_session.iteration_messages = {}
    if not hasattr(cl.user_session, 'awaiting_feedback'):
        cl.user_session.awaiting_feedback = False

    # Planner
    if status_type == 'planning_continuation':
        cl.user_session.iteration_messages[f"planner_{iteration}"] = await cl.Message(
            content=f"\n---\n #### 🧠 Iteration {iteration}: Planner\n"
        ).send()

        step = cl.Step(name='Should the generation continue?', type='llm')
        cl.user_session.current_steps[f"planning_{iteration}"] = step
        await step.__aenter__()
        step.output = '🧠 The agent checks if quality improves each iteration and decides whether to continue....'
        await step.update() 

    elif status_type == 'decision_about_continuation':
        key = f"planning_{iteration}"
        if key in cl.user_session.current_steps:
            step = cl.user_session.current_steps[key]
            continue_text = 'continue generation!' if data.get('continue_generation', False) else 'complete generation!'
            recommendations = data.get('current_recommendations', '')
            step.output = f"The agent decided to {continue_text} Planner recommendations: {recommendations}"
            await step.__aexit__(None, None, None)


    # Generator
    elif status_type == 'prompt_generation':
        cl.user_session.iteration_messages[f"prompt_generation_{iteration}"] = await cl.Message(
            content=f"\n---\n #### 🎨 Iteration {iteration}: Generator\n"
        ).send()

        step = cl.Step(name='Planning next prompt...', type='llm')
        cl.user_session.current_steps[f"prompt_generation_{iteration}"] = step
        await step.__aenter__()
        step.output = '🧠 Analyzing feedback and planning improvements...'
        await step.update()

    elif status_type == 'prompt_ready':
        key = f"prompt_generation_{iteration}"
        if key in cl.user_session.current_steps:
            step = cl.user_session.current_steps[key]
            step.output = f"✓ New prompt planned:\n\n*{data.get('prompt', '-')}*"
            await step.__aexit__(None, None, None)

    elif status_type == 'image_generation':
        step = cl.Step(name='Generating image...', type='tool')
        cl.user_session.current_steps[f"image_generation_{iteration}"] = step
        await step.__aenter__()
        step.output = '🎨 Image generation in progress...'
        await step.update()

    elif status_type == 'image_ready':
        key = f"image_generation_{iteration}"
        if key in cl.user_session.current_steps:
            step = cl.user_session.current_steps[key]
            step.output = '✓ Image generated successfully!'
            await step.__aexit__(None, None, None)

        with open(data.get('image_path', ''), "rb") as f:
            image_bytes = f.read()
        image = Image(content=image_bytes, 
                      display='inline',
                       name=f"iteration_{iteration}",
                      mime="image/png")

        await cl.Message(content='**✓ Image generated:**', elements=[image]).send()

    # Critic
    elif status_type == 'analyzing_image':
        cl.user_session.iteration_messages[f"critic_{iteration}"] = await cl.Message(
            content=f"\n---\n#### 🔍 Iteration {iteration}: Critic\n"
        ).send()

        step = cl.Step(name='Analyzing image...', type='tool')
        cl.user_session.current_steps[f"analyzing_image_{iteration}"] = step
        await step.__aenter__()
        step.output = '🔍 Analyzing image quality...'
        await step.update()

    elif status_type == 'analysis_ready':
        key = f"analyzing_image_{iteration}"
        if key in cl.user_session.current_steps:
            step = cl.user_session.current_steps[key]
            await step.__aexit__(None, None, None)

        step = cl.Step(name='Critique', type='tool')
        await step.__aenter__()
        step.output = f"**Critique**:\n{data.get('critique', '')}\n\n"
        await step.__aexit__(None, None, None)

        critic_key = f"critic_{iteration}"
        if critic_key in cl.user_session.iteration_messages:
            crit_msg = cl.user_session.iteration_messages[critic_key]
            crit_msg.content += f"**Quality Score**: {data.get('quality_score', 0):.1f}/10\n\n"
            await crit_msg.update()


    # Ожидание feedback от пользователя
    elif status_type == 'feedback_interrupt':
        actions = [
            cl.Action(
                name='continue_generation',
                icon='mouse-pointer-click',
                payload={'value': 'yes'},
                label='Continue',
            ),
            cl.Action(
                name='stop_generation',
                icon='mouse-pointer-click',
                payload={'value': 'no'},
                label='Stop',
            ),
        ]
        interrupt_message = cl.Message(
            content="#### ⏸️ **Interrupt: Waiting for Decision**\nChoose 'Continue' or 'Stop', or provide more details.",
            actions=actions,
        )
        cl.user_session.interrupt_message = interrupt_message
        cl.user_session.awaiting_feedback = True
        await interrupt_message.send()

    # Сообщение об ошибке от конкретного агента
    elif status_type == 'error':
        agent = data.get('agent', 'Unknown')
        error_message = data.get('error_message', 'Unknown error')
        step_key = {
            'Generator': f"generating_{iteration}",
            'Critic': f"analyzing_{iteration}",
            'Planner': f"planning_{iteration}",
        }.get(agent)

        if step_key and step_key in cl.user_session.current_steps:
            step = cl.user_session.current_steps[step_key]
            step.output = f"❌ **{agent} Failed**\n\n{error_message}"
            step.is_error = True
            await step.__aexit__(None, None, None)

        error_details = AGENT_ERROR_MESSAGE_TEMPLATE.format(agent, iteration, error_message)
        await cl.Message(content=error_details).send()
        cl.user_session.awaiting_feedback = False
        setattr(cl.user_session, 'interrupt_message', None)
        await _stop_event_listener(thread_id)


    # Режим ожидания при возникновении ошибки
    elif status_type == 'error_interrupt':
        await cl.Message(
            content=f"\n An error has occurred. Please repeat your request. The generation process will start again."
        ).send()
        cl.user_session.thread_id = None
        cl.user_session.awaiting_feedback = False
        cl.user_session.interrupt_message = None

    # Режим ожидания при завершении работы графа после Планировщика
    elif status_type == 'end_planner_interrupt':
        await cl.Message(
            content=f"\n Agent Planner decided to terminate the graph execution. Current quality score: {data.get('current_quality_score', 0):.1f}/10. \
            You may provide additional details and then continue the generation process."
        ).send()
        cl.user_session.awaiting_feedback = True


    # Режим ожидания при завершении работы графа после Критика
    elif status_type == 'end_critic_interrupt':
        await cl.Message(
            content=f"\n Workflow Complete! Agent Critic decided to terminate the graph execution. Current quality score: {data.get('current_quality_score', 0):.1f}/10. \
            You may provide additional details and then continue the generation process."
        ).send()
        cl.user_session.awaiting_feedback = True




async def _event_listener(thread_id: str) -> None:
    ws_url = f"{BACKEND_WS_URL}/workflow/{thread_id}/ws"
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


async def _ensure_event_listener(thread_id: str) -> None:
    task = listener_tasks.get(thread_id)
    if task and not task.done():
        return
    task = asyncio.create_task(_event_listener(thread_id))
    listener_tasks[thread_id] = task


async def _post(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    async with httpx.AsyncClient(base_url=BACKEND_URL, timeout=HTTP_TIMEOUT) as client:
        response = await client.post(path, json=payload)
        response.raise_for_status()
        return response.json()


async def _get(path: str) -> Dict[str, Any]:
    async with httpx.AsyncClient(base_url=BACKEND_URL, timeout=HTTP_TIMEOUT) as client:
        response = await client.get(path)
        response.raise_for_status()
        return response.json()


async def start_workflow(prompt: str, thread_id: str) -> Dict[str, Any]:
    return await _post('/workflow/start', {'prompt': prompt, 'thread_id': thread_id})


async def continue_workflow(thread_id: str, user_feedback: Optional[str] = None) -> Dict[str, Any]:
    return await _post(f'/workflow/{thread_id}/continue', {'user_feedback': user_feedback})


async def stop_workflow(thread_id: str) -> Dict[str, Any]:

    return await _post(f'/workflow/{thread_id}/stop', {})


async def get_workflow_state(thread_id: str) -> Dict[str, Any]:
    try:
        return await _get(f'/workflow/{thread_id}/state')
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            return {'thread_id': thread_id, 'state': {}}
        raise


@cl.password_auth_callback
def auth_callback(login: str):
    return cl.User(
        identifier=login,
        metadata={
            'role': 'user',
            'provider': 'open_registration',
            'system': 'chainlit_ui'
        }
    )


@cl.action_callback('continue_generation')
async def continue_generation(action: cl.Action) -> None:
    thread_id = getattr(cl.user_session, 'thread_id', None)
    interrupt_message = getattr(cl.user_session, 'interrupt_message', None)

    if interrupt_message:
        interrupt_message.content = "#### ⏸️ **Interrupt: Waiting for Decision**\nContinuing generation..."
        await interrupt_message.update()
        await interrupt_message.remove_actions()
        cl.user_session.interrupt_message = None

    if thread_id:
        cl.user_session.awaiting_feedback = False
        await continue_workflow(thread_id, "")


@cl.action_callback('stop_generation')
async def stop_generation(action: cl.Action) -> None:
    logger.info(f"stop_generation: {cl.user_session.get('id')}")
    thread_id = getattr(cl.user_session, 'thread_id', None)
    interrupt_message = getattr(cl.user_session, 'interrupt_message', None)

    if interrupt_message:
        interrupt_message.content = "#### ⏸️ **Interrupt: Waiting for Decision**\nGeneration stopped."
        await interrupt_message.update()
        await interrupt_message.remove_actions()
        cl.user_session.interrupt_message = None

    if thread_id:
        cl.user_session.thread_id = None
        cl.user_session.awaiting_feedback = False
        await stop_workflow(thread_id)
        await _stop_event_listener(thread_id)


@cl.on_chat_start
async def start() -> None:
    logger.info(f"start_chat: {cl.user_session.get('id')}")
    if not hasattr(cl.user_session, 'current_steps'):
        cl.user_session.current_steps = {}
    if not hasattr(cl.user_session, 'iteration_messages'):
        cl.user_session.iteration_messages = {}

    cl.user_session.thread_id = None
    cl.user_session.awaiting_feedback = False
    cl.user_session.interrupt_message = None

    await cl.Message(content=INITIALIZING_MESSAGE).send()
    await cl.Message(content=SYSTEM_READY_MESSAGE).send()


@cl.on_message
async def main(message: cl.Message) -> None:
    logger.info(f"on_message: {cl.user_session.get('id')}")
    user_request = message.content.strip()
    if not user_request:
        await cl.Message(content='Please describe the desired image.').send()
        return

    thread_id_web = cl.user_session.thread_id
    awaiting_feedback = cl.user_session.awaiting_feedback
    interrupt_message = cl.user_session.interrupt_message

    try:
        if thread_id_web:

            if not awaiting_feedback:
                await cl.Message(
                    content='⏳ Current generation is still running. Wait for results or a continuation signal.'
                ).send()
                return
            
            if interrupt_message:
                await interrupt_message.remove_actions()

            await cl.Message(
                content=f"#### 🔄 **Continuing Generation** \nUser clarification: *{user_request}*\n"
            ).send()
            cl.user_session.awaiting_feedback = False
            cl.user_session.interrupt_message = None
            await _ensure_event_listener(thread_id_web)
            await continue_workflow(thread_id_web, user_request)

        else:
            await cl.Message(
                content=f"### 🎯 **Starting Generation Workflow** \nUser prompt: *{user_request}*\n"
            ).send()
            thread_id_web = str(uuid.uuid4())
            await start_workflow(user_request, thread_id=thread_id_web)
            logger.debug(f"on_message: new workflow started, received thread_id={thread_id_web}")

            cl.user_session.thread_id = thread_id_web
            cl.user_session.awaiting_feedback = False
            cl.user_session.interrupt_message = None
            await _ensure_event_listener(thread_id_web)
            
    except Exception as exc:
        logger.error(f"Workflow request failed: {exc}")
        await cl.Message(
            content=f"❌ **Error**: {exc}\n\nCheck backend configuration and try again."
        ).send()

if __name__ == '__main__':
    pass
