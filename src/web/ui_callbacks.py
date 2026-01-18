import uuid
import chainlit as cl
from loguru import logger

from src.prompts.web_messages import (
    INITIALIZING_MESSAGE,
    SYSTEM_READY_MESSAGE,
)
from src.web.backend_client import BackendClient
from src.web.event_manager import EventManager
from src.web.status_handler import StatusHandler


class UICallbacks:
    def __init__(self, backend_client: BackendClient, event_manager: EventManager, status_handler: StatusHandler):
        self.backend_client = backend_client
        self.event_manager = event_manager
        self.status_handler = status_handler

    @staticmethod
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
    async def continue_generation(self, action: cl.Action) -> None:
        thread_id = getattr(cl.user_session, 'thread_id', None)
        interrupt_message = getattr(cl.user_session, 'interrupt_message', None)

        if interrupt_message:
            interrupt_message.content = "#### **Interrupt: Waiting for Decision**\nContinuing generation..."
            await interrupt_message.update()
            await interrupt_message.remove_actions()
            cl.user_session.interrupt_message = None

        if thread_id:
            cl.user_session.awaiting_feedback = False
            await self.backend_client.continue_workflow(thread_id, "")

    @cl.action_callback('stop_generation')
    async def stop_generation(self, action: cl.Action) -> None:
        logger.info(f"stop_generation: {cl.user_session.get('id')}")
        thread_id = getattr(cl.user_session, 'thread_id', None)
        interrupt_message = getattr(cl.user_session, 'interrupt_message', None)

        if interrupt_message:
            interrupt_message.content = "#### **Interrupt: Waiting for Decision**\nGeneration stopped."
            await interrupt_message.update()
            await interrupt_message.remove_actions()
            cl.user_session.interrupt_message = None

        if thread_id:
            cl.user_session.thread_id = None
            cl.user_session.awaiting_feedback = False
            await self.backend_client.stop_workflow(thread_id)
            await self.event_manager._stop_event_listener(thread_id)

    @cl.on_chat_start
    async def start(self) -> None:
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
    async def main(self, message: cl.Message) -> None:
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
                        content='Current generation is still running. Wait for results or a continuation signal.'
                    ).send()
                    return
                
                if interrupt_message:
                    await interrupt_message.remove_actions()

                await cl.Message(
                    content=f"#### **Continuing Generation** \nUser clarification: *{user_request}*\n"
                ).send()
                cl.user_session.awaiting_feedback = False
                cl.user_session.interrupt_message = None
                await self.event_manager._ensure_event_listener(thread_id_web, self.status_handler.status_callback)
                await self.backend_client.continue_workflow(thread_id_web, user_request)

            else:
                await cl.Message(
                    content=f"### **Starting Generation Workflow** \nUser prompt: *{user_request}*\n"
                ).send()
                thread_id_web = str(uuid.uuid4())
                await self.backend_client.start_workflow(user_request, thread_id=thread_id_web)
                logger.debug(f"on_message: new workflow started, received thread_id={thread_id_web}")

                cl.user_session.thread_id = thread_id_web
                cl.user_session.awaiting_feedback = False
                cl.user_session.interrupt_message = None
                await self.event_manager._ensure_event_listener(thread_id_web, self.status_handler.status_callback)
                
        except Exception as exc:
            logger.error(f"Workflow request failed: {exc}")
            await cl.Message(
                content=f"**Error**: {exc}\n\nCheck backend configuration and try again."
            ).send()