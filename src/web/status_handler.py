import chainlit as cl
from chainlit import Image
from typing import Any, Dict, Optional

from src.prompts.web_messages import AGENT_ERROR_MESSAGE_TEMPLATE
from src.web.event_manager import EventManager


class StatusHandler:
    def __init__(self, event_manager: EventManager):
        self.event_manager = event_manager

    async def status_callback(self, status_type: str, data: Dict[str, Any]) -> None:
        thread_id = (
            data.get('thread_id')
            or data.get('configurable', {}).get('thread_id')
            or getattr(cl.user_session, 'thread_id', None)
        )

        iteration = (data.get('current_iteration') or 0)

        from loguru import logger
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
            step.output = 'The agent checks if quality improves each iteration and decides whether to continue....'
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
            step.output = 'Analyzing feedback and planning improvements...'
            await step.update()

        elif status_type == 'prompt_ready':
            key = f"prompt_generation_{iteration}"
            if key in cl.user_session.current_steps:
                step = cl.user_session.current_steps[key]
                step.output = f"New prompt planned:\n\n*{data.get('prompt', '-')}*"
                await step.__aexit__(None, None, None)

        elif status_type == 'image_generation':
            step = cl.Step(name='Generating image...', type='tool')
            cl.user_session.current_steps[f"image_generation_{iteration}"] = step
            await step.__aenter__()
            step.output = 'Image generation in progress...'
            await step.update()

        elif status_type == 'comfy_progress':
            print ("data in comfy_progress:", data)
            key = f"image_generation_{iteration}"

            iteration_text = "executing"
            if key in cl.user_session.current_steps:
                step = cl.user_session.current_steps[key]
                payload = data.get('payload', {}) or {}
                data_payload = payload.get('data') or {}

                if payload.get("type") == "progress":
                    iteration_temp = data_payload.get("value")
                    iteration_max = data_payload.get("max")
                    if iteration_temp is not None and iteration_max is not None:
                        iteration_text = f"generation iteration {iteration_temp}/{iteration_max}"
                elif payload.get("type") == "executing":
                    node_name = data.get("node_name") or "unknown_node"
                    iteration_text = f"node {node_name} executing"

                step.output =f"Comfy status: {iteration_text}"
                await step.update()

        elif status_type == 'image_ready':
            key = f"image_generation_{iteration}"
            if key in cl.user_session.current_steps:
                step = cl.user_session.current_steps[key]
                step.output = 'Image generated successfully!'
                await step.__aexit__(None, None, None)

            with open(data.get('image_path', ''), "rb") as f:
                image_bytes = f.read()
            image = Image(content=image_bytes, 
                          display='inline',
                           name=f"iteration_{iteration}",
                          mime="image/png")

            await cl.Message(content='Image generated:', elements=[image]).send()

        # Critic
        elif status_type == 'critic_requirements_start':
            cl.user_session.iteration_messages[f"critic_{iteration}"] = await cl.Message(
                content=f"\n---\n#### 🔍 Iteration {iteration}: Critic\n"
            ).send()

            step = cl.Step(name='Generating requirements for critic...', type='tool')
            cl.user_session.current_steps[f"critic_requirements_{iteration}"] = step
            await step.__aenter__()
            step.output = 'Requirements generation in progress...'
            await step.update()


        elif status_type == 'critic_requirements_ready':
            key = f"critic_requirements_{iteration}"
            if key in cl.user_session.current_steps:
                addition = f"**Checklist Prepared:**\n{data.get('requirements', '')}\n\n"

                step = cl.user_session.current_steps[key]
                step.output = addition
                await step.__aexit__(None, None, None)

            step = cl.Step(name='Analyzing image...', type='tool')
            cl.user_session.current_steps[f"critic_visual_analysis_{iteration}"] = step
            await step.__aenter__()
            step.output = 'Analysis in progress...'
            await step.update()

        elif status_type == 'critic_visual_analysis_ready':
            key = f"critic_visual_analysis_{iteration}"
            if key in cl.user_session.current_steps:

                summary = data.get('analysis') or 'Visual scan ready.'
                addition = f"**Visual Scan Notes:**\n{summary}\n\n"

                step = cl.user_session.current_steps[key]
                step.output = addition
                await step.__aexit__(None, None, None)

            step = cl.Step(name='Image generation evaluation...', type='tool')
            cl.user_session.current_steps[f"critic_analysis_{iteration}"] = step
            await step.__aenter__()
            step.output = 'Analysis in progress...'
            await step.update()


        elif status_type == 'critic_analysis_ready':
            key = f"critic_analysis_{iteration}"
            if key in cl.user_session.current_steps:

                step = cl.user_session.current_steps[key]
                step.output = f"**Analysis and critique**:\n{data.get('critique', '')}\n\n"
                await step.__aexit__(None, None, None)

            await cl.Message(
                content=f"**Quality Score**: {data.get('quality_score', 0):.1f}/10\n\n",
            ).send()


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
                content="#### **Interrupt: Waiting for Decision**\nChoose 'Continue' or 'Stop', or provide more details.",
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
                step.output = f"**{agent} Failed**\n\n{error_message}"
                step.is_error = True
                await step.__aexit__(None, None, None)

            error_details = AGENT_ERROR_MESSAGE_TEMPLATE.format(agent, iteration, error_message)
            await cl.Message(content=error_details).send()
            cl.user_session.awaiting_feedback = False
            setattr(cl.user_session, 'interrupt_message', None)
            await self.event_manager._stop_event_listener(thread_id)

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
                content=f"\n Agent Planner decided to terminate the graph execution. Current quality score: **{data.get('current_quality_score', 0):.1f}/10**. \
                \nYou may provide additional details and then continue the generation process."
            ).send()
            cl.user_session.awaiting_feedback = True


        # Режим ожидания при завершении работы графа после Критика
        elif status_type == 'end_critic_interrupt':
            await cl.Message(
                content=f"\n Workflow Complete! Agent Critic decided to terminate the graph execution. Current quality score: **{data.get('current_quality_score', 0):.1f}/10**. \
                \nYou may provide additional details and then continue the generation process."
            ).send()
            cl.user_session.awaiting_feedback = True