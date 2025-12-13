import os
from dotenv import load_dotenv

load_dotenv()

from src.agents.generator import GeneratorAgent
from src.agents.critic import CriticAgent
from src.agents.planner import PlannerAgent
from src.models.image_generation_client import ImageGenerationClient
from src.models.vlm_client import VLMClient
from src.models.llm_client import LLMClient
from src.core.config import settings
from src.core.state import WorkflowState, create_initial_state

from loguru import logger
from langgraph.graph import StateGraph, END
from langgraph.types import Command, interrupt
from typing import Dict, Any, Callable, Optional, Awaitable
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver, AsyncConnectionPool

sent_messages = {
    'feedback': {},
    'error': {},
    'end_planner': {},
    'end_critic': {}
}

class ImageGenerationWorkflow:
    """
    Workflow для итеративной генерации изображений.
    
    Граф выполнения:
    START -> Planner -> Generator -> Critic -> (Planner | END)
    """
    
    def __init__(
        self,
        imagen_client: ImageGenerationClient,
        vlm_client: VLMClient,
        llm_client: LLMClient
    ):
        self.generator = GeneratorAgent(imagen_client, llm_client)
        self.critic = CriticAgent(vlm_client)
        self.planner = PlannerAgent(llm_client)
        
        self.status_callback: Optional[Callable] = None        
        self.confirmation_callback: Optional[Callable[[WorkflowState], Awaitable[bool]]] = None        
        self.graph = None

        self.checkpointer = None

    async def initialize(self):
        connection_kwargs = {
            "autocommit": True,
            "prepare_threshold": 0,
        }

        self.db_pool = AsyncConnectionPool(conninfo=ImageGenerationWorkflow._get_database_url(), kwargs=connection_kwargs)
        await self.db_pool.open() 
        
        self.checkpointer = AsyncPostgresSaver(self.db_pool)
        await self.checkpointer.setup()

        self.graph = self._build_graph()
        return self


    @staticmethod
    def _get_database_url() -> str:
        from urllib.parse import quote
        
        user = os.getenv("POSTGRES_USER", "postgres")
        password = os.getenv("POSTGRES_PASSWORD", "password")
        db = os.getenv("POSTGRES_DB", "langgraph")
        host = os.getenv("POSTGRES_HOST", "localhost")
        port = os.getenv("POSTGRES_PORT", "5432")
        
        return f"postgresql://{quote(user)}:{quote(password)}@{quote(host)}:{port}/{quote(db)}"
    
    def set_status_callback(self, callback: Callable):
        """Устанавливает callback для отправки статусов из агентов."""
        self.status_callback = callback
        self.generator.set_status_callback(callback)
        self.critic.set_status_callback(callback)
        self.planner.set_status_callback(callback)
    
    def _build_graph(self) -> StateGraph:
        self.workflow = StateGraph(WorkflowState)
        
        self.workflow.add_node("generator", self._generator_node)
        self.workflow.add_node("critic", self._critic_node)
        self.workflow.add_node("planner", self._planner_node)
        self.workflow.add_node("error_interrupt", self._error_interrupt_node)
        self.workflow.add_node("feedback_interrupt", self._feedback_interrupt_node)
        self.workflow.add_node("end_planner_interrupt", self._end_planner_interrupt_node)
        self.workflow.add_node("end_critic_interrupt", self._end_critic_interrupt_node)
        
        # START -> Planner
        self.workflow.set_entry_point("planner")

        # Planner -> Generator
        self.workflow.add_conditional_edges(
            "planner",
            self._should_continue_planner,
            {
                "error": "error_interrupt",
                "end": "end_planner_interrupt",
                "continue": "generator"
            }
        )

        # Generator -> Critic
        self.workflow.add_conditional_edges(
            "generator",
            self._has_error,
            {
                "error": "error_interrupt",
                "continue": "critic"
            }
        )
        
        # Critic -> (Interrupt | END)
        self.workflow.add_conditional_edges(
            "critic",
            self._should_continue_critic,
            {
                "error": "error_interrupt",
                "end": "end_critic_interrupt",
                "continue": "feedback_interrupt"
            }
        )
        self.workflow.add_edge("error_interrupt", "planner")
        self.workflow.add_edge("feedback_interrupt", "planner")
        self.workflow.add_edge("end_planner_interrupt", "planner")
        self.workflow.add_edge("end_critic_interrupt", "planner")
        
        return self.workflow.compile(checkpointer=self.checkpointer)
    
    def visualize_graph(self, output_path: str = "workflow_graph.png"):
        if self.graph is None:
            raise ValueError("Граф не инициализирован. Вызовите initialize() сначала.")
        
        png_bytes = self.graph.get_graph().draw_mermaid_png()
        with open(output_path, "wb") as f:
            f.write(png_bytes)
        logger.info(f"Граф сохранен в {output_path}")
    
    def print_graph(self):
        if self.graph is None:
            raise ValueError("Граф не инициализирован. Вызовите initialize() сначала.")
        
        ascii_graph = self.graph.get_graph().draw_ascii()
        print(ascii_graph)



    async def _generator_node(self, state: WorkflowState) -> Dict[str, Any]:
        result = await self.generator.execute(state)
        return result
    
    async def _critic_node(self, state: WorkflowState) -> Dict[str, Any]:
        result = await self.critic.execute(state)
        return result
    
    async def _planner_node(self, state: WorkflowState) -> Dict[str, Any]:
        result = await self.planner.execute(state)
        return result
    


    async def _feedback_interrupt_node(self, state: WorkflowState):
        """
        Обрабатывает прерывание для получения обратной связи от пользователя.
        Вызывается после определенного числа итераций без обратной связи.
        """
        if state["iterations_without_feedback"]<=0:
            thread_id = state["thread_id"]
            if not sent_messages['feedback'].get(thread_id, False):
                sent_messages['feedback'][thread_id] = True

                if self.status_callback:  # STATUS_CALLBACK: feedback_interrupt
                    await self.status_callback("feedback_interrupt", {
                        "thread_id": thread_id,
                        "current_iteration": state['current_iteration'],
                    })

            resume = interrupt("Ожидание пользовательского решения")
            sent_messages['feedback'][thread_id] = False
            
            if resume != "":
                state["user_feedback"] = resume
            
            state["iterations_without_feedback"]=settings.default_iterations_without_feedback

        return state

    async def _error_interrupt_node(self, state: WorkflowState):
        """
        Обрабатывает прерывание при возникновении ошибки в workflow.
        """
        thread_id = state["thread_id"]
        if not sent_messages['error'].get(thread_id, False):
            sent_messages['error'][thread_id] = True

            if self.status_callback:  # STATUS_CALLBACK: error_interrupt
                await self.status_callback("error_interrupt", {
                    "thread_id": thread_id,
                    "current_iteration": state['current_iteration'],
                })

        user_feedback = interrupt("Ожидание пользовательского решения")
        sent_messages['error'][thread_id] = False

        return state
    
    async def _end_planner_interrupt_node(self, state: WorkflowState):
        """
        Обрабатывает прерывание, когда планировщик решает не продолжать генерацию.
        Позволяет пользователю принять решение о дальнейших действиях.
        """
        thread_id = state["thread_id"]
        if not sent_messages['end_planner'].get(thread_id, False):
            sent_messages['end_planner'][thread_id] = True

            if self.status_callback:  # STATUS_CALLBACK: end_planner_interrupt
                await self.status_callback("end_planner_interrupt", {
                    "thread_id": thread_id,
                    "current_iteration": state['current_iteration'],
                    "current_quality_score": state['current_quality_score']
                })

        user_feedback = interrupt("Ожидание пользовательского решения")
        
        state["current_user_feedback"] = user_feedback
        sent_messages['end_planner'][thread_id] = False

        return state
    
    async def _end_critic_interrupt_node(self, state: WorkflowState):
        """
        Обрабатывает прерывание, когда достигнут минимальный уровень качества генерации.
        Позволяет пользователю принять решение о завершении или продолжении.
        """
        thread_id = state["thread_id"]
        if not sent_messages['end_critic'].get(thread_id, False):
            sent_messages['end_critic'][thread_id] = True

            if self.status_callback:  # STATUS_CALLBACK: end_critic_interrupt
                await self.status_callback("end_critic_interrupt", {
                    "thread_id": thread_id,
                    "current_iteration": state['current_iteration'],
                    "current_quality_score": state['current_quality_score']
                })

        user_feedback = interrupt("Ожидание пользовательского решения")
        
        state["current_user_feedback"] = user_feedback
        sent_messages['end_critic'][thread_id] = False

        return state



    async def _has_error(self, state: WorkflowState) -> str:
        """
        Проверяет, возникла ли ошибка в текущем состоянии workflow.
        """
        if state.get('error_message'):
            return "error"
        return "continue"
    
    async def _should_continue_planner(self, state: WorkflowState) -> str:
        """
        Определяет, продолжать ли итерации или завершить workflow после агента Планировщика.
        """
        if state.get('error_message'):
            return "error"
        
        if not state.get('continue_generation', True):
            return "end"
    
        return "continue"
    
    
    async def _should_continue_critic(self, state: WorkflowState) -> str:
        """
        Определяет, продолжать ли итерации или завершить workflow после агента Критика.
        """
        if state.get('error_message'):
            return "error"
        
        if state.get('current_quality_score'):
            if state['current_quality_score'] >= settings.min_quality_score:
                logger.info(f"Final score: {state['current_quality_score']:.1f}/10. \n \
                            Quality threshold ({settings.min_quality_score}) reached")
                return "end"
        
        return "continue"
    


    
    async def get_current_state(self, thread_id: str) -> Any:
        """Получает текущее состояние генерации"""
        config = {"configurable": {"thread_id": thread_id}}
        return await self.graph.aget_state(config)

    async def start_generation(self, initial_prompt: str, thread_id: str) -> Dict[str, Any]:
        """Начинает новую генерацию с указанным thread_id"""
        config = {"configurable": {"thread_id": thread_id}}

        initial_state = create_initial_state(initial_prompt, thread_id)
        
        result = await self.graph.ainvoke(initial_state, config)
        return result
    
    async def continue_generation(self, thread_id: str, user_feedback: str = None) -> Dict[str, Any]:
        """Продолжает существующую генерацию"""
        config = {"configurable": {"thread_id": thread_id}}

        result = await self.graph.ainvoke(Command(resume=user_feedback or ""), config)
        return result

    async def stop_generation(self, thread_id: str) -> Dict[str, Any]:
        """Останавливает существующую генерацию"""
        config = {"configurable": {"thread_id": thread_id}}
        
        result = await self.graph.ainvoke(Command(goto=END), config)
        return result
    


async def create_workflow() -> ImageGenerationWorkflow:
    imagen_client = ImageGenerationClient()
    vlm_client = VLMClient()
    llm_client = LLMClient()
    
    workflow = ImageGenerationWorkflow(
        imagen_client=imagen_client,
        vlm_client=vlm_client,
        llm_client=llm_client
    )

    await workflow.initialize()
    
    return workflow
