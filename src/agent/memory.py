from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


def get_agent_with_memory(agent_executor) -> RunnableWithMessageHistory:
    return RunnableWithMessageHistory(
        runnable=agent_executor,
        get_session_history=lambda _: InMemoryChatMessageHistory(),
        input_messages_key="input",        # where user input goes
        history_messages_key="chat_history"  # key passed into agent for context
    )
