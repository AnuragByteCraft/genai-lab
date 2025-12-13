from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from get_llm import get_llm
from langchain_core.messages import HumanMessage, AIMessage


llm = get_llm()


# Define the agent state - what persists across steps
class AgentState(TypedDict):
    messages: list  # Chat history is saved automatically


# Node 1: Process user input
def process_input(state: AgentState):
    """Echo the user's message and add AI response"""
    last_message = state["messages"][-1]
    print(f"User: {last_message.content}")

    # In a real agent, you'd call an LLM here
    response = f"Processing: {last_message.content}"

    return {"messages": state["messages"] + [AIMessage(content=response)]}


# Build the graph
builder = StateGraph(AgentState)
builder.add_node("process", process_input)
builder.add_edge(START, "process")
builder.add_edge("process", END)

# Compile with MemorySaver - this enables persistence
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# Use a thread_id to maintain state across invocations
config = {"configurable": {"thread_id": "conversation_1"}}

# First message
print("=== First Invocation ===")
graph.invoke(
    {"messages": [HumanMessage(content="Hello")]},
    config
)

# Second message - history is restored
print("\n=== Second Invocation ===")
graph.invoke(
    {"messages": [HumanMessage(content="How are you?")]},
    config
)

# View the checkpoint (saved state)
print("\n=== Checkpoint State ===")
state = graph.get_state(config)
print(f"Messages in memory: {len(state.values['messages'])}")
for msg in state.values["messages"]:
    print(f"  {msg.__class__.__name__}: {msg.content}")