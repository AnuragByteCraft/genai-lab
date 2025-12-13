from typing import List, Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.checkpoint.memory import MemorySaver  # Use MemorySaver for in-memory
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

# -----------------------------
# 1. Define State
# -----------------------------
class MyState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

# -----------------------------
# 2. Define a node
# -----------------------------
def process_message(state: MyState):
    """Node that responds to user messages."""
    last_user_msg = None
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            last_user_msg = m.content
            break
    if last_user_msg is None:
        last_user_msg = "No user message found"

    return {
        "messages": [AIMessage(content=f"Processing: {last_user_msg}")]
    }

# -----------------------------
# 3. Build the graph
# -----------------------------
workflow = StateGraph(MyState)
workflow.add_node("processor", process_message)
workflow.add_edge(START, "processor")
workflow.add_edge("processor", END)

# FIXED: Use MemorySaver for reliable in-memory checkpointing
checkpointer = MemorySaver()
graph = workflow.compile(checkpointer=checkpointer)

# -----------------------------
# 4. Helper to print state
# -----------------------------
def print_state(title: str, state):
    print(f"\n=== {title} ===")
    messages = state["messages"]
    print(f"Total messages: {len(messages)}")
    for i, msg in enumerate(messages, 1):
        role = "User" if isinstance(msg, HumanMessage) else "AI"
        print(f"  {i}. {role}: {msg.content}")


def inspect_state(state, thread_id: str = "chat_1"):
    """Inspect current channel values."""
    config = {"configurable": {"thread_id": thread_id}}

    print("\n🔍 CURRENT STATE CHANNEL VALUES:")
    print("=" * 50)
    for channel, value in state.items():
        print(f"📦 {channel}:")
        if isinstance(value, list):
            print(f"   {len(value)} items")
            for i, msg in enumerate(value[-3:], 1):  # Last 3 messages
                role = "🧑 YOU" if isinstance(msg, HumanMessage) else "🤖 AI"
                print(f"     {i}. {role}: {msg.content[:60]}...")
        else:
            print(f"   {value}")

    print(f"\nNext nodes to execute: {list(state.next)}")
    print(f"Current config: {state.config}")

# -----------------------------
# 5. Run the conversation
# -----------------------------
config = {"configurable": {"thread_id": "demo"}}

# First message
print("=== First Invocation ===")
state1 = graph.invoke(
    {"messages": [HumanMessage(content="Hello")]},
    config
)
print_state("After 1st message", state1)

# Second message (loads from checkpoint)
print("\n=== Second Invocation ===")
state2 = graph.invoke(
    {"messages": [HumanMessage(content="How are you?")]},
    config
)
print_state("After 2nd message", state2)
# inspect_state(state2, thread_id="demo")

# Third message
print("\n" + "="*60)
print("🔁 RESTARTING FROM CHECKPOINT — Adding 3rd message")
print("="*60)
state3 = graph.invoke(
    {"messages": [HumanMessage(content="What's the weather today?")]},
    config
)
print_state("After 3rd message", state3)