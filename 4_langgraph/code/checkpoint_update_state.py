from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage


# Define state
class ChatState(TypedDict):
    messages: list
    user_mood: str  # State we might want to override


# Node: Process based on mood
def process_mood(state: ChatState):
    mood = state["user_mood"]
    response = f"You're feeling {mood}. Here's my response..."

    return {"messages": state["messages"] + [AIMessage(content=response)]}


# Build graph
builder = StateGraph(ChatState)
builder.add_node("process", process_mood)
builder.add_edge(START, "process")
builder.add_edge("process", END)

# Compile with checkpointer
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "user_session_1"}}

# --- Step 1: Initial run ---
print("=== Initial Run ===")
result1 = graph.invoke(
    {
        "messages": [HumanMessage(content="I'm not feeling great")],
        "user_mood": "sad"  # User entered this
    },
    config
)
print(f"Response: {result1['messages'][-1].content}")

# --- Step 2: Check current state ---
print("\n=== Check Current Checkpoint ===")
current_state = graph.get_state(config)
print(f"Current state values: {current_state.values}")
print(f"Checkpoint ID: {current_state.config['configurable']['checkpoint_id']}")

# --- Step 3: MODIFY the checkpoint ---
print("\n=== Update Checkpoint (Human corrects the mood) ===")
# Human review: "Actually, the user said they're excited, not sad!"
new_state_config = graph.update_state(
    config,
    {"user_mood": "excited"},  # Override the mood value
    as_node="process"  # Update is treated as coming from this node
)
print(f"New checkpoint created with ID: {new_state_config['configurable']['checkpoint_id']}")

# --- Step 4: Get the modified state ---
print("\n=== View Modified State ===")
modified_state = graph.get_state(new_state_config)
print(f"Updated mood: {modified_state.values['user_mood']}")

# --- Step 5: Resume execution from modified checkpoint ---
print("\n=== Resume from Modified Checkpoint ===")
result2 = graph.invoke(
    {
        "messages": [HumanMessage(content="New message after correction")],
        "user_mood": "excited"  # This continues from corrected state
    },
    new_state_config
)
print(f"Response after modification: {result2['messages'][-1].content}")

# --- Step 6: View full history (shows branching) ---
print("\n=== Full Thread History (Shows Fork) ===")
history = list(graph.get_state_history(config))
print(f"Total checkpoints in thread: {len(history)}")
for i, snapshot in enumerate(history):
    print(f"  Checkpoint {i}: mood={snapshot.values.get('user_mood')}, "
          f"checkpoint_id={snapshot.config['configurable']['checkpoint_id'][:8]}...")
