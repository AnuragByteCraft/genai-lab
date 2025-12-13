import uuid
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver

# Define the graph state
class State(TypedDict):
    summary: str

# Node that generates a summary (simulated)
def generate_summary(state: State) -> State:
    return {
        "summary": "The cat sat on the mat and looked at the stars."
    }

# Node that lets human review and edit the summary
def human_review_edit(state: State) -> State:
    # Pause execution and ask human to review/edit
    result = interrupt({
        "task": "Please review and edit the generated summary if necessary.",
        "generated_summary": state["summary"]
    })
    # Update state with human-edited summary
    return {
        "summary": result["edited_summary"]
    }

# Node that uses the edited summary downstream
def downstream_use(state: State) -> State:
    print(f"✅ Using edited summary: {state['summary']}")
    return state

# Build the graph
builder = StateGraph(State)
builder.add_node("generate_summary", generate_summary)
builder.add_node("human_review_edit", human_review_edit)
builder.add_node("downstream_use", downstream_use)

builder.set_entry_point("generate_summary")
builder.add_edge("generate_summary", "human_review_edit")
builder.add_edge("human_review_edit", "downstream_use")
builder.add_edge("downstream_use", END)

# Use in-memory checkpointer for demo
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# Create a thread ID for this run
thread_id = uuid.uuid4()
config = {"configurable": {"thread_id": thread_id}}

# Run the graph until it hits the interrupt (human review)
result = graph.invoke({}, config=config)
print("Interrupt payload:", result["__interrupt__"])

# Simulate human editing the summary
human_edited_summary = "The cat lay on the rug, gazing peacefully at the night sky."

# Resume the graph with the edited summary
final_result = graph.invoke(
    Command(resume={"edited_summary": human_edited_summary}),
    config=config
)
print("Final graph state:", final_result)
