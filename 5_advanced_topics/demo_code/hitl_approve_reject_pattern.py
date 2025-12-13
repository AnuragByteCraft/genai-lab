import uuid
from typing import Literal, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver

# Define the graph state
class State(TypedDict):
    llm_output: str
    decision: str

# Node that simulates generating output (e.g., from an LLM)
def generate_llm_output(state: State):
    return {"llm_output": "This is the generated output."}

# Human approval node with interrupt
def human_approval(state: State) -> Command[Literal["approved_path", "rejected_path"]]:
    decision = interrupt({
        "question": "Do you approve the following output?",
        "llm_output": state["llm_output"]
    })

    if decision.lower() in ("approve", "yes", "y"):
        return Command(goto="approved_path", update={"decision": "approved"})
    else:
        return Command(goto="rejected_path", update={"decision": "rejected"})

# Node executed if approved
def approved_node(state: State) -> State:
    print("✅ Approved path taken.")
    return state

# Node executed if rejected
def rejected_node(state: State) -> State:
    print("❌ Rejected path taken.")
    return state

# Build the graph
builder = StateGraph(State)
builder.add_node("generate_llm_output", generate_llm_output)
builder.add_node("human_approval", human_approval)
builder.add_node("approved_path", approved_node)
builder.add_node("rejected_path", rejected_node)

builder.set_entry_point("generate_llm_output")
builder.add_edge("generate_llm_output", "human_approval")
builder.add_edge("approved_path", END)
builder.add_edge("rejected_path", END)

# Use an in-memory checkpointer (for persistence during runtime)
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

graph.get_graph().draw_mermaid_png(output_file_path="approve_reject_graph.png")

# Create a unique thread ID for this conversation
thread_id = uuid.uuid4()
config = {"configurable": {"thread_id": thread_id}}

# Run the graph until the interrupt (human approval) is hit
result = graph.invoke({}, config=config)
print("Interrupt payload:", result["__interrupt__"])

# Simulate human input (approve or reject)
# Replace "approve" with "reject" to test rejection path
human_input = "approve"
# human_input = "reject"

# Resume the graph with human input
final_result = graph.invoke(Command(resume=human_input), config=config)
print("Final result:", final_result)
