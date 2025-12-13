from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict


class State(TypedDict):
    input: str


def step_1(state): print("Step 1")
def step_2(state): print("Step 2")
def step_3(state): print("Step 3")

builder = StateGraph(State)
builder.add_node("step_1", step_1)
builder.add_node("step_2", step_2)
builder.add_node("step_3", step_3)
builder.add_edge(START, "step_1")
builder.add_edge("step_1", "step_2")
builder.add_edge("step_2", "step_3")
builder.add_edge("step_3", END)

checkpointer = MemorySaver()

# Set static interrupt before step_3
graph = builder.compile(
    checkpointer=checkpointer,
    interrupt_before=["step_3"]
)

# Run until breakpoint before step_3
config = {"configurable": {"thread_id": "abc1"}}

result = graph.invoke({"input": "hello"}, config=config)
print(result)  # Pauses before step_3

print("BEF step 3")

# Resume execution (pass None to continue)
result = graph.invoke(None, config=config)
print(result)  # Continues with step_3 and ends
