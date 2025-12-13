from langgraph.types import Command, interrupt

from langgraph.graph import StateGraph, START
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict
import uuid


class State(TypedDict):
    some_text: str


def human_node(state: State):
    # Pause and ask human to revise text
    revised_text = interrupt({"text_to_revise": state["some_text"]})
    return {"some_text": revised_text}


builder = StateGraph(State)
builder.add_node("human_node", human_node)
builder.add_edge(START, "human_node")

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": uuid.uuid4()}}

# Run until interrupt
result = graph.invoke({"some_text": "original text"}, config=config)
# print(result["__interrupt__"])  # Interrupt payload shown
print(result)

# Resume with human input
resumed = graph.invoke(Command(resume="Edited text"), config=config)
print(resumed)  # {'some_text': 'Edited text'}
