import uuid
from typing import Union

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool as create_tool
from langchain_core.runnables import RunnableConfig


# Define the graph state
class State(dict):
    pass

# Example tool that we want to review before execution
@create_tool
def example_tool(query:str) -> str:
    """
    This example tool is to illustrate HITL tool review capability
    :param config:
    :param query:
    :return:
    """
    # Pause execution to review the tool call
    review = interrupt({
        "question": "Please review the tool call",
        "tool_call": {
            "name": "example_tool",
            "args": {"query": query}
        }
    })

    # Handle human response
    print(review)
    review = review[0]
    if review["type"] == "accept":
        # Execute the tool normally
        return f"Tool executed with query: {query}"
    elif review["type"] == "edit":
        # Use edited arguments
        new_args = review["args"]["args"]
        return f"Tool executed with edited query: {new_args['query']}"
    elif review["type"] == "response":
        # Use human feedback as tool response
        return review["args"]
    else:
        raise ValueError(f"Unsupported review type: {review['type']}")

# Node that calls the tool
def call_tool_node(state: State) -> State:
    # Call the example tool with some query
    result = example_tool.invoke({"query": "Initial query"})
    return {"tool_result": result}

# Node to finish the graph
def end_node(state: State) -> State:
    print("Tool call result:", state.get("tool_result"))
    return state

# Build the graph
builder = StateGraph(State)
builder.add_node("call_tool_node", call_tool_node)
builder.add_node("end_node", end_node)
builder.add_edge(START, "call_tool_node")
builder.add_edge("call_tool_node", "end_node")
builder.add_edge("end_node", END)

# Setup checkpointer
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# Create a thread ID
thread_id = uuid.uuid4()
config = {"configurable": {"thread_id": thread_id}}

# Run the graph until interrupt (tool call review)
result = graph.invoke({}, config=config)
print("Interrupt payload:", result["__interrupt__"])

# Simulate human review input:
# Options:
# 1. Accept: [{"type": "accept"}]
# 2. Edit: [{"type": "edit", "args": {"args": {"query": "Edited query"}}}]
# 3. Respond with feedback: [{"type": "response", "args": "Human feedback response"}]

inp = input("Enter accept/edit/..")

human_review_response = [{"type": inp}]  # Change as needed

# Resume graph with human review response
final_result = graph.invoke(Command(resume=human_review_response), config=config)
print("Final graph state:", final_result)
