from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from agentic_ide_backend_enhanced import create_graph

app = create_graph()


# In main demo block (replace or add after Demo 2)
print("\n\n🧪 DEMO 1: Checkpoint Resumption after Crash\n")


# Simulate a crash after first write
def run_with_simulated_crash():
    user_request = """
    Add type hints and Google-style docstrings to demo_app.py. 
    Save the updated demo_app.py
    """

    config = {"configurable": {"thread_id": "refactor_demo"}}
    initial_state = {
        "messages": [HumanMessage(content=user_request)],
        "current_file": "demo_app.py",
        "error_context": None,
        "plan": None,
        "iterations": 0
    }

    # Run until first tool call (e.g., read_file or write_file)
    for event in app.stream(initial_state, config=config):
        for key, value in event.items():
            if "messages" in value:
                msg = value["messages"][-1]
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    print(f"🛠️  First action: {msg.tool_calls[0]['name']}")
                    print("💥 Simulating crash!")
                    return  # Simulate crash - if there is no return statement, the agent would have completed normal


if __name__ == '__main__':
    # First run (crashes)
    run_with_simulated_crash()

    # Second run: resume from checkpoint
    print("\n🔁 Restarting from checkpoint...\n")
    config = {"configurable": {"thread_id": "refactor_demo"}}
    # Reuse same thread_id → LangGraph resumes from last state
    for event in app.stream(None, config=config):  # state=None → load from checkpoint
        for key, value in event.items():
            if "messages" in value:
                msg = value["messages"][-1]
                if isinstance(msg, AIMessage):
                    if msg.content:
                        print(f"💬 {msg.content[:150]}...")
                    if msg.tool_calls:
                        call = msg.tool_calls[0]
                        print(f"🛠️  Tool: {call['name']}({call['args']})")

    print("✅ Refactor completed after restart!")
