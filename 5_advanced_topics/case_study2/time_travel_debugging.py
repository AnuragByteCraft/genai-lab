from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from agentic_ide_backend_enhanced import create_graph, checkpointer  # app, checkpointer


app = create_graph()


def list_checkpoints(thread_id: str):
    """List checkpoints and return (checkpoint_id, message_count) pairs."""
    config = {"configurable": {"thread_id": thread_id}}

    checkpoint_count = 0
    for checkpoint_tuple in checkpointer.list(config):
        # Each item is a CheckpointTuple
        checkpoint_id = checkpoint_tuple.config["configurable"]["checkpoint_id"]

        # Count messages in this checkpoint's state (adjust key as needed)
        messages = checkpoint_tuple.checkpoint.get("channel_values", {}).get("messages", [])
        msg_count = len(messages)

        checkpoint_count += 1
        yield checkpoint_id, msg_count

    if checkpoint_count == 0:
        print(f"No checkpoints found for thread_id: {thread_id}")


def replay_to_step(thread_id: str, step_index: int):
    """Replay to a specific step/checkpoint by index in thread history."""
    # step_index = -step_index  # -1 is the latest, so we invert

    config = {"configurable": {"thread_id": thread_id}}

    # Get ALL checkpoints in reverse chronological order (newest first)
    history = list(checkpointer.list(config))

    if step_index >= len(history):
        raise ValueError(f"Step {step_index} out of range. Only {len(history)} checkpoints found.")

    # Get the target checkpoint (index 0 = newest, index -1 = oldest)
    target_checkpoint_tuple = history[step_index]

    # Return the checkpoint state
    return target_checkpoint_tuple.checkpoint


if __name__ == '__main__':
    print("\n\n🧪 DEMO 3: Time Travel Debugging\n")
    thread_id = "buggy_refactor3"

    # Simulate buggy agent run (e.g., overwrites file incorrectly)
    user_request = (
                    "You are required to simulate generating a buggy code."
                    "Simulate the error by modifying demo_app.py by removing the function body of greet(). "
                    "This bug is intentionally introduced and shouldn't be corrected by executor."
                    "Write the modified code back to the file demo_app.py")
    initial_state = {
        "messages": [HumanMessage(content=user_request)],
        "current_file": "demo_app.py",
        "plan": None,
        "iterations": 0
    }
    config = {"configurable": {"thread_id": thread_id}}

    print("Running the backend agents to simulate buggy code...")
    # for _ in app.stream(initial_state, config=config):
    #     pass  # Let it finish (with bug)

    for event in app.stream(initial_state, config=config):
        for key, value in event.items():
            print(f"\n🔹 Node '{key}' executed.")
            if "messages" in value:
                msg = value["messages"][-1]
                if isinstance(msg, AIMessage):
                    if msg.content:
                        print(f"   💬 {msg.content[:200]}...")
                    if msg.tool_calls:
                        call = msg.tool_calls[0]
                        print(f"   🛠️  Tool: {call['name']}({call['args']})")

    x = input("Check demo_app.py to locate bugs and enter: ")

    # Show history
    print("📜 Checkpoint history:")
    for idx, msg_count in list_checkpoints(thread_id):
        print(f"  Step {idx}: {msg_count} messages")

    # User chooses to revert to step 2 (before bad write)
    print("\n⏪ Reverting to step before bug...")
    good_state = replay_to_step(thread_id, step_index=2)

    # Now continue with corrected plan
    corrected_request = "Now do it correctly: keep function body and add docstring."
    corrected_request = """
    Correct the buggy code in demo_app.py. Wwrite that back as a file using write_file tools.
    """
    # good_state["channel_values"]["messages"].append(HumanMessage(content=corrected_request))
    good_state["channel_values"]["messages"] = [(HumanMessage(content=corrected_request))]

    print("Good state = ", good_state)

    new_initial_state = {
        "messages": [HumanMessage(content=corrected_request)],
        "current_file": "demo_app.py",
        "plan": None,
        "iterations": 0
    }
    config = {"configurable": {"thread_id": thread_id}}

    for event in app.stream(new_initial_state, config=config):
        for key, value in event.items():
            print(f"\n🔹 Node '{key}' executed.")
            if "messages" in value:
                msg = value["messages"][-1]
                if isinstance(msg, AIMessage):
                    if msg.content:
                        print(f"   💬 {msg.content[:200]}...")
                    if msg.tool_calls:
                        call = msg.tool_calls[0]
                        print(f"   🛠️  Tool: {call['name']}({call['args']})")


    # for event in app.stream(good_state, config=config):
    #     pass

    print("✅ Fixed version written!")