import uuid
import os
# from agentic_ide_backend2 import app # Imports the compiled graph from the file above
from agentic_ide_backend_enhanced import app
from langchain_core.messages import HumanMessage
from checkpoint_inspector import print_checkpoint_info


# 1. Setup a specific thread (Session ID)
thread_id = "demo_session_v1"
config = {"configurable": {"thread_id": thread_id}}

def print_separator(title):
    print(f"\n{'='*20} {title} {'='*20}\n")

# ==========================================
# STEP 1: INITIAL REQUEST
# ==========================================
print_separator("STEP 1: User Input")
user_input = "Create a file 'math_utils.py' with a function to calculate fibonacci, and a test file for it."
# user_input = "Create a file 'graph_algo.py' with a function to implement Dijkstra's algorithm, and a test file for it."
# user_input = "Using LangChain give me a agentic application example using create_agent"

print(f"User: {user_input}")

# Start the graph
# It will run 'planner' then PAUSE before 'executor' because of interrupt_before
for event in app.stream(
    {"messages": [HumanMessage(content=user_input)]},
    config,
    stream_mode="values"
):
    if "messages" in event:
        last_msg = event["messages"][-1]
        print(f"Stream: {last_msg.type} - {last_msg.content[:50]}...")

# ==========================================
# STEP 2: HUMAN-IN-THE-LOOP (INSPECT STATE)
# ==========================================
print_separator("STEP 2: PAUSED! (Human Inspection)")

# The graph is paused. Let's look at the state.
snapshot = app.get_state(config)
current_plan = snapshot.values.get("plan")
print(f"👀 Current AI Plan:\n{current_plan}")

input("\nHit Enter to approve the plan and let the AI Code... (Or Ctrl+C to stop)")

# ==========================================
# STEP 3: RESUME EXECUTION
# ==========================================
print_separator("STEP 3: Resuming Graph (AI Coding)")

# Passing 'None' as input just resumes the thread from where it paused
for event in app.stream(None, config, stream_mode="values"):
    if "messages" in event:
        last_msg = event["messages"][-1]
        if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
            print(f"🛠️  AI is calling Tool: {last_msg.tool_calls[0]['name']}")
        elif last_msg.type == "ai":
            print(f"💬 AI says: {last_msg.content}")

# ==========================================
# STEP 4: VERIFY FILE CREATION
# ==========================================
print_separator("STEP 4: Verification")
if os.path.exists("math_utils.py"):
    print("✅ SUCCESS: 'math_utils.py' was created!")
    print(open("math_utils.py").read())
else:
    print("❌ FAIL: File not created.")

# ==========================================
# STEP 5: TIME TRAVEL (Demo Feature)
# ==========================================
# This shows we can "undo" actions in the backend
print_separator("STEP 5: Time Travel (Undo)")

history = list(app.get_state_history(config))
print(f"Found {len(history)} checkpoints in history.")
print_checkpoint_info(history, config)

print("Restoring state to BEFORE the code was written...")

# # Let's verify we can access the plan again
# # print(history)
#
# # history consists of a list of state snapshots, each is a typed dict - messages,
# for i, snap in enumerate(history):
#     print(snap.values)
#     break
#
# # original_plan = history[-2].values.get("plan") # Approximate index for demo
# # print(f"Restored Memory of Plan: {original_plan[:50]}...")