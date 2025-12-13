import json
import os
from typing import TypedDict, List
from langgraph.graph import StateGraph, END

from core.case_study_outline_generator.process_slides import get_contents_for_ppt
from core.case_study_outline_generator.outline_generator import create_outline_from_slides, outline_critic


# To simulate LLM calls, you would typically use a library like langchain_openai
# from langchain_openai import ChatOpenAI
# For this example, we will use mock functions instead.

# --- 1. Define the State for the Graph ---
# The state is a dictionary that will be passed between nodes.
# It holds all the information necessary for the graph to run.

class AgentState(TypedDict):
    """
    Represents the state of our agentic workflow.
    """
    ppt_filename: str  # The input presentation file
    chapter_outline: str  # The generated chapter outline
    feedback: List[str]  # A list of feedback points from the critic
    retry_count: int  # A counter for the number of revision attempts
    human_feedback: str  # Feedback from the human user


def generate_outline_agent(state: AgentState):
    """
    Generates or refines the chapter outline.
    Now prioritizes human feedback over AI critic feedback.
    """
    print("--- 👨‍💻 Generator: Generating Outline ---")
    retry_count = state.get('retry_count', 0)
    critic_feedback = state.get('feedback', [])
    human_feedback = state.get('human_feedback')

    # Prioritize human feedback for revisions
    if human_feedback and human_feedback != "approved":
        print(f"   -> Revision based on HUMAN feedback.")
        outline = create_outline_from_slides(
            outline=json.dumps(state.get("chapter_outline", "")),
            feedback=f"human: {human_feedback}"
        )

        # print("#" * 100)
        # print("After HITL: ", outline)

        # After using human feedback, clear it and reset the AI state
        return {
            "chapter_outline": outline,
            "retry_count": 0,  # Reset AI retry count
            "feedback": [],  # Clear old AI critic feedback
            "human_feedback": ""  # Consume the feedback
        }

    # If no human feedback, proceed with AI critic feedback
    if critic_feedback:
        print(f"   -> Revision attempt #{retry_count + 1}. Incorporating AI critic feedback.")
        last_feedback = critic_feedback[-1]
        outline = create_outline_from_slides(
            outline=json.dumps(state.get("chapter_outline", "")),
            feedback=last_feedback
        )
    else:
        # First run
        print("   -> First attempt. Creating a basic outline.")
        contents = get_contents_for_ppt(state["ppt_filename"])
        contents_json = json.dumps(contents)

        print("#" * 100)
        print(contents)

        outline = create_outline_from_slides(contents=contents_json)

    print(f"   -> Generated Outline:\n{outline}\n")

    return {
        "chapter_outline": outline,
        "retry_count": (retry_count or 0) + 1
    }


def outline_critic_agent(state: AgentState):
    """
    A mock node that simulates a critic reviewing the chapter outline.
    It provides feedback if the outline doesn't meet certain criteria.
    """
    print("--- 🕵️‍♀️ Critic: Reviewing Outline ---")
    outline = state['chapter_outline']
    outline = json.dumps(outline)
    current_feedback = outline_critic(outline)
    return {"feedback": current_feedback}


# --- 3. Define Conditional Edges (Control Flow) ---
# This function decides the next step based on the current state.

MAX_RETRIES = 3


def human_review_node(state: AgentState):
    """
    A new node to formally handle the human review process.
    """
    print("\n--- ✋ HUMAN REVIEW ---")
    print("The AI agents have produced the following outline:")
    print("=========================================")
    print(state['chapter_outline'])
    print("=========================================")

    while True:
        human_input = input("Do you approve this outline? (yes/no): ").lower()
        if human_input in ["yes", "y"]:
            print("✅ Outline approved by human.")
            return {"human_feedback": "approved"}
        elif human_input in ["no", "n"]:
            human_feedback_text = input("❌ Outline rejected. Please provide your feedback for revision: ")
            return {"human_feedback": human_feedback_text or "No feedback provided."}
        else:
            print("Invalid input. Please enter 'yes' or 'no'.")


def should_continue_or_request_approval(state: AgentState) -> str:
    """
    Determines the next step: revise with AI, or ask for human review.
    """
    print("--- 🤔 AI/Human Decision Point ---")
    feedback = state.get('feedback', [])
    retry_count = state.get('retry_count', 0)

    if not feedback or retry_count >= MAX_RETRIES:
        if retry_count >= MAX_RETRIES:
             print(f"   -> Reached maximum AI retries ({MAX_RETRIES}). Proceeding to Human Review.")
        else:
             print("   -> AI Critic is satisfied. Proceeding to Human Review.")
        return "request_approval"
    else:
        print(f"   -> AI Critic has feedback. Continuing to AI revision.")
        return "continue_revision"


def route_after_human_review(state: AgentState) -> str:
    """
    Routes to the generator for rework or ends the process based on human input.
    """
    print("---  Routing after human review ---")
    if state.get("human_feedback") == "approved":
        print("   -> Human approved. Ending process.")
        return END
    print("   -> Human requested changes. Rerouting to generator.")
    return "rework"


def build_app_with_hitl():
    workflow = StateGraph(AgentState)

    workflow.add_node("outline_generator", generate_outline_agent)
    workflow.add_node("outline_critic", outline_critic_agent)
    workflow.add_node("human_review", human_review_node)  # Add the new node

    workflow.set_entry_point("outline_generator")

    workflow.add_edge("outline_generator", "outline_critic")

    workflow.add_conditional_edges(
        "outline_critic",
        should_continue_or_request_approval,
        {
            "continue_revision": "outline_generator",
            "request_approval": "human_review"  # Go to the new human review node
        }
    )

    workflow.add_conditional_edges(
        "human_review",
        route_after_human_review,
        {
            "rework": "outline_generator",  # Loop back to the generator
            END: END
        }
    )

    # Compile the graph. No interruption is needed now.
    app = workflow.compile()
    app.get_graph().draw_mermaid_png(output_file_path="hitl_outline1.png")

    return app


# --- 5. Run the Graph ---
if __name__ == '__main__':
    ppt_folder = r"C:\home\ananth\trainings\adobe\genai_2025\agentic_ai_july_2025"
    name = "session1_beyond_parrots_and_calculators.pptx"
    ppt_file_name = os.path.join(ppt_folder, name)

    # Construct the Graph
    # app = build_app()
    app = build_app_with_hitl()

    # Define the initial input for the workflow
    initial_input = {
        "ppt_filename": ppt_file_name,
        "retry_count": 0,
        "chapter_outline": "",
        "feedback": None
    }

    print("🚀 Starting Agentic Outline Generation Process...\n")

    # Get the final state after the graph has finished running
    final_state = app.invoke(initial_input)

    print("\n✅ Process Finished!")
    print("=========================================")
    print("Final Approved Chapter Outline:")
    print("=========================================")
    print(final_state['chapter_outline'])
    print(f"\nTotal revisions required: {final_state['retry_count'] - 1}")
