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


# --- 2. Define the Agent Nodes (Generator and Critic) ---
# These are the core functions that will perform the work.

def generate_outline_agent(state: AgentState):
    """
    A mock node that simulates generating a chapter outline from a PPT.
    In a real implementation, this would involve parsing the PPT and calling an LLM.
    """
    print("--- 👨‍💻 Generator: Generating Outline ---")
    retry_count = state.get('retry_count', 0)
    feedback = state.get('feedback', [])

    # Simulate improving the outline based on feedback
    if not feedback:
        print("   -> First attempt. Creating a basic outline.")
        # Initial basic outline
        contents = get_contents_for_ppt(state["ppt_filename"])
        contents = json.dumps(contents)

        print("#" * 100)
        print(contents)

        outline = create_outline_from_slides(contents)
    else:
        # we have now some feedback - we need to consider slide contents, current outline and critic's feedback
        # and generate the new outline
        print(f"   -> Revision attempt #{retry_count}. Incorporating feedback.")
        # Simulate improvement based on the last feedback
        last_feedback = feedback[-1]
        outline = json.dumps(state["chapter_outline"])
        outline = create_outline_from_slides(outline=outline, feedback=last_feedback)

    print(f"   -> Generated Outline:\n{outline}\n")

    return {
                "chapter_outline": outline,
                "retry_count": retry_count + 1  # Increment retry counter
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


def should_continue(state: AgentState) -> str:
    """
    Determines whether to continue with revisions or to end the process.
    """
    print("--- 🤔 Decision Point ---")
    feedback = state['feedback']
    retry_count = state['retry_count']

    if not feedback:
        print("   -> Critic is satisfied. Concluding the process.")
        return "end"
    elif retry_count >= MAX_RETRIES:
        print(f"   -> Reached maximum retries ({MAX_RETRIES}). Ending process.")
        return "end"
    else:
        print(f"   -> Critic has feedback. Continuing to revision #{retry_count + 1}.")
        return "continue"


# --- 4. Construct the Graph ---


def build_app():
    # Initialize the state graph
    workflow = StateGraph(AgentState)

    # Add the nodes to the graph
    workflow.add_node("outline_generator", generate_outline_agent)
    workflow.add_node("outline_critic", outline_critic_agent)

    # Set the entry point of the graph
    workflow.set_entry_point("outline_generator")

    # Add edges to define the flow
    workflow.add_edge("outline_generator", "outline_critic")

    # Add the conditional edge for the feedback loop
    workflow.add_conditional_edges(
        "outline_critic",
        should_continue,
        {
            "continue": "outline_generator",  # If feedback exists and retries are not exhausted, go back to the generator
            "end": END  # Otherwise, end the workflow
        }
    )

    # Compile the graph into a runnable application
    app = workflow.compile()
    app.get_graph().draw_mermaid_png(output_file_path="outline1.png")

    return app


# --- 5. Run the Graph ---
if __name__ == '__main__':
    ppt_folder = r"C:\home\ananth\trainings\adobe\genai_2025\agentic_ai_july_2025"
    name = "session1_beyond_parrots_and_calculators.pptx"

    # ppt_folder = r"C:\home\ananth\trainings\adobe\genai_2025\batch2_may_june"
    # name = "session1_beyond_parrots_and_calculators.pptx"

    ppt_file_name = os.path.join(ppt_folder, name)

    # Construct the Graph
    app = build_app()

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
