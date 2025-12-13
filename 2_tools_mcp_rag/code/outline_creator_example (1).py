from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool

from core.summarizer.summary_getter import get_file_summaries

# --------------------------
# 1. Define custom tools
# --------------------------
@tool
def get_nanochat_summary():
    """
    Summarizes the nanochat code repository, where each file in the repository is summarized.
    The summary can be used for understanding the repository.
    :return:
    """
    print("I AM IN get_nanochat_summary")
    return get_file_summaries()

# --------------------------
# 2. Create tool list
# --------------------------
tools = [get_nanochat_summary]  #

# --------------------------
# 4. Choose LLM
# --------------------------
llm = ChatOpenAI(
    model="google/gemma-3-12b",           # or the model name your LMStudio serves
    openai_api_base="http://localhost:1234/v1",
    openai_api_key="EMPTY",       # LMStudio doesn’t usually need a key
    temperature=0,
)

# --------------------------
# 5. Create agent
# --------------------------
# Create the agent1
agent1 = create_agent(llm, tools)

#------------------------------
# 6. Invoke Agent
# -------------------------------
system_prompt = "You are an expert in authoring micro learning content."
user_query = "Create a micro learning content outline for the nanochat repository"

# Include system prompt in messages
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_query}
]

results = agent1.invoke({"messages": messages})
draft_outline = results["messages"][-1].content

print(draft_outline)

#------------------------------
# 7. Invoke Critique Agent
# -------------------------------
system_prompt = """
You are an expert in understanding code repositories and a great reviewer.
You pay attention to details and emphasize a conceptual understanding before understanding the code.
You are required to review the given micro learning content outline for the nanochat repository.
You must seek to understand the nanochat repository and use it as the basis for your review.
Provide the following outputs and stop:

1. Your feedback on the outline
2. Your rating 1-10 of the outline
3. Your revised outline that incorporates the feedback.
"""

user_query = f"""
Draft Outline for review:
{draft_outline}
"""

# Include system prompt in messages
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_query}
]

results = agent1.invoke({"messages": messages})
print(results["messages"][-1].content)

