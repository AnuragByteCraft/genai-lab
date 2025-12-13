from agents import Agent, Runner
from core.utils.openai_agent_sdk_settings import initialize_openai_agent_sdk

agent = Agent(name="Assistant", instructions="You are a helpful assistant")

# We need to do this first in order to make our local LLM as default
initialize_openai_agent_sdk()

result = Runner.run_sync(agent, "Write a haiku about recursion in programming.")
print(result.final_output)
