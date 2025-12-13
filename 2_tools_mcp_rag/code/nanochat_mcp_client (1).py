"""
This module is a generalized code from rag_mcp_client.py by adding more MCP functions for nanochat repo access.
This accesses nanochat MCP server.
This module is the MCP client sample to access MCP functions that operate on nanochat repository.

Example tools available in nanochat MCP Server:

1. rag tool: given a query pertaining to the nanochat codebase or concepts, returns the response
2. list files: provides the folder structure of the nanochat repo
3. summarize_file: given the file id, returns the summary of the file
4. summarize_all: returns the complete summary for every file
"""
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

from langchain_openai import ChatOpenAI  # Use LangChain's OpenAI-compatible chat model
from core.utils.print_dict import print_dict, pretty_print

model_id = "google/gemma-3-12b"


async def main():
    # Instantiate the chat model pointing to your LMStudio local server
    llm = ChatOpenAI(
        model=model_id,  # or the model name your LMStudio serves
        openai_api_base="http://localhost:1234/v1",
        openai_api_key="EMPTY",  # LMStudio usually does not require a key; use "EMPTY" or ""
        temperature=0,
    )

    client = MultiServerMCPClient(
        {
            "NanochatMCPService": {
                "url": "http://localhost:8000/mcp",
                "transport": "streamable_http",
            }
        }
    )
    tools = await client.get_tools()
    print("tools = ", tools)

    agent = create_agent(
        llm,
        tools,
        system_prompt="You are given a set of tools. Choose the best tool that can respond to the given input. If you are asked for code summaries, obtain the summaries from the tool and return that without modification in JSON format."
    )

    while True:
        prompt = input("Prompt: > ")
        if prompt in ["exit", "quit", "q", "bye"]:
            break
        messages = {"messages": [{"role": "user", "content": prompt}]}
        rag_response = await agent.ainvoke(messages)
        print("RAG response:", rag_response)
        pretty_print(rag_response)
        print(rag_response["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
