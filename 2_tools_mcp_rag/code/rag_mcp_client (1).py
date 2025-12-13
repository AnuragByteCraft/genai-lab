import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

from langchain_openai import ChatOpenAI  # Use LangChain's OpenAI-compatible chat model
from core.utils.print_dict import print_dict, pretty_print

model_id = "openai/gpt-oss-20b"


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
            "nanochat_rag": {
                "url": "http://localhost:8100/mcp",
                "transport": "streamable_http",

            }
        }
    )
    tools = await client.get_tools()
    print("tools = ", tools)
    agent = create_agent(
        llm,
        tools,
        system_prompt="Choose the best tool that can answer the given query."
    )

    while True:
        prompt = input("Query> ")
        if prompt in ["exit", "quit", "q", "bye"]:
            break
        messages = {"messages": [{"role": "user", "content": prompt}]}
        rag_response = await agent.ainvoke(messages)
        print("RAG response:", rag_response)
        pretty_print(rag_response)
        print(rag_response["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
