from typing import Optional
from mcp.server.fastmcp import FastMCP
from core.session1_foundations.model_code import do_rag

# mcp = FastMCP("Weather")
port = 8100
# Initialize FastMCP with a specific port
mcp = FastMCP("MyMCPService", port=port)


@mcp.tool()
async def rag_tool(query: str) -> Optional[str]:
    """
    Given a query pertaining to nanochat code repository, perform RAG and return the results.

    Args:
        query: The city name or zip code to get weather for

    Returns:
        Result of the rag tool that answers to the input query.
    """
    try:
        return do_rag(query)
    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == "__main__":
    # Set up logging
    import logging

    logging.basicConfig(level=logging.INFO)
    # mcp.run(transport="streamable-http", port=8100)
    mcp.run(transport="streamable-http")

