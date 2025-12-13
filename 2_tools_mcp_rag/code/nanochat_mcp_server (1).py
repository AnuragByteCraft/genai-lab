import json
from typing import Optional
from mcp.server.fastmcp import FastMCP

from core.session1_foundations.model_code import do_rag
from core.summarizer.summarize_repo import read_summaries


# Initialize FastMCP with a specific port
mcp = FastMCP("NanochatMCPService")


@mcp.tool()
async def rag_tool(query: str) -> str:
    """
    This tool is meant for question answering pertaining to nanochat repository.
    This doesn't provide summarization of each file of the repository.
    For such summarization requirements, invoke the tool: summarize_tool.
    Given a query pertaining to nanochat code repository, perform RAG and return the results.

    Args:
        query: Query pertaining to nanochat code repository.

    Returns:
        Result of the rag tool that answers to the input query.
    """
    try:
        result = do_rag(query)
        return str(result) if result else "No results found"
    except Exception as e:
        return f"Error: {str(e)}"


@mcp.tool()
async def summarize_tool() -> str:
    """
    This tool provides the summary of each file in the nanochat code repository.
    Returns a JSON string containing a dictionary where each key is a filename 
    and each value contains a 'summary' field with the actual summary text.

    Args:
        None

    Returns:
        JSON string containing summaries of all files in the nanochat code repository.
    """
    try:
        summaries = read_summaries()
        # Convert dict to JSON string for MCP compatibility
        return json.dumps(summaries, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == "__main__":
    # Set up logging
    import logging

    logging.basicConfig(level=logging.INFO)
    mcp.run(transport="streamable-http")
