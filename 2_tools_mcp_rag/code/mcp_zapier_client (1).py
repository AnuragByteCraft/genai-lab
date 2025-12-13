import asyncio
import json

from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

from core.utils.print_mcp_tools import pretty_print_tools

# Create the transport with your MCP server URL
server_url = "https://mcp.zapier.com/api/mcp/s/ZDU5NTM0ODktMzMwZi00ZmIxLTkwNzYtZjUxMTY4MmJiZjBhOmQ2NDAxNGI4LTBjYTYtNDgxNi1hYjk1LTM4MzFhNTY2MDQ3Mw==/mcp"
transport = StreamableHttpTransport(server_url)

# Initialize the client with the transport
client = Client(transport=transport)


async def main():
    # Connection is established here
    print("Connecting to MCP server...")
    async with client:
        print(f"Client connected: {client.is_connected()}")

        # Make MCP calls within the context
        print("Fetching available tools...")
        tools = await client.list_tools()

        print(f"Available tools: {json.dumps([t.name for t in tools], indent=2)}")
        pretty_print_tools(tools)

        print("Calling gmail_find_email...")
        result = await client.call_tool(
            "gmail_send_email",
            {
                "instructions": "Execute the Gmail: Find Email tool with the following parameters",
                # "query": "example-string",
                "to": "narayana.anantharaman@gmail.com",
                "subject": "from Ananth!",
                "body": "Hello World, hello all, this is our agents session!!!!!"
            },
        )
        # print(result.content)
        # Parse the JSON string from the TextContent and print it nicely formatted
        json_result = json.loads(result.content[0].text)
        print(
            f"\ngmail_find_email result:\n{json.dumps(json_result, indent=2)}"
        )

    # Connection is closed automatically when exiting the context manager
    print("Example completed")


if __name__ == "__main__":
    asyncio.run(main())
