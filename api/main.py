import asyncio
import os
import sys

# Ensure the api directory is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mcp_client import MCP_Client


async def main():
    if len(sys.argv) < 2:
        print("Usage: python api/main.py <path_to_server_script.py|js>")
        print("Example: python api/main.py my_mcp_server.py")
        return

    server_script = sys.argv[1]
    client = MCP_Client()

    try:
        await client.connect_to_server(server_script)
        print("\nConnected! Available tools loaded.")
        print("Type your question below (or type 'exit' to quit):\n")

        while True:
            query = input("You: ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit", "q"]:
                break

            response = await client.process_query(query)
            # Display the last assistant response
            last_message = response[-1]
            content = (
                last_message.get("content")
                if isinstance(last_message, dict)
                else getattr(last_message, "content", "")
            )
            print(f"\nAssistant: {content}\n")

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
