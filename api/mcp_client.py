import asyncio
import json
import os
import sys
import traceback
from contextlib import AsyncExitStack
from typing import Optional

from dotenv import load_dotenv
from groq import AsyncGroq
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Support running directly or as a module
try:
    from utils.logger import logger
except ModuleNotFoundError:
    from api.utils.logger import logger


class MCP_Client:
    def __init__(self, model: str = "llama-3.3-70b-versatile"):
        load_dotenv()
        # Checks for GROQ_API_KEY or Groq_api_key in .env
        api_key = os.getenv("GROQ_API_KEY") or os.getenv("Groq_api_key")
        if not api_key:
            raise ValueError(
                "Groq API key not found. Please set GROQ_API_KEY (or Groq_api_key) in your .env file."
            )

        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.llm = AsyncGroq(api_key=api_key)
        self.model = model
        self.tools = []
        self.messages = []
        self.logger = logger

    # Connect to MCP server
    async def connect_to_server(self, server_script_path: str):
        try:
            is_python = server_script_path.endswith(".py")
            is_js = server_script_path.endswith(".js")
            if not (is_python or is_js):
                raise ValueError("Server script must be a .py or .js file")

            # Use current virtualenv's Python on Windows/Mac/Linux
            command = sys.executable if is_python else "node"
            server_parameters = StdioServerParameters(
                command=command,
                arguments=[server_script_path],
                env=dict(os.environ),
            )

            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_parameters)
            )
            self.stdio, self.write = stdio_transport
            self.session = await self.exit_stack.enter_async_context(
                ClientSession(self.stdio, self.write)
            )
            await self.session.initialize()
            self.logger.info("Connected to MCP server")

            # Get MCP tools and format them for Groq (OpenAI function-calling format)
            mcp_tools = await self.get_mcp_tools()
            self.tools = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": tool.inputSchema,
                    },
                }
                for tool in mcp_tools
            ]
            self.logger.info(
                f"Available tools: {[t['function']['name'] for t in self.tools]}"
            )

        except Exception as e:
            self.logger.error(f"Error Connecting to MCP Server: {e}")
            traceback.print_exc()
            raise

    # Get MCP tool list
    async def get_mcp_tools(self):
        try:
            response = await self.session.list_tools()
            return response.tools
        except Exception as e:
            self.logger.error(f"Error getting tools: {e}")
            raise

    # Process query
    async def process_query(self, query: str):
        try:
            self.logger.info(f"Processing query: {query}")
            user_message = {"role": "user", "content": query}
            self.messages.append(user_message)

            while True:
                response = await self.call_llm()
                choice = response.choices[0]
                response_message = choice.message

                # If no tool calls were requested, return the final text response
                if not response_message.tool_calls:
                    assistant_message = {
                        "role": "assistant",
                        "content": response_message.content or "",
                    }
                    self.messages.append(assistant_message)
                    await self.log_conversation()
                    break

                # If the model requested tool calls, add assistant message to history
                self.messages.append(response_message)
                await self.log_conversation()

                # Call each tool requested by the model
                for tool_call in response_message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = (
                        json.loads(tool_call.function.arguments)
                        if isinstance(tool_call.function.arguments, str)
                        else tool_call.function.arguments
                    )
                    self.logger.info(
                        f"Calling tool {tool_name} with args {tool_args}"
                    )
                    try:
                        result = await self.session.call_tool(tool_name, tool_args)
                        # Extract text from MCP CallToolResult
                        if hasattr(result, "content") and isinstance(
                            result.content, list
                        ):
                            content_str = "\n".join(
                                item.text if hasattr(item, "text") else str(item)
                                for item in result.content
                            )
                        else:
                            content_str = str(result)

                        self.logger.info(
                            f"Tool {tool_name} result: {content_str[:200]}..."
                        )

                        # Append tool result in Groq/OpenAI format
                        self.messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": tool_name,
                                "content": content_str,
                            }
                        )
                        await self.log_conversation()
                    except Exception as e:
                        self.logger.error(f"Error calling tool {tool_name}: {e}")
                        raise

            return self.messages

        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            traceback.print_exc()
            raise

    # Call LLM (Groq)
    async def call_llm(self):
        try:
            self.logger.info("Calling LLM (Groq)")
            return await self.llm.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=self.tools if self.tools else None,
                max_tokens=2048,
            )
        except Exception as e:
            self.logger.error(f"Error when calling LLM: {e}")
            raise

    # Clean up
    async def cleanup(self):
        try:
            await self.exit_stack.aclose()
            self.logger.info("Disconnected from MCP server")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
            traceback.print_exc()
            raise

    # Log conversation history to file
    async def log_conversation(self):
        try:
            serialized = []
            for msg in self.messages:
                if hasattr(msg, "model_dump"):
                    serialized.append(msg.model_dump())
                elif isinstance(msg, dict):
                    serialized.append(msg)
                else:
                    serialized.append(str(msg))
            with open("conversation_history.json", "w", encoding="utf-8") as f:
                json.dump(serialized, f, indent=2, default=str)
        except Exception as e:
            self.logger.warning(f"Could not log conversation: {e}")
