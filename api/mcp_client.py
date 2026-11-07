import asyncio
from json import tool
from typing import Optional
from contextlib import AsyncExitStack
import traceback

# from utils.logger import logger
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from datetime import datetime
from utils.logger import logger
import json
import os
from dotenv import load_dotenv
from anthropic import Anthropic
from anthropic.types import Message
import groq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")



class MCP_Client:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.llm = AsyncGroq(api_key=api_key)
        self.tools = []
        self.messages = []
        self.logger = logger

    # connect to mcp server
    async def connect_to_server(self,server_script_path:str):
        try:
            is_python = server_script_path.endswith(".py")
            is_js = server_script_path.endswith(".js")
            if not (is_python or is_js):
                raise ValueError("Server script must be .py or .js file")
            command = "python" if is_python else "node"
            server_parameters = StdioServerParameters(
                command=command, arguments=[server_script_path], env=None
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

            # get mcl tool
            mcp_tools = await self.get_mcp_tools()
            self.tools = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema,
                }
                for tool in mcp_tools
            ]
            self.logger.info(f"Available tools:{self.tools}")

            return True

        except Exception as e:
            self.logger.error(f"Error Connecting to MCP Server: {e}")
            traceback.print_exc()
            raise

    #call a mcp tool

    #get mcp tool list
    async def get_mcp_tools(self):
        try:
            response = await self.session.list_tools()
            return response.tools
        except Exception as e:
            self.logger.error(f"Error getting tools: {e}")
            raise

    #process query
    async def process_query(self,query:str):
        try:
            self.logger.info(f"Processing query: {query}")
            user_message = {"role": "user","context":query}
            self.message = [user_message]

            while True:
                response = await self.call_llm()

                # if response is a test response
                if response.content[0].type == "text" and len(response.content) == 1:
                    assistant_message = {
                        "role": "assistant",
                        "content": response.content[0].text,
                    }
                    self.messages.append(assistant_message)
                    await self.log_conversation()
                    break

                # if response is a tool calling response
                assistant_message = {
                    "role": "assistant",
                    "content": response.to_dict()["content"],
                }
                self.messages.append(assistant_message)
                await self.log_conversation()

                for content in response.content:
                    if content.type == "tool_use":
                        tool_name = content.name
                        tool_args = content.input
                        tool_use_id = content.id
                        self.logger.info(
                            f"Calling tool {tool_name} with args {tool_args}"
                        )
                        try:
                            result = await self.session.call_tool(tool_name, tool_args)
                            self.logger.info(f"Tool {tool_name} result: {result}...")
                            self.messages.append(
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "tool_result",
                                            "tool_use_id": tool_use_id,
                                            "content": result.content,
                                        }
                                    ],
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

    # call llm
    async def call_llm(self):
        try:
            self.logger.info("Calling LLM")
            return await self.llm.message.create(
                model = "openai/gpt-oss-120b",
                max_token = 2048,
                message = self.message,
                tool = self.tools,
            )

        except Exception as e:
            self.logger.error(f"Error when calling llm: {e}")
            raise

    #clean up
    async def cleanup(self):
        try:
            await self.exit_stack.aclose()
            self.logger.info("Disconnected from MCP server")
        except Exception as e:
            self.logger.error(f"Error During cleanup: {e}")
            traceback.print_exc()
            raise

    #extra

    #log coversation
