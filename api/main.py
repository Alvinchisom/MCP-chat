import asyncio
import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
from contextlib import asynccontextmanager
from mcp_client import MCP_Client
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
# from mcp_cli import MCP_Client

load_dotenv()


class Settings(BaseSettings):
    server_script_path: str = ""

settings = Settings()

@asynccontextmanager
async def lifespan(app:FastAPI):
    client = MCP_Client()
    try:
        connected = await client.connect(settings.server_script_path)
        if not connected:
            raise HTTPException(
                status_code=500, detail="Server connection failed."
            )
        app.state.client = client
        yield client
    except Exception as e:
        print(f"Error during lifespan:{e}")
        raise e

    finally:
        #shotdown
        await client.cleanup()

app = FastAPI(title="MCP Chat", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


class QueryRequest(BaseModel):
    query: str


class Message(BaseModel):
    role: str
    content: Any


class ToolCall(BaseModel):
    name: str
    args: Dict[str, Any]


@app.post("/query")
async def process_query(request: QueryRequest):
    """Process a query and return the response"""
    try:
        messages = await app.state.client.process_query(request.query)
        return {"messages": messages}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tools")
async def get_tools():
    """Get the list of available tools"""
    try:
        tools = await app.state.client.get_mcp_tools()
        return {
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema,
                }
                for tool in tools
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

