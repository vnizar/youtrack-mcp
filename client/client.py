from google import genai
from google.genai.types import GenerateContentConfig, Tool
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import os
import asyncio
import json
import argparse
from typing import Literal

from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_deepseek import ChatDeepSeek
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents import AgentExecutor
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser

class CustomEncoder(json.JSONEncoder):
    def default(self, o):
        if hasattr(o, "content"):
            return {"type": o.__class__.__name__, "content": o.content}
        return super().default(o)

def create_ollama_agent(llm, tools):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant that can use tools to help answer questions. "
                  "You have access to the following tools:\n\n{tools}\n\n"
                  "Use these tools to help answer the user's question. "
                  "If you don't know the answer or can't use the tools to find it, say so."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    llm_with_tools = llm.bind(
        functions=[convert_to_openai_function(t) for t in tools]
    )

    agent = (
        {
            "input": lambda x: x["input"],
            "chat_history": lambda x: x.get("chat_history", []),
            "agent_scratchpad": lambda x: format_to_openai_function_messages(
                x["intermediate_steps"]
            ),
            "tools": lambda x: "\n".join([f"{tool.name}: {tool.description}" for tool in tools]),
        }
        | prompt
        | llm_with_tools
        | OpenAIFunctionsAgentOutputParser()
    )

    return AgentExecutor(agent=agent, tools=tools, verbose=True)

def get_llm(model_type: Literal["gemini", "deepseek", "ollama"], model_name: str, temperature: float = 0):
    if model_type == "gemini":
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            max_retries=2,
            google_api_key=os.getenv("GEMINI_API_KEY")
        )
    elif model_type == "deepseek":
        return ChatDeepSeek(
            model=model_name,
            temperature=temperature,
            api_key=os.getenv("DEEPSEEK_API_KEY")
        )
    elif model_type == "ollama":
        return ChatOllama(
            model=model_name,
            temperature=temperature,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def parse_args():
    parser = argparse.ArgumentParser(description="MCP Client with model selection")
    parser.add_argument(
        "--model-type",
        choices=["gemini", "deepseek", "ollama"],
        default="gemini",
        help="Type of model to use (gemini, deepseek, or ollama)"
    )
    parser.add_argument(
        "--model-name",
        default={
            "gemini": "gemini-2.5-pro-preview-05-06",
            "deepseek": "deepseek-chat",
            "ollama": "deepseek-llm:7b"
        }.get(parser.get_default("model_type")),
        help="Name of the model to use"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0,
        help="Temperature for model generation (0-1)"
    )
    parser.add_argument(
        "--ollama-base-url",
        default="http://localhost:11434",
        help="Base URL for Ollama API (default: http://localhost:11434)"
    )
    parser.add_argument(
        "--youtrack-url",
        default=os.getenv("YOUTRACK_URL"),
        help="YouTrack instance URL (default: from YOUTRACK_URL env var)"
    )
    parser.add_argument(
        "--youtrack-token",
        default=os.getenv("YOUTRACK_TOKEN"),
        help="YouTrack API token (default: from YOUTRACK_TOKEN env var)"
    )
    return parser.parse_args()

server_path = os.path.abspath(__file__).replace("client/client.py", "server/youtrack.py")

def get_server_params(args):
    if not args.youtrack_url or not args.youtrack_token:
        raise ValueError("YouTrack URL and token are required. Set them via environment variables or command line arguments.")
    
    return StdioServerParameters(
        command="python",
        args=[
            server_path,
            "--youtrack-url", args.youtrack_url,
            "--youtrack-token", args.youtrack_token
        ],
        env={}  # No need to pass env vars since we're using command line args
    )

mcp_client = None

async def run():
    global mcp_client
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            mcp_client = type("MCPClientHolder", (), {"session": session})()
            mcp_tools = await load_mcp_tools(session)
            agent = create_react_agent(llm, mcp_tools)
            print("MCP Client Started with ReAct Agent! Type 'quit' to exit.")

            while True:
                query = input("\\nQuery: ").strip()
                if query.lower() == "quit":
                    break
                
                try:
                    response = await agent.ainvoke({"messages": query})

                    try:
                        formatted = json.dumps(response, indent=2, cls=CustomEncoder)
                    except Exception:
                        formatted = str(response)
                    print("\\nResponse:")
                    print(formatted)

                except Exception as e:
                    print(f"\nError: {str(e)}")
                
    return

if __name__ == "__main__":
    args = parse_args()
    if args.model_type == "ollama":
        os.environ["OLLAMA_BASE_URL"] = args.ollama_base_url
    
    # Get server parameters with YouTrack credentials
    server_params = get_server_params(args)
    
    llm = get_llm(args.model_type, args.model_name, args.temperature)
    results = asyncio.run(run())
    print("\nAll tasks completed!")