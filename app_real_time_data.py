import asyncio
import os
from datetime import datetime
from dotenv import load_dotenv
import yfinance as yf

# === Azure AutoGen Imports ===
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.agents.models import BingGroundingTool

# === Load environment ===
load_dotenv()

API_KEY = os.getenv("api_key")
PROJECT_CONNECTION_STRING = os.getenv("PROJECT_CONNECTION_STRING")
BING_CONNECTION_NAME = os.getenv("BING_CONNECTION_NAME")
MODEL_DEPLOYMENT_NAME = os.getenv("MODEL_DEPLOYMENT_NAME")
MODEL_API_VERSION = os.getenv("MODEL_API_VERSION")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
PROJECT_CLIENT_ENDPOINT=os.getenv("PROJECT_CLIENT_ENDPOINT")

###############################################################################
#                          AZURE CLIENT SETUP
###############################################################################
az_model_client = AzureOpenAIChatCompletionClient(
    azure_deployment=MODEL_DEPLOYMENT_NAME,
    model=MODEL_DEPLOYMENT_NAME,
    api_version=MODEL_API_VERSION,
    azure_endpoint=AZURE_ENDPOINT,
    api_key=API_KEY,
)

project_client = AIProjectClient(
    endpoint=PROJECT_CLIENT_ENDPOINT,
    credential=DefaultAzureCredential(),
)

bing_connection = project_client.connections.get(name=BING_CONNECTION_NAME)
conn_id = bing_connection.id

###############################################################################
#                          TOOL DEFINITIONS
###############################################################################

async def stock_price_trends_tool(stock_name: str) -> str:
    """Bing-based stock trend summary."""
    print(f"[stock_price_trends_tool] Fetching trends for {stock_name}...")
    bing = BingGroundingTool(connection_id=conn_id)
    agent = project_client.agents.create_agent(
        model="gpt-4o",
        name="stock_trends_agent_tool",
        instructions=f"Summarize recent price movements and trends for {stock_name}.",
        tools=bing.definitions,
        headers={"x-ms-enable-preview": "true"},
    )

    thread = project_client.agents.create_thread()
    project_client.agents.create_message(
        thread_id=thread.id,
        role="user",
        content=f"Get stock price trends for {stock_name}.",
    )
    project_client.agents.create_and_process_run(thread_id=thread.id, assistant_id=agent.id)
    messages = project_client.agents.list_messages(thread_id=thread.id)
    project_client.agents.delete_agent(agent.id)

    return messages["data"][0]["content"][0]["text"]["value"]

# === Real-Time Price Tool using yfinance ===
async def realtime_stock_price_tool(stock_name: str) -> str:
    """Fetch current stock price using yfinance (real-time)."""
    print(f"[realtime_stock_price_tool] Getting real-time price for {stock_name}...")
    ticker = yf.Ticker(stock_name)
    data = ticker.history(period="1d", interval="1m")
    if data.empty:
        return f"Could not fetch real-time price for {stock_name}."
    current_price = data["Close"].iloc[-1]
    timestamp = data.index[-1].strftime("%Y-%m-%d %H:%M:%S")
    return f"As of {timestamp}, the real-time price of {stock_name} is ₹{current_price:.2f}."

# === Bing-based news ===
async def news_analysis_tool(stock_name: str) -> str:
    print(f"[news_analysis_tool] Fetching news for {stock_name}...")
    bing = BingGroundingTool(connection_id=conn_id)
    agent = project_client.agents.create_agent(
        model="gpt-4o",
        name="news_analysis_tool_agent",
        instructions=f"Summarize the latest financial news for {stock_name}.",
        tools=bing.definitions,
        headers={"x-ms-enable-preview": "true"},
    )

    thread = project_client.agents.create_thread()
    project_client.agents.create_message(
        thread_id=thread.id,
        role="user",
        content=f"Find the latest stock news about {stock_name}.",
    )
    project_client.agents.create_and_process_run(thread_id=thread.id, assistant_id=agent.id)
    messages = project_client.agents.list_messages(thread_id=thread.id)
    project_client.agents.delete_agent(agent.id)
    return messages["data"][0]["content"][0]["text"]["value"]

###############################################################################
#                          ASSISTANT AGENTS
###############################################################################

stock_trends_agent_assistant = AssistantAgent(
    name="stock_trends_agent",
    model_client=az_model_client,
    tools=[stock_price_trends_tool, realtime_stock_price_tool],
    system_message=(
        "You are the Stock Trends Agent. You analyze both historical and real-time data "
        "to summarize current stock movement patterns."
    ),
)

news_agent_assistant = AssistantAgent(
    name="news_agent",
    model_client=az_model_client,
    tools=[news_analysis_tool],
    system_message=(
        "You are the News Agent. Retrieve and summarize the latest relevant news for the stock."
    ),
)

sentiment_agent_assistant = AssistantAgent(
    name="sentiment_agent",
    model_client=az_model_client,
    system_message=(
        "You are the Sentiment Agent. Summarize the overall market mood "
        "and investor confidence for the given stock."
    ),
)

decision_agent_assistant = AssistantAgent(
    name="decision_agent",
    model_client=az_model_client,
    system_message=(
        "You are the Decision Agent. Combine insights from trends, real-time prices, and news "
        "to decide whether to INVEST or NOT INVEST. End with 'Decision Made'."
    ),
)

###############################################################################
#                          ROUND ROBIN GROUP CHAT
###############################################################################

termination = TextMentionTermination("Decision Made") | MaxMessageTermination(15)

investment_team = RoundRobinGroupChat(
    [
        stock_trends_agent_assistant,
        news_agent_assistant,
        sentiment_agent_assistant,
        decision_agent_assistant,
    ],
    termination_condition=termination,
)

###############################################################################
#                          MAIN LOOP (REAL-TIME UPDATES)
###############################################################################

async def run_realtime_analysis(stock_name: str, interval: int = 60):
    """Run real-time analysis every N seconds."""
    while True:
        print(f"\n[{datetime.now():%H:%M:%S}] Running real-time round-robin analysis...\n")
        await Console(
            investment_team.run_stream(
                task=f"Analyze trends, real-time prices, and latest news for {stock_name}. "
                     "Then make an investment decision."
            )
        )
        print(f"\nWaiting {interval} seconds before next update...\n")
        await asyncio.sleep(interval)

async def main():
    stock_name = "ICICIBANK.NS"  # NSE symbol for yfinance
    await run_realtime_analysis(stock_name, interval=60)

if __name__ == "__main__":
    import sys
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
