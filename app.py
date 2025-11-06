from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from azure.ai.agents.models import BingGroundingTool
import asyncio
from dotenv import load_dotenv
import os
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient

load_dotenv()

# Initialize environment variables
API_KEY = os.getenv("api_key")
PROJECT_CONNECTION_STRING = os.getenv("PROJECT_CONNECTION_STRING")
BING_CONNECTION_NAME = os.getenv("BING_CONNECTION_NAME")
MODEL_DEPLOYMENT_NAME = os.getenv("MODEL_DEPLOYMENT_NAME")
MODEL_API_VERSION = os.getenv("MODEL_API_VERSION")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
PROJECT_CLIENT_ENDPOINT = os.getenv("PROJECT_CLIENT_ENDPOINT")

###############################################################################
#                              Azure OpenAI Client
###############################################################################
az_model_client = AzureOpenAIChatCompletionClient(
    azure_deployment=MODEL_DEPLOYMENT_NAME,
    model=MODEL_DEPLOYMENT_NAME,
    api_version=MODEL_API_VERSION,
    azure_endpoint=AZURE_ENDPOINT,
    api_key=API_KEY
)

###############################################################################
#                              AI Project Client
###############################################################################
project_client = AIProjectClient(
    endpoint=PROJECT_CLIENT_ENDPOINT,
    credential=DefaultAzureCredential()
)

# Retrieve the Bing connection
bing_connection = project_client.connections.get(name=BING_CONNECTION_NAME)

conn_id = bing_connection.id

###############################################################################
#                               BING QUERY TOOLS
###############################################################################
async def stock_price_trends_tool(stock_name: str) -> str:
    """
    A dedicated Bing call focusing on real-time stock prices,
    changes over the last few months for 'stock_name'.
    """
    print(f"[stock_price_trends_tool] Fetching stock price trends for {stock_name}...")
    bing = BingGroundingTool(connection_id=conn_id)
    agent = project_client.agents.create_agent(
        model="gpt-4o",
        name="stock_price_trends_tool_agent",
        instructions=(
            f"Focus on retrieving real-time stock prices, changes over the last few months, "
            f"and summarize market trends for {stock_name}."
        ),
        tools=bing.definitions,
        headers={"x-ms-enable-preview": "true"}
    )

    # Create a new thread and send the user query
    thread = project_client.agents.create_thread()
    message = project_client.agents.create_message(
        thread_id=thread.id,
        role="user",
        content=f"Please get stock price trends data for {stock_name}."
    )
    # Process the run
    run = project_client.agents.create_and_process_run(thread_id=thread.id, assistant_id=agent.id)
    messages = project_client.agents.list_messages(thread_id=thread.id)

    # Clean up
    project_client.agents.delete_agent(agent.id)

    # Return the Bing result
    return messages["data"][0]["content"][0]["text"]["value"]


async def news_analysis_tool(stock_name: str) -> str:
    """
    A dedicated Bing call focusing on the latest news for 'stock_name'.
    """
    print(f"[news_analysis_tool] Fetching news for {stock_name}...")
    bing = BingGroundingTool(connection_id=conn_id)
    agent = project_client.agents.create_agent(
        model="gpt-4o",
        name="news_analysis_tool_agent",
        instructions=f"Focus on the latest news highlights for the stock {stock_name}.",
        tools=bing.definitions,
        headers={"x-ms-enable-preview": "true"}
    )

    thread = project_client.agents.create_thread()
    message = project_client.agents.create_message(
        thread_id=thread.id,
        role="user",
        content=f"Retrieve the latest news articles and summaries about {stock_name}."
    )
    run = project_client.agents.create_and_process_run(thread_id=thread.id, assistant_id=agent.id)
    messages = project_client.agents.list_messages(thread_id=thread.id)

    # Clean up
    project_client.agents.delete_agent(agent.id)

    return messages["data"][0]["content"][0]["text"]["value"]


async def market_sentiment_tool(stock_name: str) -> str:
    """
    A dedicated Bing call focusing on overall market sentiment
    for 'stock_name'.
    """
    print(f"[market_sentiment_tool] Fetching sentiment for {stock_name}...")
    bing = BingGroundingTool(connection_id=conn_id)
    agent = project_client.agents.create_agent(
        model="gpt-4o",
        name="market_sentiment_tool_agent",
        instructions=(
            f"Focus on analyzing general market sentiment regarding {stock_name}."
        ),
        tools=bing.definitions,
        headers={"x-ms-enable-preview": "true"}
    )

    thread = project_client.agents.create_thread()
    message = project_client.agents.create_message(
        thread_id=thread.id,
        role="user",
        content=(
            f"Gather market sentiment, user opinions, and overall feeling about {stock_name}."
        )
    )
    run = project_client.agents.create_and_process_run(thread_id=thread.id, assistant_id=agent.id)
    messages = project_client.agents.list_messages(thread_id=thread.id)

    # Clean up
    project_client.agents.delete_agent(agent.id)

    return messages["data"][0]["content"][0]["text"]["value"]


async def analyst_reports_tool(stock_name: str) -> str:
    """
    A dedicated Bing call focusing on analyst reports
    for 'stock_name'.
    """
    print(f"[analyst_reports_tool] Fetching analyst reports for {stock_name}...")
    bing = BingGroundingTool(connection_id=conn_id)
    agent = project_client.agents.create_agent(
        model="gpt-4o",
        name="analyst_reports_tool_agent",
        instructions=(
            f"Focus on any relevant analyst reports or professional analyses about {stock_name}."
        ),
        tools=bing.definitions,
        headers={"x-ms-enable-preview": "true"}
    )

    thread = project_client.agents.create_thread()
    message = project_client.agents.create_message(
        thread_id=thread.id,
        role="user",
        content=(f"Find recent analyst reports, price targets, or professional opinions on {stock_name}.")
    )
    run = project_client.agents.create_and_process_run(thread_id=thread.id, assistant_id=agent.id)
    messages = project_client.agents.list_messages(thread_id=thread.id)

    # Clean up
    project_client.agents.delete_agent(agent.id)

    return messages["data"][0]["content"][0]["text"]["value"]


async def expert_opinions_tool(stock_name: str) -> str:
    """
    A dedicated Bing call focusing on expert or industry leaders' opinions
    for 'stock_name'.
    """
    print(f"[expert_opinions_tool] Fetching expert opinions for {stock_name}...")
    bing = BingGroundingTool(connection_id=conn_id)
    agent = project_client.agents.create_agent(
        model="gpt-4o",
        name="expert_opinions_tool_agent",
        instructions=(
            f"Focus on industry expert or thought leader opinions regarding {stock_name}."
        ),
        tools=bing.definitions,
        headers={"x-ms-enable-preview": "true"}
    )

    thread = project_client.agents.create_thread()
    message = project_client.agents.create_message(
        thread_id=thread.id,
        role="user",
        content=(f"Collect expert opinions or quotes about {stock_name}.")
    )
    run = project_client.agents.create_and_process_run(thread_id=thread.id, assistant_id=agent.id)
    messages = project_client.agents.list_messages(thread_id=thread.id)

    # Clean up
    project_client.agents.delete_agent(agent.id)

    return messages["data"][0]["content"][0]["text"]["value"]


###############################################################################
#                              AGENT FUNCTIONS
###############################################################################
#
# These "agent functions" are how each assistant actually calls the above tools.
# The difference is that each AssistantAgent below will have 'tools=[...]'
# pointing to these Python functions. Then the agent can call them
# (directly or via the round-robin workflow).
#
###############################################################################

# -- Trend Data
async def stock_price_trends_agent(stock_name: str) -> str:
    """Agent function for 'stock trends', calls stock_price_trends_tool."""
    return await stock_price_trends_tool(stock_name)

# -- News
async def news_analysis_agent(stock_name: str) -> str:
    """Agent function for 'latest news', calls news_analysis_tool."""
    return await news_analysis_tool(stock_name)

# -- Market Sentiment
async def market_sentiment_agent(stock_name: str) -> str:
    """Agent function for 'market sentiment', calls market_sentiment_tool."""
    return await market_sentiment_tool(stock_name)

# -- Analyst Reports
async def analyst_reports_agent(stock_name: str) -> str:
    """Agent function for 'analyst reports', calls analyst_reports_tool."""
    return await analyst_reports_tool(stock_name)

# -- Expert Opinions
async def expert_opinions_agent(stock_name: str) -> str:
    """Agent function for 'expert opinions', calls expert_opinions_tool."""
    return await expert_opinions_tool(stock_name)


###############################################################################
#                         ASSISTANT AGENT DEFINITIONS
###############################################################################
#
# In RoundRobinGroupChat, each of these agents is called in turn. The system_message
# clarifies each agent’s role, and the 'tools=[...]' argument lists the Python
# functions that agent can call.
#
###############################################################################
stock_trends_agent_assistant = AssistantAgent(
    name="stock_trends_agent",
    model_client=az_model_client,
    tools=[stock_price_trends_agent],
    system_message=(
        "You are the Stock Price Trends Agent. "
        "You fetch and summarize stock prices, changes over the last few months, and general market trends. "
        "Do NOT provide any final investment decision."
    )
)

news_agent_assistant = AssistantAgent(
    name="news_agent",
    model_client=az_model_client,
    tools=[news_analysis_agent],
    system_message=(
        "You are the News Agent. "
        "You retrieve and summarize the latest news stories related to the given stock. "
        "Do NOT provide any final investment decision."
    )
)

sentiment_agent_assistant = AssistantAgent(
    name="sentiment_agent",
    model_client=az_model_client,
    tools=[
        market_sentiment_agent,
        analyst_reports_agent,
        expert_opinions_agent
    ],
    system_message=(
        "You are the Market Sentiment Agent. "
        "You gather overall market sentiment, relevant analyst reports, and expert opinions. "
        "Do NOT provide any final investment decision."
    )
)

decision_agent_assistant = AssistantAgent(
    name="decision_agent",
    model_client=az_model_client,
    # The final agent typically calls the 'investment_decision_agent' to
    # synthesize all the data. If you want it to call the other tools directly,
    # you can also add them here. But typically we rely on the round-robin approach.
    # tools=[investment_decision_agent],
    system_message=(
        "You are the Decision Agent. After reviewing the stock data, news, sentiment, analyst reports, "
        "and expert opinions from the other agents, you provide the final investment decision. In the final decision make a call to either Invest or Not. Also provide the current stock price. "
        "End your response with 'Decision Made' once you finalize the decision."
    )
)

###############################################################################
#                        TERMINATION & TEAM CONFIGURATION
###############################################################################
# Stop once "Decision Made" is in the response, or if 15 messages have passed
text_termination = TextMentionTermination("Decision Made")
max_message_termination = MaxMessageTermination(15)
termination = text_termination | max_message_termination

# Round-robin chat among the four agents
investment_team = RoundRobinGroupChat(
    [
        stock_trends_agent_assistant,
        news_agent_assistant,
        sentiment_agent_assistant,
        decision_agent_assistant,
    ],
    termination_condition=termination
)

###############################################################################
#                                   MAIN
###############################################################################
async def main():
    stock_name = "Websol Energy Systems Ltd"
    await Console(
        investment_team.run_stream(
            task=f"Analyze stock trends, news, and sentiment for {stock_name}, plus analyst reports and expert opinions, and then decide whether to invest."
        )
    )

if __name__ == "__main__":
    import asyncio
    import sys

    # Fix Windows asyncio bug
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main())
