import os

from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

def get_weather(city: str):
    """Get Weather for a given city"""
    return {"condition":"sunny","temperature": 25}

def get_location():
    """Get user's current location. use this when user asks about weather
        without specifying a city"""
    return "Rome,Italy"

llm = ChatGoogleGenerativeAI(
    api_key=os.getenv("GOOGLE_API_KEY"),
    model="gemini-3-flash-preview",
    temperature=0.7,
)

system_prompt = """
    you are helpful weather assistant.
    YOUR WORKFLOW:
        1. if the user ask about weather WITHOUT specifying a location you MUST:
            - first call get_location() to find their location
            - then call get_weather(city) with their location
            
        2. if the user provides a city, call get_weather(city) directly.
"""
agent = create_agent(
    model=llm,
    tools=[get_weather, get_location],
    system_prompt=system_prompt,
    checkpointer=InMemorySaver(),
)

user_query1 = input("Enter your query: ")
response1 = agent.invoke({"messages":[{'role':'user','content':user_query1}]},
                         {"configurable":{"thread_id": "1"}})
print(response1["messages"][-1].text)

user_query2 = input("Enter your query: ")
response2 = agent.invoke({"messages":[{'role':'user','content':user_query2}]},
                         {"configurable":{"thread_id": "1"}})
print(response2["messages"][-1].text)