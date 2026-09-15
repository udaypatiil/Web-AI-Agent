import os

from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langgraph.checkpoint.postgres import PostgresSaver

DB_URI = os.getenv("SUPABASE_DB_URI")

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

with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()
    agent = create_agent(
        model=llm,
        tools=[get_weather, get_location],
        system_prompt=system_prompt,
        checkpointer=checkpointer,
    )
    #"InMemorySaver agent ki conversation history ko memory mein store karta hai.
    # thread_id ek conversation ki unique identity ki tarah kaam karta hai.
    # Same thread_id use karne par previous messages next invocation mein available rehte hain."

    while True:
        user_query = input("Enter your query: ")
        if user_query in ["bye", "exit", "quit"]:
            break
        response = agent.invoke({"messages":[{'role':'user','content':user_query}]},
                                 {"configurable":{"thread_id": "1"}})
        # for i in response["messages"]:
        #     if i.type == 'human':
        #         print("you:", i.text)
        #     if i.type == 'ai' and i.text:
        #         print("Agent:", i.text)

        print(response["messages"][-1].text)