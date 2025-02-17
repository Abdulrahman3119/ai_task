import os
import chainlit as cl
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
import requests
import json

os.environ["GOOGLE_API_KEY"] = "AIzaSyC2gYImZO_O1d0b2udXNe8U5i3iCGzAiXI"
WEATHER_API_KEY = 'fb3e1d33812145b3808101042251002'
WEATHER_BASE_URL = 'https://api.weatherapi.com/v1/current.json'

def make_work_order(work_order_data: str) -> str:
    """
    Creates a work order by extracting 'name' and 'details' from a JSON string.
    Expected format (with double quotes):
    {"name": "build", "details": "go for the building"}
    """
    print(f"Received work_order_data: {repr(work_order_data)}") 

    if not work_order_data.strip():
        return "Error: Received empty work_order_data"

    try:
        data = json.loads(work_order_data)
        return (
            f'✅ Work order created!\n'
            f'📝 Name: {data.get("name", "N/A")}\n'
            f'📌 Details: {data.get("details", "N/A")}'
        )
    except json.JSONDecodeError as e:
        return f"❌ Error: Invalid JSON format. Exception: {str(e)}"

def get_current_weather(location: str) -> str:
    """
    Fetches current weather for a given location.
    """
    params = {'key': WEATHER_API_KEY, 'q': location}
    try:
        response = requests.get(WEATHER_BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()
        temp = data['current']['temp_c']
        return (
            f"Weather in {location}: {data['current']['condition']['text']}\n"
            f"Temperature: {temp}°C\n"
            f"Humidity: {data['current']['humidity']}%"
        )
    except Exception as e:
        return f"Weather error: {str(e)}"

def get_current_time(_input: str) -> str:
    return "The current time is: 10:00 PM"

time_tool = Tool(
    name="GetTime",
    func=get_current_time,
    description="Returns the current time."
)

work_order = Tool(
    name="make work order",
    func=make_work_order,
    description="Creates a work order from a JSON string input."
)

weather_tool = Tool(
    name="GetWeather",
    func=get_current_weather,
    description="Returns the current weather for a given location."
)

tools = [weather_tool, time_tool, work_order]

@cl.on_chat_start
async def on_chat_start():
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")
    
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
    )
    
    cl.user_session.set("agent", agent)

@cl.on_message
async def on_message(message: cl.Message):
    agent = cl.user_session.get("agent")
    input_data = {
        "input": message.content,
        "chat_history": [],
        '"name"': "user name"  
    }
    try:
        response = agent.run(input_data)
    except Exception as e:
        response = f"An error occurred: {str(e)}"
    await cl.Message(content=response).send()
