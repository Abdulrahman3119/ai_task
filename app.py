import chainlit as cl
import requests
import json

LLAMA_API_URL = "http://localhost:1234/v1/chat/completions"

WEATHER_API_KEY = 'fb3e1d33812145b3808101042251002'
WEATHER_BASE_URL = 'http://api.weatherapi.com/v1/current.json'

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "format": {
                        "type": "string",
                        "description": "Temperature unit format",
                        "enum": ["celsius", "fahrenheit"],
                        "default": "celsius"
                    }
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "normal_answer",
            "description": "When there is no function that suits the user input",
            "parameters": {
                "type": "object",
                "properties": {
                    "userInput": {
                        "type": "string",
                        "description": "The user input so the model can generate the response",
                    }
                },
                "required": ["userInput"]
            }
        }
    }
]

def get_current_weather(location, format="celsius"):

    params = {'key': WEATHER_API_KEY, 'q': location}

    try:

        response = requests.get(WEATHER_BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()
        current = data['current']
        temp = current['temp_c'] if format == "celsius" else current['temp_f']
        unit = "°C" if format == "celsius" else "°F"
        return f"The weather in {location} is {current['condition']['text']} with a temperature of {temp}{unit}."

    except requests.exceptions.RequestException as e:

        return f"I'm sorry, I couldn't retrieve the weather information at this time. Error: {e}"

def normal_answer(userinput):

    return f"Here is my response: {userinput}"

@cl.on_message
async def main(message):
    user_input = message.content.strip()
    
    payload = {
        "model": "llama-3.2-1b-instruct:2",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. Always return structured JSON."},
            {"role": "user", "content": user_input}
        ],
        "tools": tools,
        "temperature": 0.7,
        "max_tokens": 500,
        "stream": False
    }

    try:
        response = requests.post(LLAMA_API_URL, json=payload)
        response.raise_for_status()
        response_json = response.json()
    except requests.exceptions.RequestException as e:
        await cl.Message(content=f"Error contacting LLaMA API: {e}").send()
        return
    
    
    function_call = response_json.get("choices", [{}])[0].get("message", {}).get("function_call")
    
    if function_call:
    
        function_name = function_call["name"]
    
        arguments = function_call["arguments"]
        
    
        if function_name == "get_current_weather":
    
            result = get_current_weather(arguments["location"], arguments.get("format", "celsius"))
    
        elif function_name == "normal_answer":
    
            result = normal_answer(arguments["userInput"])
    
        else:
    
            result = "Unknown function."

        await cl.Message(content=result).send()

    else:
    
        await cl.Message(content="No function call detected.").send()
