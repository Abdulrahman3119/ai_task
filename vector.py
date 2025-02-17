import os
import chainlit as cl
import requests
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 

os.environ["GOOGLE_API_KEY"] = "AIzaSyAHGHJG_jsdk97QlqkmAlmN4uCDbSPC0cE"

DATA_PATH = 'Groth_Goodwin_2010.pdf'
DB_FAISS_PATH = 'db_faiss'

def create_vector_db():
    loader = PyPDFLoader(DATA_PATH)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
    
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)
    return db

def load_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
    return FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)


if not os.path.exists(DB_FAISS_PATH):
    create_vector_db()
vectorstore = load_vector_db()

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})


def vector_search(query: str) -> str:
    try:
        docs = retriever.get_relevant_documents(query)
        if not docs:
            return "No relevant documents found."
        return "\n\n".join([doc.page_content for doc in docs])
    except Exception as e:
        return f"Vector search error: {str(e)}"


def make_work_order(work_order_data: str) -> str:
    print(f"Received work_order_data: {repr(work_order_data)}")
    if not work_order_data.strip():
        return "Error: Received empty work_order_data"
    try:
        data = json.loads(work_order_data)
        return (
            f' Work order created!\n'
            f' Name: {data.get("name", "N/A")}\n'
            f' Details: {data.get("details", "N/A")}'
        )
    except json.JSONDecodeError as e:
        return f" Error: Invalid JSON format. Exception: {str(e)}"


def get_current_weather(location: str) -> str:
    WEATHER_API_KEY = 'fb3e1d33812145b3808101042251002'
    WEATHER_BASE_URL = 'https://api.weatherapi.com/v1/current.json'
    
    params = {'key': WEATHER_API_KEY, 'q': location}
    try:
        response = requests.get(WEATHER_BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()
        
        return (
            f"Weather in {location}: {data['current']['condition']['text']}\n"
            f"Temperature: {data['current']['temp_c']}°C\n"
            f"Humidity: {data['current']['humidity']}%"
        )
    except Exception as e:
        return f"Weather error: {str(e)}"


def get_current_time(_input: str) -> str:
    return " The current time is: 10:00 PM"

time_tool = Tool(name="GetTime", func=get_current_time, description="Returns the current time.")
work_order_tool = Tool(name="MakeWorkOrder", func=make_work_order, description="Creates a work order from a JSON string.")
weather_tool = Tool(name="GetWeather", func=get_current_weather, description="Returns the current weather for a given location.")
vector_tool = Tool(name="VectorSearch", func=vector_search, description="Searches the FAISS vector database for documents related to the query.")

tools = [weather_tool, time_tool, work_order_tool, vector_tool]

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
        "name": "user"
    }
    try:
        response = agent.run(input_data)
    except Exception as e:
        response = f" An error occurred: {str(e)}"
    
    await cl.Message(content=response).send()
