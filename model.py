import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

def get_llm(model: str="llama-3.3-70b-versatile"):
    """Supervisor LLM — tool_choice='auto' so Groq doesn't require a tool call every turn."""
    return ChatGroq(
        model=model,
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY"),
    )  # ← key fix for supervisor
