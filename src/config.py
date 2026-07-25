from langchain_groq import ChatGroq
import os 
from dotenv import load_dotenv 

load_dotenv()


def get_required_api_key(name: str, expected_prefix: str) -> str:
    key = os.getenv(name, "").strip().strip("\"'")
    if not key:
        raise RuntimeError(f"{name} is missing. Add it to your .env file.")
    if not key.startswith(expected_prefix):
        raise RuntimeError(
            f"{name} must start with {expected_prefix!r}. "
            "Create a new Groq API key and update your .env file."
        )
    return key


llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=get_required_api_key("GROQ_API_KEY", "gsk_"),
    temperature=0.2,
)
