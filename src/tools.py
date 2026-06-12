from langchain_core.tools import tool
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.utilities.arxiv import ArxivAPIWrapper
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities import SerpAPIWrapper
import requests, os

from dotenv import load_dotenv

load_dotenv()

arxiv = ArxivQueryRun(api_wrapper=ArxivAPIWrapper(top_k_results=2, doc_content_chars_max=500))
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500))
tavily = TavilySearchResults()

@tool
def get_weather(city: str) -> dict:
    """Get current weather for a given city."""
    api_key = os.getenv("OPEN_WEATHER_API")
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    data = response.json()
    if response.status_code != 200:
        return {"error": data.get("message", "Error")}
    return {
        "city": data["name"],
        "temperature": data["main"]["temp"],
        "humidity": data["main"]["humidity"],
        "weather": data["weather"][0]["description"],
        "wind_speed": data["wind"]["speed"]
    }
@tool
def google_search(query: str) -> str:
    """
    Search Google for real-time information, facts, and current events.
    Use this tool whenever you need to find answers from the internet.
    """
    search = SerpAPIWrapper()
    return search.run(query)
tools = [arxiv, tavily, wikipedia, get_weather,google_search]