import os
from dotenv import load_dotenv
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.utilities.arxiv import ArxivAPIWrapper
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_tavily import TavilySearch
load_dotenv()


api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=2, doc_content_chars_max=500)
arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv)

# Wikipedia
api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
wikipedia = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)

# Tavily
tavily = TavilySearchResults(api_key=os.getenv("TAVILY_API_KEY"))

tools = [arxiv, tavily, wikipedia]
