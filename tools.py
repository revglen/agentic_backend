import os
import requests
from dotenv import load_dotenv

load_dotenv()

from langchain_tavily import TavilySearch
#from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.tools import tool
from backend.vectorstore import VectorStore

#_search = DuckDuckGoSearchResults(output_format="list", num_results=0)

os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
_search = TavilySearch(max_results=5)
TAVILY_URL = "https://api.tavily.com/search"

@tool
def web_search(query: str) -> str:
    """Search the live web for current information — e.g. RERA/MHADA approval
    status, developer track record, or market rates for redevelopment deals
    in a given Mumbai locality. Returns a list of results with titles, links,
    and snippets."""

    try:
        response = requests.post(
            TAVILY_URL,
            json={
                "api_key": TAVILY_API_KEY,
                "query": query,
                "max_results": 5,
            },
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        return f"Search temporarily unavailable: {e}"

    results = data.get("results", [])
    if not results:
        return "No results found"

    formatted = []
    for r in results:
        title = r.get("title", "")
        link = r.get("url", "")
        snippet = r.get("content", "")
        formatted.append(f"Title: {title}\nURL: {link}\nSnippet: {snippet}")

    return "\n\n".join(formatted)

@tool
def retrieve_society_docs(query: str) -> str:
    """Retrieve the most relevant chunks from the society's own uploaded
    documents (agreements, bylaws, minutes, developer proposals) for a given
    query. Use this before answering any question about the society's own
    paperwork."""

    vectorstore: VectorStore = VectorStore()    
    retriever=vectorstore.get_retriever()

    if retriever is None:
        return "No documents have been uploaded to the society knowledge base yet."

    docs = retriever.invoke(query)
    if not docs:
        return "No relevant content found in uploaded society documents."
   
    return vectorstore.format_docs(docs)

ALL_TOOLS = [web_search, retrieve_society_docs]
