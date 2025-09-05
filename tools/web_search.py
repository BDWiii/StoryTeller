import os
from dotenv import load_dotenv
from typing import Optional, Dict, List, Literal
from langchain.tools import tool
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from pydantic import BaseModel, Field


# ====================== Web Search Tool =======================
class SearchInput(BaseModel):
    query: str = Field(..., description="The search query.")
    max_results: Optional[int] = Field(3, description="Number of results to return.")
    include_raw_content: Optional[bool] = Field(
        False, description="Flag to return more content"
    )  # only works for paid plans.


@tool(args_schema=SearchInput)
def search_web(
    query: str, max_results: int = 3, include_raw_content: bool = False
) -> List[Dict]:
    """
    Search the web for real time information and return the top results.
    """
    try:
        load_dotenv()
        search = TavilySearchAPIWrapper(tavily_api_key=os.getenv("TAVILY_API_KEY"))
        results = search.results(
            query=query,
            max_results=max_results,
            include_raw_content=include_raw_content,
        )
        return results
    except Exception as e:
        return [{"ERROR": str(e)}]
