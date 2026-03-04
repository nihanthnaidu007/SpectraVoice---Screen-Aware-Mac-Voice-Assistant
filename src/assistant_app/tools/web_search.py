"""Web Search Service - Intelligent real-time search using Tavily API."""

import os
import re
from enum import Enum
from typing import NamedTuple

from assistant_app.utils.logging_config import get_logger


class SearchCategory(Enum):
    NONE = "none"
    WEATHER = "weather"
    NEWS = "news"
    PRICE = "price"
    EVENT = "event"
    ENTITY = "entity"
    CURRENT_AFFAIRS = "current_affairs"
    SPORTS = "sports"
    ENTERTAINMENT = "entertainment"
    TECHNOLOGY = "technology"
    GENERAL = "general"


class SearchResult(NamedTuple):
    query: str
    answer: str
    sources: list[str]
    success: bool


class WebSearch:
    """Intelligent web search with auto-detection for real-time queries."""
    
    WEATHER_PATTERNS = frozenset({"weather", "temperature", "forecast", "rain", "snow", "sunny", "cloudy", "humid", "humidity", "wind", "storm", "degrees", "celsius", "fahrenheit", "climate", "uv index", "air quality"})
    NEWS_PATTERNS = frozenset({"news", "headline", "breaking", "update", "announce", "announced", "report", "reported", "happening", "occurred", "incident"})
    PRICE_PATTERNS = frozenset({"price", "cost", "stock", "stocks", "market", "markets", "crypto", "cryptocurrency", "bitcoin", "btc", "ethereum", "eth", "dollar", "euro", "yen", "rupee", "worth", "value", "trading", "invest", "share", "shares", "nasdaq", "dow", "s&p", "nifty", "sensex", "forex", "gold price", "oil price"})
    EVENT_PATTERNS = frozenset({"schedule", "upcoming", "event", "game", "match", "release", "launch", "premiere", "concert", "show", "festival", "conference", "election", "vote", "deadline"})
    ENTITY_PATTERNS = frozenset({"president", "prime minister", "ceo", "cfo", "cto", "founder", "chairman", "director", "mayor", "governor", "senator", "minister", "leader", "chief", "captain", "coach", "manager"})
    SPORTS_PATTERNS = frozenset({"score", "scores", "standings", "ranking", "playoffs", "championship", "tournament", "league", "nba", "nfl", "mlb", "nhl", "soccer", "football", "cricket", "ipl", "world cup", "olympics", "medal", "winner", "champion", "finals"})
    ENTERTAINMENT_PATTERNS = frozenset({"movie", "film", "box office", "streaming", "netflix", "spotify", "album", "song", "chart", "billboard", "grammy", "oscar", "emmy", "award", "celebrity", "actor", "singer", "artist", "band", "tour", "episode", "season"})
    TECH_PATTERNS = frozenset({"iphone", "android", "samsung", "google", "apple", "microsoft", "tesla", "spacex", "ai", "chatgpt", "openai", "nvidia", "amd", "intel", "chip", "gpu", "software", "app", "update", "version", "specs", "review"})
    TIME_ANCHORS = frozenset({"today", "now", "right now", "currently", "this week", "this month", "this year", "2024", "2025", "2026", "yesterday", "tomorrow", "recent", "recently", "latest", "newest", "current", "live", "ongoing"})
    
    CURRENT_ENTITY_REGEX = re.compile(r'\b(who|what|which)\s+(is|are|was|were)\s+(the\s+)?(current|new|latest|present)\s+', re.IGNORECASE)
    LEADER_QUERY_REGEX = re.compile(r'\b(who|what)\s+(is|are)\s+(the\s+)?(president|prime minister|ceo|leader|head|chief|king|queen)\s+(of\s+)?', re.IGNORECASE)
    WON_QUERY_REGEX = re.compile(r'\b(who|which|what)\s+(won|wins|lost)\s+', re.IGNORECASE)
    HAPPENED_QUERY_REGEX = re.compile(r'\b(what|when|how)\s+(happened|is happening|occurred)\s+', re.IGNORECASE)
    STATIC_PATTERNS = frozenset({"explain", "definition", "define", "meaning of", "what does", "how does", "why does", "theory", "concept", "history of", "invented", "discovered", "biography", "equation", "formula", "algorithm", "tutorial", "difference between", "compare", "vs"})
    HISTORICAL_PATTERNS = frozenset({"ancient", "medieval", "renaissance", "century", "bc", "ad", "1800s", "1900s", "world war", "civil war", "revolution", "empire", "dynasty", "era", "historical"})
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.api_key = os.getenv("TAVILY_API_KEY")
        self._client = None
        if not self.api_key:
            self.logger.warning("⚠️ TAVILY_API_KEY not set - web search disabled")

    @property
    def client(self):
        if self._client is None and self.api_key:
            try:
                from tavily import TavilyClient
                self._client = TavilyClient(api_key=self.api_key)
            except Exception as e:
                self.logger.error(f"❌ Tavily init failed: {e}")
        return self._client

    @property
    def is_available(self) -> bool:
        return self.client is not None

    def needs_search(self, query: str) -> tuple[bool, SearchCategory]:
        if not query or not self.is_available:
            return False, SearchCategory.NONE
        
        query_lower = query.lower().strip()
        words = set(re.findall(r'\b\w+\b', query_lower))
        
        # Skip static knowledge queries
        if words & self.STATIC_PATTERNS:
            if not (words & self.TIME_ANCHORS):
                return False, SearchCategory.NONE
        
        if words & self.HISTORICAL_PATTERNS:
            if not (words & self.TIME_ANCHORS):
                return False, SearchCategory.NONE
        
        # Check regex patterns (high priority)
        if self.CURRENT_ENTITY_REGEX.search(query_lower):
            return True, SearchCategory.CURRENT_AFFAIRS
        
        if self.LEADER_QUERY_REGEX.search(query_lower):
            return True, SearchCategory.CURRENT_AFFAIRS
        
        if self.WON_QUERY_REGEX.search(query_lower):
            if words & self.TIME_ANCHORS or words & self.SPORTS_PATTERNS:
                return True, SearchCategory.SPORTS
            if words & self.ENTERTAINMENT_PATTERNS:
                return True, SearchCategory.ENTERTAINMENT
            return True, SearchCategory.CURRENT_AFFAIRS
        
        if self.HAPPENED_QUERY_REGEX.search(query_lower):
            return True, SearchCategory.NEWS
        
        # Check category patterns
        if words & self.WEATHER_PATTERNS:
            return True, SearchCategory.WEATHER
        
        if words & self.PRICE_PATTERNS:
            return True, SearchCategory.PRICE
        
        if words & self.SPORTS_PATTERNS:
            return True, SearchCategory.SPORTS
        
        if words & self.ENTERTAINMENT_PATTERNS:
            return True, SearchCategory.ENTERTAINMENT
        
        if words & self.TECH_PATTERNS:
            if words & self.TIME_ANCHORS:
                return True, SearchCategory.TECHNOLOGY
        
        if words & self.EVENT_PATTERNS:
            return True, SearchCategory.EVENT
        
        if words & self.ENTITY_PATTERNS:
            if "who" in words or words & self.TIME_ANCHORS:
                return True, SearchCategory.ENTITY
        
        if words & self.NEWS_PATTERNS:
            return True, SearchCategory.NEWS
        
        # Time anchor + question = likely needs search
        if words & self.TIME_ANCHORS:
            question_words = {"who", "what", "when", "where", "how", "which", "why"}
            if words & question_words:
                return True, SearchCategory.GENERAL
        
        return False, SearchCategory.NONE
    
    def search(self, query: str, category: SearchCategory = SearchCategory.GENERAL) -> SearchResult:
        if not self.is_available:
            return SearchResult(query=query, answer="", sources=[], success=False)
        
        try:
            search_params = self._get_search_params(query, category)
            self.logger.info(f"🔍 Searching ({category.value}): {query[:50]}...")
            
            response = self.client.search(**search_params)
            
            answer = response.get("answer", "")
            results = response.get("results", [])
            sources = [r.get("url", "") for r in results[:5] if r.get("url")]
            
            if not answer and results:
                snippets = []
                for r in results[:3]:
                    content = r.get("content", "")
                    if content:
                        snippets.append(content[:300])
                answer = " ".join(snippets)
            
            if category == SearchCategory.PRICE and results:
                extra_data = []
                for r in results[:3]:
                    title = r.get("title", "")
                    content = r.get("content", "")[:200]
                    if title or content:
                        extra_data.append(f"{title}: {content}")
                if extra_data:
                    answer = f"{answer}\n\nAdditional data:\n" + "\n".join(extra_data)
            
            self.logger.info(f"✅ Search complete: {len(sources)} sources")
            
            return SearchResult(query=query, answer=answer, sources=sources, success=True)
            
        except Exception as e:
            self.logger.error(f"❌ Search error: {e}")
            return SearchResult(query=query, answer="", sources=[], success=False)
    
    def _get_search_params(self, query: str, category: SearchCategory) -> dict:
        params = {"query": query, "include_answer": True, "include_raw_content": False}
        if category == SearchCategory.PRICE:
            q = query.lower()
            if "stock" in q and "today" not in q:
                query = f"{query} today live price"
            elif any(c in q for c in ["bitcoin", "ethereum", "crypto"]) and "current" not in q:
                query = f"{query} current price USD"
            params.update({"query": query, "search_depth": "advanced", "max_results": 5, "include_raw_content": True})
        elif category in (SearchCategory.CURRENT_AFFAIRS, SearchCategory.ENTITY, SearchCategory.SPORTS, SearchCategory.NEWS):
            params.update({"search_depth": "advanced", "max_results": 5})
        else:
            params.update({"search_depth": "basic", "max_results": 3})
        return params

    def get_context(self, query: str) -> str:
        needs, category = self.needs_search(query)
        
        if not needs:
            return ""
        
        result = self.search(query, category)
        
        if not result.success or not result.answer:
            return ""
        
        context = f"""
[LIVE WEB SEARCH RESULT - {category.value.upper()}]
This is real-time data retrieved just now. Use this as the authoritative source for current/recent information.

{result.answer}
"""
        
        if result.sources:
            context += f"\nSources: {', '.join(result.sources[:2])}"
        
        context += "\n[END OF LIVE DATA]\n"
        
        return context
