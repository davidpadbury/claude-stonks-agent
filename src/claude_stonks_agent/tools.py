from claude_stonks_agent.alpha_vantage import AlphaVantageService, SearchResult
from claude_stonks_agent.claude import XmlBuilder
from langchain_core.tools import StructuredTool, tool
from typing import Optional
from datetime import datetime

def create_alpha_vantage_tools(alpha_vantage: AlphaVantageService) -> list[StructuredTool]:

    def search_for_symbol(term: str) -> Optional[str]:
        """
        Takes a single word search term for a company and looks up its stock symbol.
        """
        results = alpha_vantage.search(term)

        builder = XmlBuilder()
        with builder.tag_with_children('results'):
            for result in results:
                with builder.tag_with_children('result'):
                    builder.tag_with_text('symbol', result.symbol)
                    builder.tag_with_text('name', result.name)

        return str(builder)

    def latest_price(symbol: str) -> float:
        """
        Looks up the latest price for a stock symbol.
        The price returned is in US dollars.
        """
        return alpha_vantage.latest_price(symbol)

    def price_at_date(symbol: str, date: str) -> float:
        """
        Looks up the price of a stock symbol at a specific date (formatted as YYYY-MM-DD e.g. 2021-01-01)
        The price returned is in US dollars.
        """
        return alpha_vantage.price_on_date(symbol, date)

    def latest_market_capitalization(symbol: str) -> float:
        """
        Looks up the latest market capitalization for a stock symbol.
        The value returned is in US dollars.
        """
        return alpha_vantage.latest_market_cap(symbol)


    def current_date() -> str:
        """
        Returns the current date in the format YYYY-MM-DD
        """
        return datetime.now().strftime('%Y-%m-%d')


    return [
        StructuredTool.from_function(search_for_symbol),
        StructuredTool.from_function(latest_price),
        StructuredTool.from_function(price_at_date),
        StructuredTool.from_function(latest_market_capitalization),
        StructuredTool.from_function(current_date)
    ]