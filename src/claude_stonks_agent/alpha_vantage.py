import requests
import os
from dataclasses import dataclass
from functools import lru_cache
from datetime import datetime


@dataclass
class SearchResult:
    name: str
    symbol: str
    type: str
    region: str
    currency: str


@dataclass
class TimeSeriesDaily:
    date: str
    date_value: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass
class Overview:
    symbol: str
    name: str
    description: str
    market_cap: float


def _parse_date(date_text: str) -> datetime:
    return datetime.strptime(date_text, "%Y-%m-%d")


class AlphaVantageClient:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError('API key is required')

        self.api_key = api_key
        self.base_url = 'https://www.alphavantage.co/query'

    def search(self, term: str) -> list[SearchResult]:
        params = {
            'function': 'SYMBOL_SEARCH',
            'keywords': term,
            'datatype': 'json',
            'apikey': self.api_key
        }
        response = requests.get(self.base_url, params=params)
        response.raise_for_status()
        response_data = response.json()

        def build_search_result(data: dict) -> SearchResult:
            return SearchResult(
                name=data['2. name'],
                symbol=data['1. symbol'],
                type=data['3. type'],
                region=data['4. region'],
                currency=data['8. currency']
            )

        if 'bestMatches' not in response_data:
            raise Exception(f'Unexpected response: {response_data}')

        return list(map(build_search_result, response_data['bestMatches']))

    def fetch_daily(self, symbol: str) -> list[TimeSeriesDaily]:
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': symbol,
            'outputsize': 'full',
            'datatype': 'json',
            'apikey': self.api_key
        }
        response = requests.get(self.base_url, params=params)
        response.raise_for_status()
        response_data = response.json()

        def build_time_series_daily(date: str, data: dict) -> TimeSeriesDaily:
            return TimeSeriesDaily(
                date=date,
                date_value=_parse_date(date),
                open=float(data['1. open']),
                high=float(data['2. high']),
                low=float(data['3. low']),
                close=float(data['4. close']),
                volume=int(data['5. volume'])
            )

        data: dict[str, dict[str, str]] = response_data['Time Series (Daily)']

        return list(map(lambda item: build_time_series_daily(item[0], item[1]), data.items()))

    def fetch_overiew(self, symbol: str) -> Overview:
        params = {
            'function': 'OVERVIEW',
            'symbol': symbol,
            'datatype': 'json',
            'apikey': self.api_key
        }
        response = requests.get(self.base_url, params=params)
        response.raise_for_status()
        response_data = response.json()

        return Overview(
            symbol=response_data['Symbol'],
            name=response_data['Name'],
            description=response_data['Description'],
            market_cap=float(response_data['MarketCapitalization'])
        )
        


class AlphaVantageService:
    """
    Wrapper on top of the client to provide caching and other features
    """

    @staticmethod
    def create() -> 'AlphaVantageService':
        api_key = os.getenv('ALPHA_VANTAGE_API_KEY')

        if not api_key:
            raise ValueError('ALPHA_VANTAGE_API_KEY env variable is required')

        return AlphaVantageService(AlphaVantageClient(api_key))

    def __init__(self, client: AlphaVantageClient):
        self.client = client

    @lru_cache(maxsize=1024)
    def search(self, term: str) -> list[SearchResult]:
        """Filters to just US equities."""
        def is_us_equity(result: SearchResult) -> bool:
            return result.type == 'Equity' and result.region == 'United States'

        return list(filter(is_us_equity, self.client.search(term)))

    @lru_cache(maxsize=128)
    def fetch_daily(self, symbol: str) -> list[TimeSeriesDaily]:
        return self.client.fetch_daily(symbol)

    @lru_cache(maxsize=1024)
    def overview(self, symbol: str) -> Overview:
        return self.client.fetch_overiew(symbol)


    def latest_price(self, symbol: str) -> float:
        daily = self.fetch_daily(symbol)

        if len(daily) == 0:
            raise ValueError(f'No data found for {symbol}')

        return daily[0].close

    def _find_daily_for_date(self, symbol: str, date: str) -> TimeSeriesDaily:
        daily = self.fetch_daily(symbol)
        date_value = _parse_date(date)

        for item in daily:
            # assume sorted in reverse chonological order
            if item.date_value <= date_value:
                return item

        raise ValueError(f'No data found for {symbol} on {date}')


    def price_on_date(self, symbol: str, date: str) -> float:
        return self._find_daily_for_date(symbol, date).close


    def latest_market_cap(self, symbol: str) -> float:
        overview = self.overview(symbol)

        return overview.market_cap
