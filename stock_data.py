from alpha_vantage.timeseries import TimeSeries

class StockData:
    """ Interface for interacting with StockData APIs """
    def get_intraday(self, symbol, interval):
        raise NotImplementedError

    def get_daily(self, symbol):
        raise NotImplementedError

class AlphaVantage(StockData):
    def __init__(self, api_key):
        self.ts = TimeSeries(key=api_key, output_format='pandas', indexing_type='date')

    def get_intraday(self, symbol, interval):
        data, meta_data = self.ts.get_intraday(symbol, interval, outputsize='full')
        return data

    def get_daily(self, symbol):
        data, meta_data = self.ts.get_daily(symbol, outputsize='full')
        return data