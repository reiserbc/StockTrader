import yfinance
import sys

class StockDataPuller:
    """ Interface for interacting with StockData APIs """
    def get_intraday(self, symbol, interval):
        raise NotImplementedError

    def get_daily(self, symbol):
        raise NotImplementedError

    def get_current_price(self, symbol):
        raise NotImplementedError

def rename_ohlcv_columns(df):
    df.columns = [ 'open', 'high', 'low', 'close', 'volume']
    return df


class YahooFinance(StockDataPuller):
    def get_intraday(self, symbol, interval, timesteps):
        data = yfinance.download(tickers=symbol, period="7d", interval=interval, auto_adjust=True)  

        # reverse so that latest is at index 0
        data = data.iloc[::-1]
        
        # take timesteps num recent results
        data = data[0: timesteps]

        return rename_ohlcv_columns(data)

    def get_daily(self, symbol):
        data = yfinance.download(tickers=symbol, interval="1d", auto_adjust=True)
        # reverse so that latest is at index 0
        data = data.iloc[::-1]  
        return rename_ohlcv_columns(data)
    
    def get_current_price(self, symbol):
        data = yfinance.download(tickers=symbol, period="1d", interval='1m', auto_adjust=True) 
        # reverse so that latest is at index 0
        data = data.iloc[::-1] 
        last_close = rename_ohlcv_columns(data)['close'][0]
        return last_close
        
if __name__ == '__main__':
    try:
        symbol = sys.argv[1]
        interval = sys.argv[2]
        save_path = sys.argv[3]

        yf = YahooFinance()
        df = yf.get_intraday(symbol, interval)
        df.to_csv(save_path, index=False)
        print("Saved to \"{}\"!\n".format(save_path))
        
    except:
        print("argv[1] = symbol\nargv[2] = interval\nargv[3] = save_path")

